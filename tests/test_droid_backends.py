import torch
import droid_backends
import numpy as np
import math
import pytest
import droid_slam.utilities

tolerance=1e-4

def within_bounds(h,w,H,W):
  return h >= 0 and h < H and w >= 0 and w < W

def compute_stats(x):
  return {"shape":x.shape,
          "mean":float(x.sum()/x.size),
          "stddev":float(x.std()),
          "min":float(x.min()),
          "max":float(x.max()),
          "nnz":int((x!=0).sum())}

def evaluate_compare_cpu_gpu(get_values_func,prints=False):
  vals_cpu=get_values_func(device="cpu",prints=prints).cpu().numpy()
  if prints: print("vals_cpu",vals_cpu)
  vals_gpu=get_values_func(device="cuda",prints=prints).cpu().numpy()
  if prints: print("vals_gpu",vals_gpu)
  error=vals_cpu-vals_gpu
  error_linf=np.abs(error).max()
  print("vals_gpu stats",compute_stats(vals_gpu))
  print("vals_cpu stats",compute_stats(vals_cpu))
  print(f"error_linf={error_linf}")
  assert (error_linf<tolerance).all()

def corr_index_forward_reference(volume,coords,radius):
  N,h1,w1,h2,w2=volume.shape
  r=radius

  rd=2*r+1

  corr=np.ndarray([N,rd,rd,h1,w1],dtype=np.float32)
  corr[...]=0

  # cnt=0

  n=0
  for x in range(w1):
    for y in range(h1):

      x0=coords[n,0,y,x]
      y0=coords[n,1,y,x]

      x0f=int(math.floor(x0))
      y0f=int(math.floor(y0))

      dx=x0-x0f
      dy=y0-y0f

      # print(dx,dy)

      for i in range(rd+1):
        for j in range(rd+1):
          x1=x0f - r + i
          y1=y0f - r + j

          # cnt+=1

          if within_bounds(y1,x1,h2,w2):
            s=volume[n,y,x,y1,x1]
            #print(s)

            if i > 0 and j > 0:
              corr[n,i-1,j-1,y,x] += s * (dx * dy)

            if i > 0 and j < rd:
              corr[n,i-1,j,y,x] += s * (dx * (1-dy))

            if i < rd and j > 0:
              corr[n,i,j-1,y,x] += s * ((1-dx) * dy)

            if i < rd and j < rd:
              corr[n,i,j,y,x] += s * ((1-dx) * (1-dy))

  # print("cnt",cnt)

  return torch.tensor(corr)

def test_corr_index_forward():
  def get_values(device,reference=False,prints=False):
    h1=3
    w1=4
    h2=h1
    w2=w1
    b=1

    gen=np.random.default_rng(5432)

    radius=0
    volume=torch.tensor(gen.uniform(-1,1,size=[b,h1,w1,h2,w2]).astype(np.float32))
    coords=torch.tensor(np.concatenate(
      [
        gen.uniform(0,w2-1,size=[b,1,h1,w1]).astype(np.float32),
        gen.uniform(0,h2-1,size=[b,1,h1,w1]).astype(np.float32)
      ],
      axis=1))
    # coords=coords*0+1
    #corr=torch.tensor(np.ndarray([b,2*radius+1,2*radius+1,h1,w1]))

    if prints:
      print("volume",volume.shape,volume.dtype,volume)
      print("coords",coords.shape,coords.dtype,coords)
      print("radius",radius)

    cpu_device = torch.device("cpu")
    compute_device = torch.device(device)
    volume=volume.to(device=compute_device)
    coords=coords.to(device=compute_device)

    # volume is expected to be BxH1xW1xH2xW2, where the values are the correlation values.
    # coords is expected to be BxCxH1xW1, where C is the coordinates of the frame 1 points projected in frame 2. The coordinates are x,y order rather than the y,x ordered used
    # corr is expected to be Bx(2*R+1)x(2*R+1)xH1xW1

    if reference:
      corr=corr_index_forward_reference(volume,coords,radius)
    else:
      corr=droid_backends.corr_index_forward(volume,coords,radius)[0]

    if prints:
      print("corr",corr.shape,corr.dtype,corr)

    return corr

  evaluate_compare_cpu_gpu(get_values)

def generate_poses(gen,num):
  poses=gen.uniform(-1,1,size=[num,7])
  poses[:,3:]/=np.linalg.norm(poses[:,3:],axis=1)[...,None]
  return poses

def test_projective_transform():
  pass

def test_projmap():
  pass

def test_frame_distance():
  pass

def test_depth_filter():
  def get_values(device,prints=False):
    num=4
    w,h=8,6
    ni=3
    gen=np.random.default_rng(5432)
    poses=torch.tensor(generate_poses(gen,num).astype(np.float32)).to(device=device)
    disps=torch.tensor(gen.uniform(.1,.2,size=[num,h,w]).astype(np.float32)).to(device=device)
    intrinsics=torch.tensor(np.array(droid_slam.utilities.get_default_intrinsics(w,h)).astype(np.float32)).to(device=device)
    inds=torch.tensor(np.array([0,2]).astype(np.long)).to(device=device)
    thresh=torch.tensor(np.array([.1]*ni).astype(np.float32)).to(device=device)

    counter=droid_backends.depth_filter(poses,disps,intrinsics,inds,thresh)
    #print(counter)
    return counter
  
  evaluate_compare_cpu_gpu(get_values)

def test_iproj():
  def get_values(device,prints=False):
    num=9
    w,h=16,12
    gen=np.random.default_rng(5432)
    poses=torch.tensor(generate_poses(gen,num).astype(np.float32)).to(device=device)
    disps=torch.tensor(gen.uniform(.1,.2,size=[num,h,w]).astype(np.float32)).to(device=device)
    intrinsics=torch.tensor(np.array(droid_slam.utilities.get_default_intrinsics(w,h)).astype(np.float32)).to(device=device)

    points=droid_backends.iproj(poses,disps,intrinsics)
    return points

  evaluate_compare_cpu_gpu(get_values)

def test_accum():
  def get_values(device,prints=False):
    n=21
    m=9
    gen=np.random.default_rng(5432)
    inps=torch.tensor(gen.uniform(-1,1,size=[n,m]).astype(np.float32)).to(device=device)
    idxs=torch.tensor(np.array([0,4,3,5,6,7,8,1,10,16,19]).astype(np.long)).to(device=device)
    ptrs=torch.tensor(np.array([0,3,7,10]).astype(np.long)).to(device=device)

    out=droid_backends.accum(inps,ptrs,idxs)
    return out

  evaluate_compare_cpu_gpu(get_values)

def test_pose_retr():
  def get_values(device,prints=False):
    num=5
    gen=np.random.default_rng(5432)
    poses0=torch.tensor(generate_poses(gen,num).astype(np.float32)).to(device=device)
    dx=torch.tensor(gen.uniform(-1,1,size=[2,6]).astype(np.float32)).to(device=device)

    poses=poses0.clone().detach()
    droid_backends.pose_retr(poses,dx,1,3);
    assert (poses!=poses0).any()

    return poses
  
  evaluate_compare_cpu_gpu(get_values)

def test_disp_retr():
  def get_values(device,prints=False):
    num=5
    w=16
    h=12
    ni=2
    gen=np.random.default_rng(5432)
    disps0=torch.tensor(gen.uniform(-1,1,size=[num,h,w]).astype(np.float32)).to(device=device)
    dz=torch.tensor(gen.uniform(-1,1,size=[ni,w*h]).astype(np.float32)).to(device=device)
    idxs=torch.tensor(np.array([3,1]).astype(np.long)).to(device=device)
    
    disps=disps0.clone().detach()
    droid_backends.disp_retr(disps,dz,idxs)

    assert (disps!=disps0).any()

    return disps

  evaluate_compare_cpu_gpu(get_values)

def test_EEt6x6():
  def get_values(device,prints=False):
    num=5
    nk=4
    D=3
    gen=np.random.default_rng(5432)
    E=torch.tensor(gen.uniform(-1,1,size=[num,6,D]).astype(np.float32)).to(device)
    Q=torch.tensor(gen.uniform(-1,1,size=[nk,D]).astype(np.float32)).to(device)
    idx=torch.tensor(np.array([[0,0,1],[1,2,3]]).astype(np.long)).to(device)
    S=torch.tensor(np.zeros([num,6,6]).astype(np.float32)).to(device=device)

    droid_backends.EEt6x6(E,Q,idx,S)
    if prints: print(S)
    return S
  
  evaluate_compare_cpu_gpu(get_values)

def test_Ev6x1():
  def get_values(device,prints=False):
    num=5
    nk=4
    D=3
    gen=np.random.default_rng(5432)
    E=torch.tensor(gen.uniform(-1,1,size=[num,6,D]).astype(np.float32)).to(device)
    Q=torch.tensor(gen.uniform(-1,1,size=[nk,D]).astype(np.float32)).to(device)
    w=torch.tensor(gen.uniform(-1,1,size=[nk,D]).astype(np.float32)).to(device)
    idx=torch.tensor(np.array([[0,1],[1,2]]).astype(np.long)).to(device)
    v=torch.tensor(np.zeros([num,6]).astype(np.float32)).to(device=device)

    droid_backends.Ev6x1(E,Q,w,idx,v)
    if prints: print(v)
    return v

  evaluate_compare_cpu_gpu(get_values)

def test_EvT6x1():
  def get_values(device,prints=False):
    num=5
    gen=np.random.default_rng(5432)
    ni=4
    D=3
    E=torch.tensor(gen.uniform(-1,1,size=[num,6,D]).astype(np.float32)).to(device)
    x=torch.tensor(gen.uniform(-1,1,size=[ni,6]).astype(np.float32)).to(device)
    idx=torch.tensor(np.array([2,1,3,0,1]).astype(np.long)).to(device)
    w=torch.tensor(np.zeros([num,D]).astype(np.float32)).to(device=device)

    droid_backends.EvT6x1(E,x,idx,w)
    if prints: print(w)
    return w

  evaluate_compare_cpu_gpu(get_values)

if __name__=="__main__":
  pytest.main(["-x",__file__,"-s"])
