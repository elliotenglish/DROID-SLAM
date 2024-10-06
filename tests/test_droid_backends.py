import torch
#import droid_backends
import numpy as np
import math

def within_bounds(y,x,h,w):
  return y>0 and y<h and x>0 and x<w

def corr_index_forward(volume,coords,radius):
  N,h1,w1,h2,w2=volume.shape
  r=radius

  rd=2*r+1

  corr=np.ndarray([N,rd,rd,h1,w2],dtype=np.float32)

  n=0
  for x in range(w1):
    for y in range(h1):

      x0=coords[n,0,y,x]
      y0=coords[n,1,y,x]

      x0f=math.floor(x0)
      y0f=math.floor(y0)

      dx=x0-x0f
      dy=y0-y0f

      for i in range(rd+1):
        for j in range(rd+1):
          x1=x0f-r+i
          y1=y0f-r+j

          if within_bounds(y1,x1,h2,w2):
            s=volume[n,y,x,y1,x1]
            if i>0 and j>0: corr[n,i-1,j-1,y,x]+=s*(dx*dy)
            if i>0 and j<rd: corr[n,i-1,j,y,x]+=s*(dx*(1-dy))
            if i<rd and j>0: corr[n,i,j-1,y,x]+=s*((1-dx)*dy)
            if i<rd and j<rd: corr[n,i,j,y,x]+=s*((1-dx)*(1-dy))

  return corr

def test_corr_index_forward():
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

  print("volume",volume.shape,volume.dtype,volume)
  print("coords",coords.shape,coords.dtype,coords)
  print("radius",radius)

  cpu_device = torch.device("cpu")
  #gpu_device = torch.device("cuda")
  #volume=volume.to(device=gpu_device)
  #coords=coords.to(device=gpu_device)

  # volume is expected to be BxH1xW1xH2xW2, where the values are the correlation values.
  # coords is expected to be BxCxH1xW1, where C is the coordinates of the frame 1 points projected in frame 2. The coordinates are x,y order rather than the y,x ordered used
  # corr is expected to be Bx(2*R+1)x(2*R+1)xH1xW1
  
  #corr=droid_backends.corr_index_forward(volume,coords,radius)[0]
  corr=corr_index_forward(volume,coords,radius)[0]
  
  #corr=corr.to(device=cpu_device)
  corr=np.array(corr)

  print(corr)
  print(corr.shape,corr.dtype)

if __name__=="__main__":
  test_corr_index_forward()
