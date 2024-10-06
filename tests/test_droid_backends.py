import torch
import droid_backends
import numpy as np

def test_corr_index_forward():
  h1=3
  w1=4
  h2=h1
  w2=w1
  b=1

  gen=np.random.default_rng(5432)

  radius=0
  volume=torch.tensor(gen.uniform(-1,1,size=[b,h1,w1,h2,w2]).astype(np.float32))
  coords=torch.tensor(np.concat(
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
  gpu_device = torch.device("cuda")
  volume=volume.to(device=gpu_device)
  coords=coords.to(device=gpu_device)

  # volume is expected to be BxH1xW1xH2xW2, where the values are the correlation values.
  # coords is expected to be BxCxH1xW1, where C is the coordinates of the frame 1 points projected in frame 2. The coordinates are x,y order rather than the y,x ordered used
  # corr is expected to be Bx(2*R+1)x(2*R+1)xH1xW1
  corr=droid_backends.corr_index_forward(volume,coords,radius)[0]
  corr=corr.to(device=cpu_device)
  corr=np.array(corr)

  print(corr)
  print(corr.shape,corr.dtype)

if __name__=="__main__":
  test_corr_index_forward()
