import torch
import droid_backends
import numpy as np

def test_corr_index_forward():
  h1=30
  w1=40
  h2=h1
  w2=w1
  b=1

  radius=3
  volume=torch.tensor(np.ndarray([b,h1,w1,h2,w2],dtype=np.float32))
  coords=torch.tensor(np.ndarray([b,2,h1,w1],dtype=np.float32))
  #corr=torch.tensor(np.ndarray([b,2*radius+1,2*radius+1,h1,w1]))

  # volume is expected to be BxH1xW1xH2xW2, where the values are the correlation values.
  # coords is expected to be BxCxH1xW1, where C is the coordinates of the frame 1 points projected in frame 2.
  # corr is expected to be Bx(2*R+1)x(2*R+1)xH1xW1
  corr=droid_backends.corr_index_forward(volume,coords,radius)
  corr=np.array(corr)

  print(corr.shape,corr.dtype)

if __name__=="__main__":
  test_corr_index_forward()
