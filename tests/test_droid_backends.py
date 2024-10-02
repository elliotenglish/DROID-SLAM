import torch
import droid_backends
import numpy as np

def test_corr_index_forward():
  volume=torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]))
  coords=torch.tensor(np.array([[0,0]]))
  radius=5

  result=droid_backends.corr_index_forward(volume,coords,radius)

  print(result)

if __name__=="__main__":
  test_corr_index_forward()
