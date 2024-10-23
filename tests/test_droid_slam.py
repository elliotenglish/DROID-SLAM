import cv2
import csv
import os
import pytest
import numpy as np
from types import SimpleNamespace
import droid_slam.droid
import droid_slam.utilities
import cyborg.utilities.input_stream

def get_test_data(max_frames):
  return cyborg.utilities.input_stream.get_input_stream(
    os.path.join(os.path.dirname(__file__),"../data/mav0/cam0/data.csv"),max_frames)

def compute_results(device):
  system=None

  for frame_idx,frame_timestamp,frame in get_test_data(max_frames=100):
    print("------------------------------------------------")
    print(f"i={frame_idx} t={frame_timestamp} s={frame.shape}")

    frame_tensor=droid_slam.utilities.image_to_tensor(frame)
    intrinsics_tensor=droid_slam.utilities.intrinsics_to_tensor(frame,*droid_slam.utilities.get_default_intrinsics(frame.shape[2],frame.shape[1]))
    # print(frame_tensor.shape)

    if not system:
      params=droid_slam.utilities.get_default_params()
      params["image_size"]=[frame_tensor.shape[1],frame_tensor.shape[2]]
      params["device"]=device
      system=droid_slam.droid.Droid(SimpleNamespace(**params))

    frame_in=frame_tensor[None]
    system.track(frame_timestamp,frame_in,None,intrinsics_tensor)

    try:
      kf_idx=system.video.counter.value
      camera_state=system.video.poses[kf_idx-1].cpu().numpy().tolist()
      print(f"kf_idx={kf_idx} camera_state={camera_state}")
    except:
      break

    # cv2.imshow("frame",frame)
    # cv2.waitKey(1)

  results=system.video.poses[:system.video.counter.value].cpu().numpy()
  return results

def test_droid_slam(device="cuda"):
  results_cuda=compute_results(device)

  groundtruth_path=os.path.join(os.path.dirname(__file__),"test_groid_slam.groundtruth.npy")
  if os.path.exists(groundtruth_path):
    results_groundtruth=np.load(groundtruth_path)
  else:
    print("groundtruth results missing, writing out current results")
    np.save(groundtruth_path,results_cuda)
    results_groundtruth=results_cuda

  error_cuda=(results_cuda-results_groundtruth)
  error_linf=np.abs(error_cuda).max()
  print(f"error_linf={error_linf}")

if __name__=="__main__":
  import argparse
  parser=argparse.ArgumentParser()
  parser.add_argument("--device",default="cuda")
  parser.add_argument("--max_frames",type=int,default=100)
  args=parser.parse_args()
  test_droid_slam(device=args.device,max_frames=args.max_frames)
  #pytest.main(["-x",__file__,"-s"])
