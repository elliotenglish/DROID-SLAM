import numpy as np
import torch
import cv2
import os

target_width=512
target_height=384

def get_target_shape(image):
  h0, w0, _ = image.shape
  h1 = int(h0 * np.sqrt((target_height * target_width) / (h0 * w0)))
  w1 = int(w0 * np.sqrt((target_height * target_width) / (h0 * w0)))
  return h0,w0,h1,w1

def image_to_tensor(image):
  # if len(calib) > 4:
  #     K = np.eye(3)
  #     K[0,0] = fx
  #     K[0,2] = cx
  #     K[1,1] = fy
  #     K[1,2] = cy
  #     image = cv2.undistort(image, K, calib[4:])
  _,_,h1,w1=get_target_shape(image)

  image = cv2.resize(image, (w1, h1))
  image = image[:h1-h1%8, :w1-w1%8]
  image = torch.as_tensor(image).permute(2, 0, 1)

  return image

def intrinsics_to_tensor(image,fx,fy,cx,cy):
  h0,w0,h1,w1=get_target_shape(image)

  intrinsics = torch.as_tensor([fx, fy, cx, cy],dtype=torch.float32)
  intrinsics[0::2] *= (float(w1) / w0)
  intrinsics[1::2] *= (float(h1) / h0)

  return intrinsics

def get_default_params():
  params={
    "weights":os.path.join(os.path.dirname(__file__),"../droid.pth"),
    "buffer":512,
    "image_size":[384,512],
    "beta":0.3,
    "filter_thresh":2.4,
    "warmup":8,
    "keyframe_thresh":4.0,
    "frontend_thresh":16.0,
    "frontend_window":25,
    "frontend_radius":2,
    "frontend_nms":1,
    "backend_thresh":22.0,
    "backend_radius":2,
    "backend_nms":3,
    "upsample":False,
    "disable_vis":True,
    "stereo":False
  }
  return params
