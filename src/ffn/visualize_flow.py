import numpy as np
import cv2
from flow_vis import flow_to_color

path = '/home/aditya/llve/StableLLVE/uw_video/flow/pool_filpper_001_A_0829.npy'

flow = np.load(path)
flow_color = flow_to_color(flow, convert_to_bgr=True)
cv2.imwrite('flow.png', flow_color)
