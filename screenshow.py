import numpy as np
import cv2
from mss import mss
from PIL import Image

bounding_box = {'top': 0, 'left': 0, 'width': 1200, 'height': 1200}

sct = mss()

while True:
    sct_img = sct.grab(bounding_box)
    cv2.imshow('screen', np.array(sct_img))

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break
