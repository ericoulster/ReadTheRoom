import cv2
import mediapipe as mp
import numpy as np
import time
from mss import mss

mp_face_detection = mp.solutions.face_detection

face_detection = mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.3)

mp_drawing = mp.solutions.drawing_utils

bounding_box = {'top': 0, 'left': 0, 'width': 1200, 'height': 1200}

sct = mss()

# points include: 
    # LEFT_EAR_TRAGION
    # LEFT_EYE
    # MOUTH_CENTRE
    # NOSE TIP
    # RIGHT_EAR_TRAGION
    # RIGHT_EYE


while True:

    image = np.array(sct.grab(bounding_box))
    start = time.time()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image.flags.writeable = False

    results = face_detection.process(image)

    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []
    detection_list = []
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(image, detection)
            detection_list.append(detection)
    #You have to get a count of total number of face objects
    # You have to calculate a total 
    
    total_faces = len(detection_list)

    

    if len(detection_list) > 0:
        for i in detection_list:
            foo = mp_face_detection.get_key_point(
          i, mp_face_detection.FaceKeyPoint.NOSE_TIP)

    

    cv2.imshow('Face Detection', image)

    
    if cv2.waitKey(5) & 0xFF == 27:
        break

cv2.destroyAllWindows()