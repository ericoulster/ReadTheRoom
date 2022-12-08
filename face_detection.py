import cv2
import mediapipe as mp
import numpy as np
import time
from mss import mss

import streamlit as st
from screeninfo import get_monitors

from mesh_direct import mesh_direct

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_faces=1)


mp_face_detection = mp.solutions.face_detection

face_detection = mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.3)

mp_drawing = mp.solutions.drawing_utils

# Gets the dimensions of monitors. You can set which monitor to use here.
monitors = [m for m in get_monitors()]

first_monitor = monitors[0]

bounding_box = {'top': 0, 'left': 0, 'width': first_monitor.width, 'height': first_monitor.height}

sct = mss()

st.title("ReadTheRoom")

output = st.empty()

# points include: 
    # LEFT_EAR_TRAGION
    # LEFT_EYE
    # MOUTH_CENTRE
    # NOSE TIP
    # RIGHT_EAR_TRAGION
    # RIGHT_EYE

def calc_distance(p1, p2):
    """
    Basic Euclidean Distance Calculation, while changing the data-types of input.
    """
    x1 = float(p1.x)
    x2 = float(p2.x)
    y1 = float(p1.y)
    y2 = float(p2.y)
    return np.sqrt(((x2 - x1)**2) + ((y2 - y1)**2))


def calc_attention(left_eye_nose, right_eye_nose, left_ear_nose, right_ear_nose):
    eye_gaps = min(left_eye_nose, right_eye_nose)
    ear_gaps = min(left_ear_nose, right_ear_nose)
    # if the ear-eye value is smaller than this fraction, you can assume someone is looking to the side.
    # This value was dialed in manually.
    if eye_gaps > ear_gaps:
        return 1
    else:
        return 0

while True:

    image = np.array(sct.grab(bounding_box))
    start = time.time()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image.flags.writeable = False

    results = face_detection.process(image)

    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


    detection_list = []
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(image, detection)
            detection_list.append(detection)
    #You have to get a count of total number of face objects
    # You have to calculate a total 

    people_list = []

    if len(detection_list) > 0:
        for i in detection_list:
            
            box = i.location_data.relative_bounding_box

            mini_box = {'top': round(box.ymin * (first_monitor.height * 0.9)), 'left' : round(box.xmin * (first_monitor.width * 0.9)), 
            'width': round(box.width * (first_monitor.width * 1.1)), 'height': round(box.height * (first_monitor.height * 1.1))}
            attention = mesh_direct(face_mesh, sct, mini_box)
            
    paying_attention = sum(people_list)
    
    total_faces= len(people_list)

    # cv2.imshow('Face Detection', image)

    attention_status = "paying attention: " + str(paying_attention) + "\n Total: " + str(total_faces)


    # attention_status_list = [float(paying_attention), float(total_faces)]
    
    with output.container():
        st.write(attention_status)

    #print("paying attention: " + str(paying_attention) + "\n Total: " + str(total_faces))
    if cv2.waitKey(5) & 0xFF == 27:
        break

cv2.destroyAllWindows()