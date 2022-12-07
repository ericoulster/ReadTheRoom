import cv2
import mediapipe as mp
import numpy as np
import time
from mss import mss
from scipy.spatial.distance import cdist


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

def calc_distance(p1, p2):
    """
    Basic Euclidean Distance Calculation, while changing the data-types of input.
    """
    x1 = float(p1.x)
    x2 = float(p2.x)
    y1 = float(p1.y)
    y2 = float(p2.y)
    return np.sqrt(((x2 - x1)**2) + ((y2 - y1)**2))


def calc_attention(eye, left, right):
    side_gaps = min(left, right)
    # if the ear-eye value is smaller than this fraction, you can assume someone is looking to the side.
    # This value was dialed in manually.
    eye_gap_threshold = eye / 2

    if side_gaps > eye_gap_threshold:
        return True
    else:
        return False

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
    
    #total_faces = len(detection_list)
    #person_id = 0

    people_list = []

    if len(detection_list) > 0:
        for i in detection_list:
            l_ear = mp_face_detection.get_key_point(
            i, mp_face_detection.FaceKeyPoint.LEFT_EAR_TRAGION)

            l_eye= mp_face_detection.get_key_point(
            i, mp_face_detection.FaceKeyPoint.LEFT_EYE)

            r_eye= mp_face_detection.get_key_point(
            i, mp_face_detection.FaceKeyPoint.RIGHT_EYE)
    
            r_ear= mp_face_detection.get_key_point(
            i, mp_face_detection.FaceKeyPoint.RIGHT_EAR_TRAGION)
            
            eye_gap = calc_distance(l_eye, r_eye)
            left_gap = calc_distance(l_ear, l_eye)
            right_gap = calc_distance(r_ear, r_eye)
            # You can use the X & Y  of the bounding box below to detect the bounding box of each person 
            # This can single indivduals out for not paying attention
            # relative_location = i.location_data.relative_bounding_box

            is_attentive = calc_attention(eye_gap, left_gap, right_gap)
            # If you want an index for singling out people, use a dict
            # people_list.append({'person_id': person_id, 'is_attentive': is_attentive})

            #Otherwise, use a list:
            people_list.append(is_attentive)
            # Increment value for next person in loop
            # person_id += 1

    paying_attention = sum(people_list)
    
    total = len(people_list)

    cv2.imshow('Face Detection', image)

    
    if cv2.waitKey(5) & 0xFF == 27:
        break

cv2.destroyAllWindows()