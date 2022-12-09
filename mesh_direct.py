import cv2
import mediapipe as mp
import numpy as np
import time
from mss import mss

# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_faces=1)
# sct = mss()

def mesh_direct(face_mesh, sct, bounding_box):


    bounding_box = bounding_box



    subimage = np.array(sct.grab(bounding_box))

    # Flip the subimage horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    subimage = cv2.cvtColor(subimage, cv2.COLOR_BGR2RGB)

    # To improve performance
    subimage.flags.writeable = False
    
    # Get the result
    results = face_mesh.process(subimage)
    
    # To improve performance
    subimage.flags.writeable = True
    
    # Convert the color space from RGB to BGR
    subimage = cv2.cvtColor(subimage, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = subimage.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])       
            
            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360
          

            # See where the user's head tilting
            if (y < -15) or (y > 15) or (x < -15) or (x > 15):
                return 0
            else:
                return 1


            # cv2.putText(image, "x: " + str(np.round(x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # cv2.putText(image, "y: " + str(np.round(y,2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # cv2.putText(image, "z: " + str(np.round(z,2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)





        # mp_drawing.draw_landmarks(
        #             image=image,
        #             landmark_list=face_landmarks,
        #             connections=mp_face_mesh,
        #             landmark_drawing_spec=drawing_spec,
        #             connection_drawing_spec=drawing_spec)

