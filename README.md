# ReadTheRoom
A CV project for detecting who is paying attention to your videocall presentation.

Utilizing the Face Detection and Facemesh Models from Google's mediapipe library, this project captures the output of your primary computer monitor, then processes the images real-time to perform face detection and pose estimation. It tallies who is looking at the Camera and looking away, then uses streamlit to visualize the total number of people detected on screen vs. the number currently looking directly ahead.

Though this project uses mss for screen reflection, and Opencv for screen capture- the final UI is videoless, instead doing all video processing behind the scenes. 

Facemesh processing adapted directly from Nicholai Nielsen's model, which also leverages mediapipe.

## How to use:

Create a new python environment (using anaconda, venv, or your choice of environment manager) and run "pip install -r requirements.txt" within the project directory. This will install everything you need to run the project.

To run the application, run 'streamlit run face_detection.py'

## Challenges/Limitations/Biases:

As a proof-of-concept, this model makes a lot of assumptions and comes with a lot of limitations. As the models were originally designed for 'selfie' usage, the facial detection algorithm assumes proximity to the camera. It may not work in low-resolution environments, and early tests indicate challenges with small videopanes within group calls.

Another fundamental assumption about the model is that the act of "paying attention" to a call is staring directly at the screen/webcam. Webcams do not always line up directly with the screen, and it is possible to pay attention to a call without looking directly at the screen.

As this model uses detection of facial features which were learned training data, this model may struggle with faces which do not conform to the constraints of the dataset. While Mediapipe was trained to detect both darker and lighter skin pigmentations, it may struggle with facial features which were not accounted for within training datasets. This could include things like skin conditions which considerably alter the texture or shape of skin, and veils or face coverings.

## Ethics:

This project was made mostly as a proof of concept of what could be put together within a few days of working, and was made for the purposes of better understanding the computer vision landscape/fun.

As we progressively move into the age of remote work, we risk leveraging technology to create a digital panopticon. I designed this software specifically not to single anyone out, but this application can still pose a risk of unnessarily judging people incorrectly by means of a simple heuristic. As a result, I share this online not with the intention of legitimate usage as a productivity tool, and instead as a tongue-in-cheek exploration into the capabilities of modern software.
