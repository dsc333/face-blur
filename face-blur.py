import cv2
import mediapipe as mp # Medipipe requires Python 3.9-3.12 

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Capture video from webcam -- change to 0 or 1 if capture fails
cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    
    height, width, channels = image.shape

    # Make image smaller for improved efficiency
    image = cv2.resize(image,(int(width/2), int(height/2)))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:
        for detection in results.detections:
            # Get image dimensions
            img_h, img_w, _ = image.shape
            mp_drawing.draw_detection(image, detection)
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            # Get relative coordinates and convert to absolute pixel values
            x1 = int(bbox.xmin * img_w) 
            y1 = int(bbox.ymin * img_h) 
            w = int(bbox.width * img_w) 
            h = int(bbox.height * img_h)

            # blur face based on coordinates of bounding box
            # The last values are the strength of the blur filter
            image[y1:y1+h,x1:x1+h, :] = cv2.blur(image[y1:y1+h, x1:x1+w, :], (50, 50))
    
    cv2.imshow('MediaPipe Face Detection', image)

    # Press ESC to exit
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()

