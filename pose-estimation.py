import cv2
import mediapipe as mp # Medipipe requires Python 3.9-3.12 

# Initialise Media Pipe Pose features
mp_pose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
pose = mp_pose.Pose()


# Capture video from webcam (change to 0 or 1 if capture fails)
cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    height, width, channels = image.shape

    # Make image smaller for improved efficiency
    image = cv2.resize(image,(int(width/2), int(height/2)))
    
    # cv2.imshow("image", image)
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow("image", image)
    result = pose.process(rgb_img)

    # If no landmarks are detected in current frame skip it
    if not result.pose_landmarks:
        continue
        
    print(result.pose_landmarks)
    try:
        nose_x =  result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * width
        nose_y = result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * height
        l_wrist_y = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * width
        print('Nose X,Y Coords are', nose_x, nose_y)
        print('Left write Y coordinate', l_wrist_y)
    except:
        pass

    # Draw the landmarks of body and then show it in the preview window
    mpDraw.draw_landmarks(image,
                          result.pose_landmarks,
                          mp_pose.POSE_CONNECTIONS)
    # if the wrist of left hand is higher than the nose
    # we'll output the message LEFT HAND RAISED
    if l_wrist_y < nose_y:
        cv2.putText(image, 'LEFT HAND RAISED',
                    (70, 50),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.imshow("frame",image)
    # cv2.imshow("frame",rgb_img)

    # Press ESC to exit
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
