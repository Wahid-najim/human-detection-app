import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector

# Load the pre-trained models from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize HandDetector
hand_detector = HandDetector(detectionCon=0.8)

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    # Read frame from video capture
    ret, frame = cap.read()

    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Perform eye detection
    eyes = eye_cascade.detectMultiScale(gray)

    # Perform hand detection
    hands, _ = hand_detector.findHands(frame)

    # Check if any face, eyes or hands are detected
    if len(faces) != 0 or len(eyes) != 0 or hands:
        cv2.putText(frame, "It is a human", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "It is something else", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the output
    cv2.imshow('Human Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
