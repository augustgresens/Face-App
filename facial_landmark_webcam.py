import dlib
import cv2

detector = dlib.get_frontal_face_detector()
predictor_path = "files/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detects faces
    faces = detector(gray)

    # iterates over detected face
    for face in faces:
        # gets facial landmarks
        shape = predictor(gray, face)
        
        # draws a rectangle around the detected face
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # draws facial landmarks onto frame
        for i in range(68):
            x, y = shape.part(i).x, shape.part(i).y
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

    # displays frame
    cv2.imshow("Faces and Landmarks Detected", frame)

    # breaks loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
