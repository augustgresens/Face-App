import cv2
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("files/shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

mustache = cv2.imread("img/mustache.jpg", cv2.IMREAD_UNCHANGED)
sunglasses = cv2.imread("img/sunglasses.jpg", cv2.IMREAD_UNCHANGED)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to capture frame")
        break

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(grayscale)

    for face in faces:
        landmarks = predictor(grayscale, face)


        # forehead landmarks (17 to 26)
        forehead_pts = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(17, 27)]

        # upper lip landmarks (48 to 59)
        upper_lip_pts = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 60)]

        # left eye landmarks (36 to 41)
        left_eye_pts = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]

        # right eye landmarks (42 to 47)
        right_eye_pts = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

        # nose landmarks (27 to 34)
        nose_pts = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(27, 36)]

        # mouth landmarks (60 to 67)
        mouth_pts = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(60, 68)]

        # calculate the lowest y-value nose landmark
        bottom_of_nose_y = max(nose_pts[6][1], nose_pts[7][1])
    
        # calculate the highest y-value mouth landmark 
        top_of_mouth_y = min(mouth_pts[1][1], mouth_pts[2][1], mouth_pts[3][1], mouth_pts[4][1])

        # creates top-left point on the nose rectangle
        top_left_nose = (nose_pts[4][0], nose_pts[0][1])
    


        # forehead rectangle, heightened by 300%
        cv2.rectangle(frame, (forehead_pts[0][0], forehead_pts[0][1]),
              (forehead_pts[8][0], forehead_pts[2][1] + int(3 * (forehead_pts[2][1] - forehead_pts[0][1]))), (0, 255, 0), 1)

        # upper lip rectangle
        cv2.rectangle(frame, (upper_lip_pts[0][0], bottom_of_nose_y),
              (upper_lip_pts[6][0], top_of_mouth_y), (0, 255, 0), 1)

        # left eye rectangle
        cv2.rectangle(frame, (left_eye_pts[0][0], left_eye_pts[1][1]),
                  (left_eye_pts[3][0], left_eye_pts[4][1]), (0, 255, 0), 1)

        # right eye rectangle
        cv2.rectangle(frame, (right_eye_pts[0][0], right_eye_pts[1][1]),
                  (right_eye_pts[3][0], right_eye_pts[4][1]), (0, 255, 0), 1)

        # nose rectangle
        cv2.rectangle(frame, top_left_nose, (nose_pts[8][0], nose_pts[8][1]), (0, 255, 0), 1)



        # resizes mustache image to fit in upper lip rectangle
        mustache_width = upper_lip_pts[6][0] - upper_lip_pts[0][0]
        mustache_height = top_of_mouth_y - bottom_of_nose_y

        # checks if the scaling factors are positive before resizing
        if mustache_width > 0 and mustache_height > 0:
            mustache_resized = cv2.resize(mustache, (int(mustache_width), int(mustache_height)))

            # checks if image is in grayscale (2 channels)
            if len(mustache_resized.shape) == 2:
                # if the mustache image is grayscale, convert it to bgra (bgr with alpha for transparency)
                mustache_resized = cv2.cvtColor(mustache_resized, cv2.COLOR_GRAY2BGRA)
        else:
            mustache_resized = mustache

        # position of mustache
        mustache_x = upper_lip_pts[0][0]
        mustache_y = bottom_of_nose_y

        # applies the mustache to the frame
        frame[mustache_y : mustache_y + mustache_height, mustache_x : mustache_x + mustache_width] = mustache_resized

         # calculates the position of the mustache inside the upper lip rectangle
        mustache_x = upper_lip_pts[0][0]
        mustache_y = bottom_of_nose_y

        # applies the mustache to the frame
        frame[mustache_y : mustache_y + mustache_height, mustache_x : mustache_x + mustache_width] = mustache_resized



        # resizes the sunglasses to cover both ends of the face and cover the eyes
        sunglasses_width = forehead_pts[8][0] - forehead_pts[0][0]
        sunglasses_height = max(left_eye_pts[3][1] - forehead_pts[2][1], right_eye_pts[3][1] - forehead_pts[2][1])

        # checks if the scaling factors are positive before resizing
        if sunglasses_width > 0 and sunglasses_height > 0:
            sunglasses_resized = cv2.resize(sunglasses, (int(sunglasses_width), int(sunglasses_height)))

            # checks if the sunglasses image is grayscaled
            if len(sunglasses_resized.shape) == 2:
                # if the sunglasses image is grayscaled, convert it to 4 channels
                sunglasses_resized = cv2.cvtColor(sunglasses_resized, cv2.COLOR_GRAY2BGRA)
        else:
            sunglasses_resized = sunglasses

        # calculates the position of the sunglasses
        sunglasses_x = forehead_pts[0][0]
        sunglasses_y = min(left_eye_pts[0][1], right_eye_pts[0][1], forehead_pts[0][1])  # Adjust position as needed

        # applies the sunglasses to the frame
        frame[sunglasses_y : sunglasses_y + sunglasses_height, sunglasses_x : sunglasses_x + sunglasses_width] = sunglasses_resized

    # displays the frame
    cv2.imshow("Face with rectangles", frame)

    # breaks loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the capture
cap.release()
cv2.destroyAllWindows()
