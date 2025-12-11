from KalmanFilter import KalmanFilter
from Detector import detect
import cv2

import time

Filter = KalmanFilter(dt=0.1, u_x=1, u_y=1, std_acc=1, x_std_meas=0.1, y_std_meas=0.1)

cap = cv2.VideoCapture('video/randomball.avi')
trajectory_points = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    centers = detect(frame)

    if centers:
        c = centers[0]
        Filter.predict()

        # Store predicted state before update
        predicted_state_Xk = Filter.Xk.copy() 
        predicted_x, predicted_y = int(predicted_state_Xk[0][0]), int(predicted_state_Xk[1][0]) 

        Filter.update(c)
        x, y = int(Filter.Xk[0][0]), int(Filter.Xk[1][0])

        #Draw detected circle
        cv2.circle(frame, (int(c[0][0]), int(c[1][0])), 10, (0, 255, 0), 2)

        #Draw a blue rectangle as the predicted object position
        rect_size = 20
        cv2.rectangle(frame, (predicted_x - rect_size//2, predicted_y - rect_size//2), 
                      (predicted_x + rect_size//2, predicted_y + rect_size//2), (255, 0, 0), 2)

        #Draw a red rectangle as the estimated object position
        cv2.rectangle(frame, (x - rect_size//2, y - rect_size//2), 
                      (x + rect_size//2, y + rect_size//2), (0, 0, 255), 2)

        trajectory_points.append((x, y))

        #Draw the trajectory (tracking path)
        for i in range(1, len(trajectory_points)):
            cv2.line(frame, trajectory_points[i-1], trajectory_points[i], (255, 255, 0), 2)


        cv2.putText(frame, 'Kalman Filter Prediction', (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('Object Tracking', frame)

        time.sleep(0.05)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

