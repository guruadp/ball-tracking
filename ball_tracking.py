import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('ball.mov')

if (cap.isOpened()== False): 
  print("Error opening video stream or file")

def ball_tracking(img):
    img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0,172,133])
    upper_red = np.array([20,255,255])
    mask = cv2.inRange(img_hsv, lower_red, upper_red)

    dest_and = cv2.bitwise_and(img, img, mask = mask)

    gray = cv2.cvtColor(dest_and, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        M = cv2.moments(c)
        try:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            points.append([cX, cY])
        except:
            cX, cY = 0,0
        # cv2.circle(img, (cX, cY), 1, (255, 255, 255), -1)

    return img

points = []
x_points = []
y_points = []
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        img = ball_tracking(frame)
        for point in points:
            x, y = point
            cv2.circle(img, (x, y), 1, (255, 255, 255), -1)

        cv2.imshow('Frame', img)
 
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
 
    else: 
        break

points = np.array(points)

x_points = points[:,0]
y_points = points[:,1]

plt.plot(x_points, y_points)
plt.show()
cap.release()