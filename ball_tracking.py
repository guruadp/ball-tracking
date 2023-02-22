# importing required libraries 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sympy import * 

# reading video
cap = cv2.VideoCapture('ball.mov')

if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# function to mask the ball in the frame
def ball_tracking(img):
    # convert the RBG frame to a HSV frame
    img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # HSV threshold limits to mask ball
    lower_red = np.array([0,170, 130]) 
    upper_red = np.array([255,255,185]) 
    mask = cv2.inRange(img_hsv, lower_red, upper_red)

    # Erode to reduce noise
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(mask,kernel,iterations = 1)
    return erosion

# functiom to find the least sqaure fit
def least_square_fit(x_points, y):
    A = []
    for x in x_points:
        A.append([x**2, x, 1])
    A = np.matrix(A)
    B = np.matrix(y)

    x = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(A), A)), np.transpose(A)), np.transpose(B))
    return x

def find_y(x, coeff):
    a, b, c = np.array(coeff).flatten()
    y = a*pow(x,2) + b*x + c
    return y

X = []
Y = []

# reads each frame
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        # scaling to 75% of the frame size
        scale_percent = 75 
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)        
        img = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

        masked_img = ball_tracking(img)       

        # find the ball coordinates in the frame
        ball_points = np.where(masked_img != 0)
        x_points = ball_points[1]
        y_points = ball_points[0]

        try:
            # find the center of the masked ball
            x_center = (max(x_points) + min(x_points)) // 2
            y_center = (max(y_points) + min(y_points)) // 2

            # plot found center points of ball in each frame
            # plt.scatter(x_center, y_center, label="Center point of the ball") 
            #plt.scatter(X, Y)
            if x_center != None:
                X.append(x_center)
                Y.append(y_center)

            # display the trajectory in the frame 
            for i in range(len(X)):
                cv2.circle(img, (X[i], Y[i]), 1, (255, 255, 255), -1)   
        except:
            pass
        
        # Display each frame
        cv2.imshow('Frame', img)
        # press 'q' to exit the window
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break



coeff = least_square_fit(X, Y)
print('y = (%.4f) * x**2 + (%.4f) * x + (%.4f)' % (coeff[0], coeff[1], coeff[2]))

# find y values in the trajectory
x = np.array(X)
y = find_y(x, coeff)

# find y at x=300
print("When x = 300, y will be: ", find_y(300, coeff))
 
# plot the best fit parabola
plt.title('Trajectory of the ball')
plt.xlabel("X-Coordinate")
plt.ylabel("Y-Coordinate")
plt.legend(loc="upper left")

plt.scatter(X, Y, label="Center point of the ball") 
plt.plot(x,y, label="Best fit curve")

plt.show()

# close all the frames
cv2.destroyAllWindows()
cap.release()
