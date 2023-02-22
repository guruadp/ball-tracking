import cv2
import numpy as np
import matplotlib.pyplot as plt
from sympy import * 

cap = cv2.VideoCapture('ball.mov')

if (cap.isOpened()== False): 
  print("Error opening video stream or file")


def ball_tracking(img):
    img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0,170, 130]) 
    upper_red = np.array([255,255,185]) 
    mask = cv2.inRange(img_hsv, lower_red, upper_red)
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(mask,kernel,iterations = 1)
    # kernel = np.ones((5,5),np.float32)/25
    # dst = cv2.filter2D(img,-1,kernel)
    # dest_and = cv2.bitwise_and(img, img, mask = mask)

    return erosion
points = []
A = []
def least_square_fit(x_points, y_points):
    for x in x_points:
        A.append([x*x, x, 1])
    a = np.matrix(A) 
    b = np.matrix(y_points)
    print(a.shape)
    print(b.shape)
    # x = np.linalg.inv(np.transpose(a) @ a) @ np.transpose(a) @ b
    # q = 
    # x = np.linalg.inv(np.transpose(a).dot(a)).dot(np.transpose).dot(b)

    x = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(a), a)), np.transpose(a)), np.transpose(b))
    
    print(x.shape)
    return x
X = []
Y = []
# print(points)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        scale_percent = 75 
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        img = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        img = ball_tracking(img)       

        points = np.where(img != 0)
        x_points = points[1]
        y_points = points[0]

        try:
            # if x_points != None :
            x_center = (max(x_points) + min(x_points)) // 2
            y_center = (max(y_points) + min(y_points)) // 2
            # print(x_center, y_center)
            plt.scatter(x_center, y_center)
            # if abs(x_points[len(x_points)-1] - x_center) <= 100:
            # points.append([x_center, y_center]) 
            if x_center != None:
                X.append(x_center)
                Y.append(y_center)

            
            # if abs(y_points[len(y_points)-1] - y_center) <= 100:
            # Y.append(y_center) 
            # print(x_center, y_center)
            # cv2.circle(img, (x_center, y_center), 2, (255, 255, 255), -1)
        except:
            pass
        # print(points[-1])
        # for point in points:
        #     x, y = point
        #     cv2.circle(img, (x, y), 1, (255, 255, 255), -1)    
        # cv2.imshow('Frame', img)
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break
 
    else: 
        break

# for point in points:
#     x, y = point
#     X.append(x)
#     Y.append(y)
#     cv2.circle(img, (x, y), 1, (255, 255, 255), -1)    
coeff = least_square_fit(X, Y)

# print(coeff)
coeff = np.array(coeff).flatten()
print(coeff)
x = np.array(X)
print("X SHAPE: ",x.shape)
y = coeff[0]*pow(x,2) + coeff[1]*x + coeff[2]
# plt.plot(x_points, y_points)
plt.plot(x,y)
plt.show()
cap.release()
