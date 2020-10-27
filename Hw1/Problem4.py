import numpy as np
from sympy import Matrix
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import math as math
 #window management

def Problem4():
    print('hello')
    image = cv2.imread('checkers.jpg', 0)
    rows,cols = image.shape
    shear = np.zeros((rows, cols))
    ST = np.identity(2) 
    #separte these into each step

    point1 = [0,0,0]
    point2 = [5,5,0]
    point3 = [0,5,0]
    point4 = [0,5,5]
    point5 = [0,0,5]
    point6 = [5,0,0]
    point7 = [3,0,0]
    point8 = [2,0,0]
    point9 = [0,0,3]
    point10 = [0,0,2]
    point11 = [0,4,0]
    point12 = [0,2,0]
    point13 = [0,4,3]
    point14 = [0,4,2]
    point15 = [0,4,1]
    point16 = [1,4,0]
    point17 = [2,4,0]
    point18 = [3,4,0]

    original = []
    original.append(point1)
    original.append(point2)
    original.append(point3)
    original.append(point4)
    original.append(point5)
    original.append(point6)
    original.append(point7)
    original.append(point8)
    original.append(point9)
    original.append(point10)
    original.append(point11)
    original.append(point12)
    original.append(point13)
    original.append(point14)
    original.append(point15)
    original.append(point16)
    original.append(point17)
    original.append(point18)
    # print ("original -", original)

    pixel1 = [82, 48]
    pixel2 = [129, 127]
    pixel3 = [77,144]
    pixel4 = [22, 121]
    pixel5 = [26, 29]
    pixel6 = [134, 37]
    pixel7 = [114, 42]
    pixel8 = [104, 44]
    pixel9 = [48, 37]
    pixel10 = [59, 40]
    pixel11 = [78,124]
    pixel12 = [80, 86]
    pixel13 = [44,111]
    pixel14 = [55,116]
    pixel15 = [66,112]
    pixel16 = [89,122]
    pixel17 = [100,119]
    pixel18 = [109,116]

    mapping = []
    mapping.append(pixel1)
    mapping.append(pixel2)
    mapping.append(pixel3)
    mapping.append(pixel4)
    mapping.append(pixel5)
    mapping.append(pixel6)
    mapping.append(pixel7)
    mapping.append(pixel8)
    mapping.append(pixel9)
    mapping.append(pixel10)
    mapping.append(pixel11)
    mapping.append(pixel12)
    mapping.append(pixel13)
    mapping.append(pixel14)
    mapping.append(pixel15)
    mapping.append(pixel16)
    mapping.append(pixel17)
    mapping.append(pixel18)

    testpoint1 = [0,3,4]
    testpoint2 = [0,3,3]
    testpoint3 = [0,3,2]
    testpoint4 = [0,3,1]
    testpoint5 = [0,3,0]
    testpoint6 = [1,3,0]
    testpoint7 = [2,3,0]
    testpoint8 = [3,3,0]
    testpoint9 = [4,3,0]
    
    testpoints = []
    testpoints.append(testpoint1)
    testpoints.append(testpoint2)
    testpoints.append(testpoint3)
    testpoints.append(testpoint4)
    testpoints.append(testpoint5)
    testpoints.append(testpoint6)
    testpoints.append(testpoint7)
    testpoints.append(testpoint8)
    testpoints.append(testpoint9)
    
    testpixel1 = [35,89]
    testpixel2 = [45,93]
    testpixel3 = [55,97]
    testpixel4 = [67,101]
    testpixel5 = [79,106]
    testpixel6 = [90,104]
    testpixel7 = [101,101]
    testpixel8 = [111,98]
    testpixel9 = [121,95]

    testpixels = []
    testpixels.append(testpixel1)
    testpixels.append(testpixel2)
    testpixels.append(testpixel3)
    testpixels.append(testpixel4)
    testpixels.append(testpixel5)
    testpixels.append(testpixel6)
    testpixels.append(testpixel7)
    testpixels.append(testpixel8)
    testpixels.append(testpixel9)

    A = []
    for i in range(len(original)):
        x,y,z = original[i][0], original[i][1], original[i][2]
        u,v = mapping[i][0], mapping[i][1]
        A.append( [x, y, z ,1, 0, 0, 0, 0 ,  -u*x, -u*y, -u*z,  -u] )
        A.append( [0, 0, 0, 0, x, y, z, 1, -v*x, -v*y, -v*z , -v] ) 
    
    print("A = " , A)
    U, S, Vh = np.linalg.svd(A)
    h = Vh[-1,:] / Vh[-1,-1]

    print("h - ", h)
    P = []
    for i in range(3):
        print(4*i+3)
        P.append([h[4*i], h[4*i+1], h[4*i+2], h[4*i+3]])

    # Extract out C
    U, S, Vh = np.linalg.svd(P)
    C = -Vh[-1,:] / Vh[-1,-1]
    print("-C = ", np.asarray(C).shape) #THIS IS 4 down which is incorrect!

    # Extract our Q(which is K) and R
    M = np.asarray(P)[:,:3]
    print("M", M)
    Q, R = np.linalg.qr(M)

    print("Q - " , Q)

    print("R - " , R)
    ax = Q[0,0]
    ay = Q[1,1]
    s = Q[0,1]
    tx = Q[0,2]
    ty = Q[1,2]

    expectedError = 0
    for i in range(len(testpoints)):
        coord = testpoints[i]
        truepixel = testpixels[i]
        coord.append(1)
        genius = np.matmul(np.asarray(P), np.asarray(coord))
        x = int(genius[0]/genius[2])
        y = int(genius[1]/genius[2])
        dist = distance(x, truepixel[0], y, truepixel[1])
        expectedError = expectedError + dist

    print("Expected error = ", expectedError / len(testpoints))
    print("ax = ", ax)
    print("ax = ", ay)
    print("s = ", s)
    print("tx = ", tx)
    print("ty = ", ty)

def distance(x1, x2, y1, y2):
    xdiff = x2-x1
    xdiff = xdiff**2
    ydiff = y2-y1
    ydiff = ydiff**2
    total = math.sqrt(ydiff+xdiff)
    return total
    
	
if __name__ == '__main__':
	Problem4()