import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
 #window management

def Problem1():
    print('hello')
    image = cv2.imread('Jad.jpg', 0)
    rows,cols = image.shape
    theta = np.pi/4
    rotate = np.zeros((rows, cols))

    a = np.cos(theta)
    b = np.sin(theta)
    M = np.zeros((2, 3))
    M[0][0] = a
    M[0][1] = b
    M[1][0] = -b
    M[1][1] = a
    M[0][2] = (1-a) * cols/2 - b * rows/2
    M[1][2] = b*cols/2 + (1-a)*rows/2


    for i in range(rows):
        for j in range(cols):
            B = np.zeros((3,1))
            B[0][0] = i
            B[1][0] = j
            B[2][0] = 1     
            A = np.matmul(M, B)
            

            x = int(A[0])
            y = int(A[1])
            make_black = False
            if x < 0:
                x = 0
                make_black = True
            if y < 0:
                y = 0
                make_black = True
            if x >= cols:
                x = cols-1
                make_black = True
            if y >= rows:
                y = rows-1
                make_black = True
            if make_black:
                rotate[i][j] = 0
            else:
                rotate[i][j] = image[x][y]
    plt.subplot(121),plt.imshow(image),plt.title('Input')
    plt.subplot(122),plt.imshow(rotate),plt.title('Output')
    plt.show()


	

if __name__ == '__main__':
	Problem1()
