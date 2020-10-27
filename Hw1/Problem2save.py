import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import math as math
 #window management

def Problem2():
    print('hello')
    image = cv2.imread('Jad.jpg', 0)
    rows,cols = image.shape
    

    # X' = SRM*X + T   where S = shear, R = rotation, M = magnification, T = translation
    M = [[math.sqrt(2), 0],
         [0, math.sqrt(2)]]
    
    theta = math.pi/4
    R = [[math.cos(theta), -math.sin(theta)],
         [math.sin(theta), math.cos(theta)]]
    
    S = [[1, 2],
         [0, 1]]
    
    T = [[3],
         [1]]

    magnification = np.zeros((rows, cols))
    rotation = np.zeros((rows, cols))
    shear = np.zeros((rows, cols))
    translation = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            B = np.zeros((2,1))
            B[0][0] = i
            B[1][0] = j

            ##shear
            A = np.matmul(S, B)

            x = int(A[0])
            y = int(A[1])
            if x >= 0 and x < cols and y >= 0 and y < rows:
                shear[i][j] = image[x][y]
            
            ##rotation
            A = np.matmul(R, B)

            x = int(A[0])
            y = int(A[1])
            if x >= 0 and x < cols and y >= 0 and y < rows:
                rotation[i][j] = shear[x][y]
            
            ##magnification
            A = np.matmul(M, B)

            x = int(A[0])
            y = int(A[1])
            if x >= 0 and x < cols and y >= 0 and y < rows:
                magnification[i][j] = rotation[x][y]
            
            ##translation
            A = np.add(T, B)

            x = int(A[0])
            y = int(A[1])
            if x >= 0 and x < cols and y >= 0 and y < rows:
                translation[i][j] = magnification[x][y]


    plt.subplot(151),plt.imshow(image),plt.title('Input')
    plt.subplot(152),plt.imshow(shear),plt.title('Shear')
    plt.subplot(153),plt.imshow(rotation),plt.title('Rotation')
    plt.subplot(154),plt.imshow(magnification),plt.title('Magnification')
    plt.subplot(155),plt.imshow(translation),plt.title('Translation')
    plt.show()


	

if __name__ == '__main__':
	Problem2()