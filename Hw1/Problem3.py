import numpy as np
from sympy import Matrix
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
 #window management

def Problem3():
    print('hello')
    image = cv2.imread('cse.jpg', 0)
    rows,cols = image.shape

    br = [282, 183]
    tr = [282, 172]
    bl = [259, 175]
    tl = [259, 165]

    original = []
    original.append(br)
    original.append(tr)
    original.append(bl)
    original.append(tl)
    print("original - " , original)

    sqr_br = [282, 199]
    sqr_tr = [282, 185]
    sqr_bl = [259, 199]
    sqr_tl = [259, 185]

    mapping = []
    mapping.append(sqr_br)
    mapping.append(sqr_tr)
    mapping.append(sqr_bl)
    mapping.append(sqr_tl)

    homo, status = cv2.findHomography(np.asarray(original), np.asarray(mapping))
    dst = cv2.warpPerspective(image, homo,(rows,cols), cv2.WARP_INVERSE_MAP)

    print("true homo - ", homo)
    A = []
    for i in range(4):
        x,y = original[i][0], original[i][1]
        u,v = mapping[i][0], mapping[i][1]
        A.append( [x, y, 1, 0, 0, 0, -u*x, -u*y, -u] )
        A.append( [0, 0, 0, x, y, 1, -v*x, -v*y, -v] )
    
    print("A - ", A)
    B = Matrix(A)
    h = B.nullspace()
    h = h[0].evalf()
    h3 = []
    for i in range(3):
        h3.append([h[3*i], h[3*i+1], h[3*i+2]])
    print("my homo " , h3)
    hi = Matrix(h3).inv()
    morphed_image = np.zeros((rows,cols))
    for i in range(rows):
        for j in range(cols):
            coord = Matrix([[i], [j], [1]])
            genius = hi*coord
            if genius[2] != 0:
                x = int(genius[0]/genius[2])
                y = int(genius[1]/genius[2])
                try:
                    morphed_image[j][i] = image[y][x]
                except:
                    x=5

    plt.subplot(131),plt.imshow(image),plt.title('Input')
    plt.subplot(132),plt.imshow(dst),plt.title('OpenCV')
    plt.subplot(133),plt.imshow(morphed_image),plt.title('MyAlgorithm')
    plt.show()
	
if __name__ == '__main__':
	Problem3()