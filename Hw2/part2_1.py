import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy import ndimage


def minOf(arr):
    minindex = 0
    for i in range(len(arr)):
        if arr[i] < arr[minindex]:
            minindex = i
    return minindex

def verticalCut(img, M):
    
    #iterator through last row of M and find min starting point
    minindex = 0
    for j in range(len(M[0])):
        if(M[len(M)-1][j] < M[len(M)-1][minindex]):
            minindex = j
    
    #now cut up
    j = minindex
    rows, cols = M.shape
    M_new = np.zeros((rows, cols-1))
    img_new = np.zeros((rows, cols-1))

    for i in range(len(M)-1, -1, -1):
        #get max of i-1's
        #save current and move up to max
        #delete current(now previous) pixel from M and img
        newj = 0
        # print("j = ", j, " and len(M)-1 = ", len(M)-1)
        if j == 0:
            newj = minOf([M[i-1][j], M[i-1][j+1]])
            newj = j + newj
        elif j == len(M[0])-1:
            newj = minOf([M[i-1][j-1], M[i-1][j]]) 
            newj = j + newj - 1 
        else:
            newj = minOf([ M[i-1][j-1], M[i-1][j], M[i-1][j+1] ]) 
            newj = j + newj - 1

        M_new[i] = np.delete(M[i], j)
        img_new[i] = np.delete(img[i], j)
        j = newj

    return img_new, M_new

def reduceWidth(img, pixels = 1):
    
    #smooth for less noise
    blurred = ndimage.filters.gaussian_filter(img, sigma=7)
    
    # Get x-gradient in "sx"
    sx = ndimage.sobel(blurred,axis=0,mode='constant')
    
    # Get y-gradient in "sy"
    sy = ndimage.sobel(blurred,axis=1,mode='constant')
    
    # Get square root of sum of squares
    energy=np.hypot(sx, sy)

    #get min cum energy
    M = minumumCumEnergyVertical(energy)
    while pixels > 0:
        img, M = verticalCut(img, M)
        pixels= pixels - 1

    return img


def reduceHeight(img, pixels):
    newimg = np.transpose(img)
    newimg = reduceWidth(newimg, pixels)
    newimg = np.transpose(newimg)
    return newimg
def reduceCombo(img, pixels):
    return 5

def minumumCumEnergyVertical(energy):
    #M(i,j) = energy(i,j) + min(M(i-1, j-1), M(i-1, j), M(i-1, j+1)) (down-left, down, down-right)
    M = np.zeros(energy.shape)
    for i in range(len(energy[0])):
        M[0][i] = energy[0][i]

    for i in range(1, len(energy)):
        for j in range(len(energy[0])):
            if j == 0:
                M[i][j] = energy[i][j] + min(M[i-1][j], M[i-1, j+1]) 
            elif j == len(energy[0])-1:
                M[i][j] = energy[i][j] + min(M[i-1][j-1], M[i-1][j])
            else:
                M[i][j] = energy[i][j] + min(M[i-1][j-1], min(M[i-1][j], M[i-1, j+1])) #first column be weird, this gets min of 3 pixels above
    print("Here", M.shape)
    return M               


tower = cv2.imread('Century-Tower.jpg', 0)
# tower = cv2.imread('plant1.jpg', 0)

print(np.asarray(tower).shape)
 # Get x-gradient in "sx"
blurred = ndimage.filters.gaussian_filter(tower, sigma=7)
sx = ndimage.sobel(blurred,axis=0,mode='constant')
    
# Get y-gradient in "sy"
sy = ndimage.sobel(blurred,axis=1,mode='constant')
    
# Get square root of sum of squares
energy=np.hypot(sx, sy)

cum_vert  = minumumCumEnergyVertical(energy)
cum_horiz = np.transpose(minumumCumEnergyVertical(np.transpose(energy)))

reducedWidth = reduceWidth(tower, 100)
reducedHeight = reduceHeight(tower, 100)
print("shape before reducewidth =", tower.shape, " and shape after reducewidth = ", reducedWidth.shape)
print("shape before reducedheight =", tower.shape, " and shape after reduceheight = ", reducedHeight.shape)

# plt.subplot(131),plt.imshow(np.float32(energy), cmap = 'gray')
# plt.title('Energy'), plt.xticks([]), plt.yticks([])
# plt.subplot(132),plt.imshow(np.float32(cum_vert), cmap = 'hot')
# plt.title('Vertical Cumulative'), plt.xticks([]), plt.yticks([])
# plt.subplot(133),plt.imshow(np.float32(cum_horiz), cmap = 'hot')
# plt.title('Horizontal Cumulative'), plt.xticks([]), plt.yticks([])
plt.subplot(121),plt.imshow(reducedWidth, cmap = 'gray')
plt.title('Reduced Width'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(reducedHeight, cmap = 'gray')
plt.title('Reduced Height'), plt.xticks([]), plt.yticks([])
plt.show()
