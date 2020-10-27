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



plant1 = cv2.imread('plant1.jpg', 0)
plant2 = cv2.imread('plant2.jpg', 0)
plant3 = cv2.imread('plant3.jpg', 0)
plant1_copy1 = plant1
plant1_copy2 = plant1
plant2_copy1 = plant2
plant2_copy2 = plant2
plant3_copy1 = plant3


#GOOD

#simple 100 looks good for width
reducedplant1 = reduceWidth(plant1, 100)
h, w = plant1_copy1.shape
plant1_copy1 = cv2.resize(plant1_copy1,(w-100,h))
#simple 100 looks good for height
reducedplant2 = reduceHeight(plant2, 100)
h, w = plant2_copy1.shape
plant2_copy1 = cv2.resize(plant2_copy1,(w,h-100))
# simple 200 from both looks good
reducedplant3 = reduceWidth(plant2, 200)
reducedplant3 = reduceHeight(plant2, 200)
h, w = plant3_copy1.shape
plant3_copy1 = cv2.resize(plant3_copy1,(w-200,h-200))
#BAD


reducedplant1_bad = reduceWidth(plant1, 500)
h, w = plant1_copy2.shape
plant1_copy2 = cv2.resize(plant1_copy2,(w-500,h))
reducedplant2_bad = reduceHeight(plant2, 400)
h, w = plant2_copy2.shape
plant2_copy2 = cv2.resize(plant2_copy2,(w,h-400))

rows = 5
cols = 3
fig=plt.figure(figsize=(rows, cols))
fig.add_subplot(rows, cols, 1)
plt.imshow(plant1)
fig.add_subplot(rows, cols, 2)
plt.imshow(plant1_copy1)
fig.add_subplot(rows, cols, 3)
plt.imshow(reducedplant1)
print("original shape = ", np.asarray(plant1).shape, " and output shape is ", reducedplant1.shape)

fig.add_subplot(rows, cols, 4)
plt.imshow(plant2)
fig.add_subplot(rows, cols, 5)
plt.imshow(plant2_copy1)
fig.add_subplot(rows, cols, 6)
plt.imshow(reducedplant2)
print("original shape = ", np.asarray(plant2).shape, " and output shape is ", reducedplant2.shape)

fig.add_subplot(rows, cols, 7)
plt.imshow(plant3)
fig.add_subplot(rows, cols, 8)
plt.imshow(plant3_copy1)
fig.add_subplot(rows, cols, 9)
plt.imshow(reducedplant3)
print("original shape = ", np.asarray(plant3).shape, " and output shape is ", reducedplant3.shape)

fig.add_subplot(rows, cols, 10)
plt.imshow(plant1)
fig.add_subplot(rows, cols, 11)
plt.imshow(plant1_copy2)
fig.add_subplot(rows, cols, 12)
plt.imshow(reducedplant1_bad)
print("original shape = ", np.asarray(plant1).shape, " and output shape is ", reducedplant1_bad.shape)

fig.add_subplot(rows, cols, 13)
plt.imshow(plant2)
fig.add_subplot(rows, cols, 14)
plt.imshow(plant2_copy2)
fig.add_subplot(rows, cols, 15)
plt.imshow(reducedplant2_bad)
print("original shape = ", np.asarray(plant2).shape, " and output shape is ", reducedplant2_bad.shape)


# plt.subplot(141),plt.imshow(aligator, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(142),plt.imshow(aligator, cmap = 'gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(143),plt.imshow(aligator, cmap = 'gray')
# plt.title('noise on magnitude'), plt.xticks([]), plt.yticks([])
# plt.subplot(144),plt.imshow(aligator, cmap = 'gray')
# plt.title('noise on phase'), plt.xticks([]), plt.yticks([])
plt.show()
