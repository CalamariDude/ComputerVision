import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft

aligator = cv2.imread('alligator-crossing.jpg', 0)
tower = cv2.imread('Century-Tower.jpg', 0)

f = np.fft.fft2(aligator)
fshift = np.fft.fftshift(f)
fshift_mag = fshift
fshift_orig = fshift

fshift_orig = fshift/np.abs(fshift) 
original = np.fft.ifft2(fshift_orig)
original = np.abs(original)


fshift_mag = np.real(fshift_mag)


#add noise
for i in range(len(fshift)):
    for j in range(len(fshift[0])):
        fshift_mag[i][j] = fshift_mag[i][j] + np.random.normal(0, 10000)
mag_img_back = np.fft.ifft2(fshift_mag)
mag_img_back = np.abs(mag_img_back)     



fshift = fshift/np.abs(fshift) 

#add noise
for i in range(len(fshift)):
    for j in range(len(fshift[0])):
        fshift[i][j] = fshift[i][j] + np.random.normal(0, .9)
img_back = np.fft.ifft2(fshift)
img_back = np.abs(img_back)

plt.subplot(141),plt.imshow(aligator, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(142),plt.imshow(original, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(143),plt.imshow(img_back, cmap = 'gray')
plt.title('noise on phase-only'), plt.xticks([]), plt.yticks([])
plt.subplot(144),plt.imshow(mag_img_back, cmap = 'gray')
plt.title('noise on magnitude-only'), plt.xticks([]), plt.yticks([])
plt.show()

