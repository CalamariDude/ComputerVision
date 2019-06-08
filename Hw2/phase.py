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

magnitude_spectrum = 20*np.log(np.abs(fshift))


fshift_mag = np.real(fshift_mag)
mag_img_back = np.fft.ifft2(fshift_mag)
mag_img_back = np.abs(mag_img_back)

fshift = fshift/np.abs(fshift) 
img_back = np.fft.ifft2(fshift)
img_back = np.abs(img_back)

plt.subplot(141),plt.imshow(aligator, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(142),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(143),plt.imshow(img_back, cmap = 'gray')
plt.title('phase only'), plt.xticks([]), plt.yticks([])
plt.subplot(144),plt.imshow(mag_img_back, cmap = 'gray')
plt.title('Magnitude-only'), plt.xticks([]), plt.yticks([])
plt.show()

