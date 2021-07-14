import numpy as np
import cv2
from pathlib import Path
import pickle  # for python3

import scipy.signal

cap = cv2.VideoCapture(0)

f1 = open('videorecord.txt', 'wb')
f2 = open('videorecord_DS.txt', 'wb')
f3 = open('videorecord_DS_compressed.txt', 'wb')

#for j in range(25):
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret == True:
        # show captured frame:
        cv2.imshow('Original',frame)
        # cv2.imshow('B Komponente', frame[:, :, 0])
        # cv2.imshow('G Komponente', frame[:, :, 1])
        # cv2.imshow('R Komponente', frame[:, :, 2])

        # RGB 2 YCBCR:
        Y = (0.299 * frame[:, :, 2] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 0])
        Cb = (-0.16864 * frame[:, :, 2] - 0.33107 * frame[:, :, 1] + 0.49970 * frame[:, :, 0])
        Cr = (0.499813 * frame[:, :, 2] - 0.418531 * frame[:, :, 1] - 0.081282 * frame[:, :, 0])

        # cv2.imshow('Luminance Y', Y/255.)
        # cv2.imshow('Chrominance Cb', np.abs(Cb/255))
        # cv2.imshow('Chrominance Cr', np.abs(Cr/255))

        N = 2
        M = 8
        # Rectangular filter kernel:
        filt1 = np.ones((M, M)) / M
        # Triangular filter kernel:
        filt2 = scipy.signal.convolve2d(filt1, filt1) / M

        # Here goes the processing to reduce data...  #UINT8

        reduced = frame.copy()

        enc = np.zeros(reduced.shape)
        enc1 = np.zeros(reduced.shape)

        # Applying Pyramid Kernel filter
        pyCb = scipy.signal.convolve2d(Cb, filt2, mode='same')
        pyCr = scipy.signal.convolve2d(Cr, filt2, mode='same')

        # filtered
        cv2.imshow("filtered Cb", pyCb)
        cv2.imshow("filtered Cr", pyCr)
        cv2.imshow('Pyramid Kernel Filter', filt2)

        # Downsampled with Zeros            # Keep the Zeros
        Cbds = np.zeros(pyCb.shape)
        Cbds[0::N, 0::N] = pyCb[0::N, 0::N]

        Crds = np.zeros(pyCr.shape)
        Crds[0::N, 0::N] = pyCr[0::N, 0::N]

        # Downsampled without storing Zeros     #Remove the Zeros

        Cbno = pyCb[0::N, 0::N]
        Crno = pyCr[0::N, 0::N]

        enc[:, :, 0] = Y
        enc[:, :, 1] = pyCb
        enc[:, :, 2] = pyCr

        enc1[:, :, 0] = Y
        enc1[:, :, 1] = Cbds
        enc1[:, :, 2] = Crds


        cv2.imshow('Downsampled Cb with zeros', np.abs(Cbds / 255.))
        cv2.imshow('Downsampled Cr with zeros', np.abs(Crds / 255.))

        cv2.imshow('Downsampled Cb without zeros', np.abs(Cbno / 255.))
        cv2.imshow('Downsampled Cr without zeros', np.abs(Crno / 255.))

        pickle.dump(enc, f1)
        pickle.dump(enc1, f2)


        pickle.dump(Y, f3)
        pickle.dump(Cbno, f3)
        pickle.dump(Crno, f3)

        # Display the resulting frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
cap.release()
f1.close()
f2.close()
f3.close()

a = Path('videorecord.txt').stat().st_size
b = Path('videorecord_DS.txt').stat().st_size
c = Path('videorecord_DS_compressed.txt').stat().st_size
print("file size Original", a)
print("file size of Downsampled with zero", b)
print("file size of Downsampled without zero", c)
print("Compression Ratio", c/a)
cv2.destroyAllWindows()
