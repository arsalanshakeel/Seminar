import numpy as np
import cv2
# import sys
import pickle
import scipy.signal
N = 2

f3 = open('videorecord_DS_compressed.txt', 'rb')
try:
    while (True):
        # load next frame from file f and "de-pickle" it, convert from a string back to matrix or tensor:

        Y = pickle.load(f3)
        Y = Y.astype(np.uint8)
        r, c = Y.shape

        #Upsampling Cb:

        Cbno = pickle.load(f3)
        r1 = Cbno.shape[0]
        c1 = Cbno.shape[1]
        upCb = np.zeros((r1 * N, c1 * N))
        upCb[0::N, 0::N] = Cbno

        # Upsampling Cr:

        Crno = pickle.load(f3)
        r2 = Crno.shape[0]
        c2 = Crno.shape[1]
        upCr = np.zeros((r2 * N, c2 * N))
        upCr[0::N, 0::N] = Crno

        #reduced = np.zeros((r, c, 3))

        #reduced[:, :, 0] = Y
        #reduced[:, :, 1] = upCb
        #reduced[:, :, 2] = upCr

        M = 8
        # Rectangular filter kernel:
        filt1 = np.ones((M, M)) / M;
        # Triangular filter kernel:
        filt2 = scipy.signal.convolve2d(filt1, filt1) / M

        pyCb = scipy.signal.convolve2d(upCb, filt2, mode='same')
        pyCr = scipy.signal.convolve2d(upCr, filt2, mode='same')

        reduced = np.zeros((r, c, 3))

        reduced[:, :, 0] = Y
        reduced[:, :, 1] = pyCb
        reduced[:, :, 2] = pyCr



       # here goes the decoding:
        dec = reduced.copy()

        Y0 = (dec[:, :, 0]).astype(np.uint8) / 255.
        Cb0 = (dec[:, :, 1]) / 255.
        Cr0 = (dec[:, :, 2]) / 255.

        # YCbCr 2 RGB:

        R = (1.0 * Y0 + 0.0 * Cb0 + 1.4025 * Cr0);
        G = (1.0 * Y0 - 0.34434 * Cb0 - 0.7144 * Cr0);
        B = (1.0 * Y0 + 1.7731 * Cb0 + 0.0 * Cr0);


        #cv2.imshow('Red', R)
        #cv2.imshow('Green', G)
        #cv2.imshow('Blue', B)

        RGBframe = np.zeros(dec.shape)

        RGBframe[:, :, 2] = R
        RGBframe[:, :, 1] = G
        RGBframe[:, :, 0] = B

        cv2.imshow("Original", RGBframe)


            # Wait for key for 50ms, to get about 20 frames per second playback
            # (depends also on speed of the machine, and recording frame rate, try out):
        if cv2.waitKey(200) & 0xFF == ord('q'):
            break
except (EOFError):
    pass
cv2.destroyAllWindows()
