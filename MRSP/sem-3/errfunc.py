#Error function for filter optimization with weighted squared error
#Gerald Schuller, May 2018

import scipy.signal as signal

def errfunc(h):
   numfreqsamples=512
   #desired passband:
   pb=int(numfreqsamples/4.0)
   tb=int(numfreqsamples/8.0)
   w, H = signal.freqz(h,1,numfreqsamples)
   H_desired=np.concatenate((np.ones(pb),np.zeros(numfreqsamples-pb)))
   weights = np.concatenate((np.ones(pb), np.zeros(tb), 1000*np.ones(numfreqsamples-pb-tb)))
   err = np.sum(np.abs(H-H_desired)*weights)
   return err
   
if __name__ == '__main__':
   #testing:
   #Optimize with:
   import scipy.optimize as opt
   import matplotlib.pyplot as plt
   import numpy as np

   minout=opt.minimize(errfunc,np.random.rand(16))
   h=minout.x
   plt.plot(h)
   plt.xlabel('Sample No.')
   plt.ylabel('Value')
   plt.title('Low Pass Impulse Response from Weighted Error Optimization')
   plt.show()
   omega, H =signal.freqz(h)
   plt.plot(omega, 20*np.log10(abs(H)+1e-6))
   #plt.axis([0, 3.14, -30, 30])
   plt.xlabel('Normalized Frequency')
   plt.ylabel('Magnitude (dB)')
   plt.title('Magnitude Response of optimized filter')
   plt.title('LP Mag. Frequency Response from Weighted Error Optimization')
   plt.show()

