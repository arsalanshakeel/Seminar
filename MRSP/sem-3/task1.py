import numpy as np
import scipy.signal as sig
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from sound import *
import scipy.optimize as opt

N = 8 # number of subbands
wc = (1.0/N)*np.pi # bandwidth
l = 16 # length of filter
n = np.arange(l)

frequency, s = wav.read("Track32.wav")
s1 = s[:,0]   #channel 1


# optimization
def errfunc(h):
    numfreqsamples = 512

    # desired passband:
    pb = int(numfreqsamples / 4.0)
    # desired transition band:
    tb = int(numfreqsamples / 8.0)

    w, H = sig.freqz(h, 1, numfreqsamples)
    H_desired = np.concatenate((np.ones(pb), np.zeros(numfreqsamples - pb)))
    weights = np.concatenate((np.ones(pb), np.zeros(tb), 1000 * np.ones(numfreqsamples - pb - tb)))
    err = np.sum(np.abs(H - H_desired) * weights)
    return err
minout=opt.minimize(errfunc,np.random.rand(l))  # get h when error is minimal
h = minout.x
#h = h/max(h)

# plot impluse response of optimized window 
plt.subplot(211)
plt.plot(h, color='r')
plt.title('Impulse response of optimized window')
plt.xlabel('Time in samples')
plt.ylabel('Sample values')


# plot frequency response of optimized window 
f,fs = sig.freqz(h)
plt.subplot(212)
plt.plot(f,20*np.log10(np.abs(fs)+1e-6), color='r')
plt.title('Frequency response of optimized window')
plt.xlabel('Normalized frequency')
plt.ylabel("dB")
plt.show()

kw = np.kaiser(l,8)

# ideal lowpass, passband = 1/8 pi
#filt[0] = np.sin(fc*(n-nd))/(np.pi*(n-nd))
ideal_lp = np.sin(wc*(n-(l-1)/2))/(np.pi*(n-(l-1)/2))
# ideal highpass, passband = 1/8 pi
#ideal_hp = ideal_lp*kw*np.cos(np.pi * n)
ideal_hp = np.sin(np.pi*(n-(l-1)/2))/(np.pi*(n-(l-1)/2))-np.sin(0.875*np.pi*(n-(l-1)/2))/(np.pi*(n-(l-1)/2))
# ideal bandpass, passband = 1/16 pi
ideal_bp = np.sin((wc/2)*(n-(l-1)/2))/(np.pi*(n-(l-1)/2)) 

nfilt = np.zeros((N,l))  # new filters
nfilt[0] = h*ideal_lp
nfilt[7] = h*ideal_hp
# center is wc*(i+0.5)
for i in range(1,7):
   nfilt[i] = ideal_bp*h*np.cos(n*wc*(i+0.5))
     
f1,fs1 = sig.freqz(nfilt[0])    # frequency response of filters
f2,fs2 = sig.freqz(nfilt[1])    
f3,fs3 = sig.freqz(nfilt[2]) 
f4,fs4 = sig.freqz(nfilt[3]) 
f5,fs5 = sig.freqz(nfilt[4]) 
f6,fs6 = sig.freqz(nfilt[5]) 
f7,fs7 = sig.freqz(nfilt[6]) 
f8,fs8 = sig.freqz(nfilt[7])

l1, = plt.plot(f1,20*np.log10(np.abs(fs1)+1e-6))
l2, = plt.plot(f2,20*np.log10(np.abs(fs2)+1e-6))
l3, = plt.plot(f3,20*np.log10(np.abs(fs3)+1e-6))
l4, = plt.plot(f4,20*np.log10(np.abs(fs4)+1e-6))
l5, = plt.plot(f5,20*np.log10(np.abs(fs5)+1e-6))
l6, = plt.plot(f6,20*np.log10(np.abs(fs6)+1e-6))
l7, = plt.plot(f7,20*np.log10(np.abs(fs7)+1e-6))
l8, = plt.plot(f8,20*np.log10(np.abs(fs8)+1e-6))

plt.title('optimized Window Frequency Response')
plt.legend(handles = [l1,l2,l3,l4,l5,l6,l7,l8,],labels = ['filter1','filter2','filter3','filter4','filter5','filter6','filter7','filter8'])
plt.xlabel('Normalized frequency')
plt.ylabel("Magnitude (dB)")
plt.show()


L = 128 # length of filter
n = np.arange(L)

w = np.kaiser(L,8) # Kaiser window

filt = np.zeros((N,L))  # filters

# L = 128, Kaiser window passband = 0.04pi
# After modulation, bandwidth is doubled, desired = 0.0625pi
# ideal passband = 0.0225pi 
filt0 = np.sin(0.0225*np.pi*(n-(L-1)/2))/(np.pi*(n-(L-1)/2))

# center is wc*(i+0.5)
for i in range(0,8):
   filt[i] = filt0*w*np.cos(n*wc*(i+0.5))

f1,fs1 = sig.freqz(filt[0])    # frequency response of filters
f2,fs2 = sig.freqz(filt[1])    
f3,fs3 = sig.freqz(filt[2]) 
f4,fs4 = sig.freqz(filt[3]) 
f5,fs5 = sig.freqz(filt[4]) 
f6,fs6 = sig.freqz(filt[5]) 
f7,fs7 = sig.freqz(filt[6]) 
f8,fs8 = sig.freqz(filt[7])

l1, = plt.plot(f1,20*np.log10(np.abs(fs1)+1e-6))
l2, = plt.plot(f2,20*np.log10(np.abs(fs2)+1e-6))
l3, = plt.plot(f3,20*np.log10(np.abs(fs3)+1e-6))
l4, = plt.plot(f4,20*np.log10(np.abs(fs4)+1e-6))
l5, = plt.plot(f5,20*np.log10(np.abs(fs5)+1e-6))
l6, = plt.plot(f6,20*np.log10(np.abs(fs6)+1e-6))
l7, = plt.plot(f7,20*np.log10(np.abs(fs7)+1e-6))
l8, = plt.plot(f8,20*np.log10(np.abs(fs8)+1e-6))

plt.title('Frequency response of filters with Kaiser window')
plt.legend(handles = [l1,l2,l3,l4,l5,l6,l7,l8,],labels = ['filter1','filter2','filter3','filter4','filter5','filter6','filter7','filter8'])
plt.xlabel('Normalized frequency')
plt.ylabel("dB")
plt.show()


# Analysis
# filtering

filtered1 = np.zeros((N,len(s1)))
for i in range(N):
   filtered1[i] = sig.lfilter(nfilt[i],1,s1)

# downsampling

ds = np.zeros((N,int(len(s1)/N)))
for i in range(N):
   ds[i] = filtered1[i,0::N]

#print("Playing downsampled subband...")
#sound(ds[2], int(frequency/N))

#Synthesis
# upsampling
us = np.zeros((N,len(s1)))
for i in range(N):
   us[i,0::N] = ds[i]

# filtering
filtered2 = np.zeros((N,len(s1)))
for i in range(N):
   filtered2[i] = sig.lfilter(nfilt[i],1,us[i])

# reconstructed signal
sys = filtered2[0]+filtered2[1]+filtered2[2]+filtered2[3]+filtered2[4]+filtered2[5]+filtered2[6]+filtered2[7]
#print("Playing reconstructed signal...")
#sound(sys, frequency)

s1 = s1/max(s1)
sys = sys/max(sys)
l1, = plt.plot(s1)
l2, = plt.plot(sys)
plt.legend(handles = [l1,l2,], labels = ['Original','Reconstructed'])
plt.title('Original signal and reconstructed signal')
plt.xlabel('Samples')
plt.ylabel('Sample values')
plt.show()

f1,fs1 = sig.freqz(s1)    # frequency response of filters
f2,fs2 = sig.freqz(sys)
l1, = plt.plot(f1,20*np.log10(np.abs(fs1)+1e-6))
l2, = plt.plot(f2,20*np.log10(np.abs(fs2)+1e-6))
plt.title('Frequency response of signals')
plt.legend(handles = [l1,l2,],labels = ['Original','Reconstructed'])
plt.xlabel('Normalized frequency')
plt.ylabel("dB")
plt.show()

