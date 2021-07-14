import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from sound import *
import scipy.io.wavfile as wav

# reading audio
fs, s = wav.read("Track32.wav")
s1 = s[:,0]   #channel 1


N = 8 # number of subbands
L = 64 #Filter length
n = np.arange(L)
nd = (L-1)/2 #delay
fc = (1.0/N)*np.pi # lowpass bandwidth #end of pass band

kw = np.kaiser(L,8)

filt = np.zeros((N, L))  # filters

# ideal low pass filter with rectangular window
filt[0] = np.sin(fc*(n-nd))/(np.pi*(n-nd))
# High pass filter using Modulation
#filt[7] = highpass=np.sin(np.pi*(n-(L-1)/2))/(np.pi*(n-(L-1)/2))-np.sin(0.875*np.pi*(n-(L-1)/2))/(np.pi*(n-(L-1)/2))
filt[7]= filt[0]*kw*np.cos(np.pi * n)


# center is fc*(i+0.5) #Using Modulation
for i in range(1,7):
   filt[i] = filt[0]*kw * np.cos(n * fc * (i + 0.5))
   #filt[i] = filt0*kw*np.cos(n*wc*(i+0.5))


f1,fs1 = sig.freqz(filt[0])    # frequency response of filters
f2,fs2 = sig.freqz(filt[1])
f3,fs3 = sig.freqz(filt[2])
f4,fs4 = sig.freqz(filt[3])
f5,fs5 = sig.freqz(filt[4])
f6,fs6 = sig.freqz(filt[5])
f7,fs7 = sig.freqz(filt[6])
f8,fs8 = sig.freqz(filt[7])

p1 = plt.plot(f1,20*np.log10(np.abs(fs1)+1e-6))
p2 = plt.plot(f2,20*np.log10(np.abs(fs2)+1e-6))
p3 = plt.plot(f3,20*np.log10(np.abs(fs3)+1e-6))
p4 = plt.plot(f4,20*np.log10(np.abs(fs4)+1e-6))
p5 = plt.plot(f5,20*np.log10(np.abs(fs5)+1e-6))
p6 = plt.plot(f6,20*np.log10(np.abs(fs6)+1e-6))
p7 = plt.plot(f7,20*np.log10(np.abs(fs7)+1e-6))
p8 = plt.plot(f8,20*np.log10(np.abs(fs8)+1e-6))

plt.title('Frequency response of filters')
plt.legend(labels = ['filter1','filter2','filter3','filter4','filter5','filter6','filter7','filter8'])
plt.xlabel('Normalized frequency')
plt.ylabel("Magnitude (dB)")
plt.show()



#TASK#2
# rectangular window
w1 = np.ones(L)
#Hanning
w2 = 0.5 - 0.5*np.cos(2*np.pi/L*(n+0.5))
#Sine
w3 = np.sin(np.pi/L*(n+0.5))
#Kaiser
w4 = np.kaiser(L,8)
#Kaiser Besel
w5 = np.sin(np.pi/2*np.sin(np.pi/L*(n+0.5))**2)


h1 = filt[0]*w1
h2 = filt[0]*w2
h3 = filt[0]*w3
h4 = filt[0]*w4
h5 = filt[0]*w5

a1,as1 = sig.freqz(h1)
a2,as2 = sig.freqz(h2)
a3,as3 = sig.freqz(h3)
a4,as4 = sig.freqz(h4)
a5,as5 = sig.freqz(h5)

g1, = plt.plot(a1,20*np.log10(np.abs(as1)+1e-6))
g2, = plt.plot(a2,20*np.log10(np.abs(as2)+1e-6))
g3, = plt.plot(a3,20*np.log10(np.abs(as3)+1e-6))
g4, = plt.plot(a4,20*np.log10(np.abs(as4)+1e-6))
g5, = plt.plot(a5,20*np.log10(np.abs(as5)+1e-6))

plt.title('Frequency Response of filters of Different Window Types')
plt.legend(handles = [g1,g2,g3,g4,g5,],labels = ['Rectangular','Hanning','Sine','Kaiser','Kaiser Besel'])
plt.xlabel('Normalized frequency')
plt.ylabel("Magnitude (dB)")
plt.show()

#TASK 3


# Analysis filter bank
filtered1 = np.zeros((N,len(s1)))
for i in range(N):
   filtered1[i] = sig.lfilter(filt[i],1,s1)

# downsampling

ds = np.zeros((N,int(len(s1)/N)))
for i in range(N):
   ds[i] = filtered1[i,0::N]

#Synthesis filter bank
# upsampling
us = np.zeros((N,len(s1)))
for i in range(N):
   us[i,0::N] = ds[i]

# filtering
filtered2 = np.zeros((N,len(s1)))
for i in range(N):
   filtered2[i] = sig.lfilter(filt[i],1,us[i])

# reconstructed signal
sys = filtered2[0]+filtered2[1]+filtered2[2]+filtered2[3]+filtered2[4]+filtered2[5]+filtered2[6]+filtered2[7]

s1 = s1/max(s1)
sys = sys/max(sys)
l1, = plt.plot(s1, color = 'green')
l2, = plt.plot(sys, color = 'red')
plt.legend(handles = [l1,l2,], labels = ['Original','Reconstructed'])
plt.title('Original signal and reconstructed signal')
plt.show()
print(s1)