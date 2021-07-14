import numpy as np
import scipy.signal as signal
import sound as snd
import matplotlib.pyplot as plt

# filter
N = 8 # num of subbands
#Low pass
#f1 = signal.remez(numtaps, [0, cutoff, cutoff + trans_width, 0.5*fs], [1, 0], Hz=fs)
f1 = signal.remez(64, [0,0.0625, 0.0627, 0.5], [1,0], [1, 100])# nyquist frequency  normalized to 0.5
#Band pass
#f2:f7 = signal.remez(numtaps, [0, band[0] - trans_width, band[0], band[1], band[1] + trans_width, 0.5*fs], [0, 1, 0], Hz=fs
f2 = signal.remez(64, [0, 0.0625, 0.0627, 0.125, 0.127, 0.5], [0, 1, 0], [100, 1, 100])
f3 = signal.remez(64, [0, 0.125, 0.127, 0.1875, 0.1877, 0.5], [0, 1, 0], [100, 1, 100])
f4 = signal.remez(64, [0, 0.1875, 0.1877, 0.25, 0.27, 0.5], [0, 1, 0], [100, 1, 100])
f5 = signal.remez(64, [0, 0.25, 0.27, 0.3125, 0.3127, 0.5], [0, 1, 0], [100, 1, 100])
f6 = signal.remez(64, [0, 0.3125, 0.3127, 0.375, 0.377, 0.5], [0, 1, 0], [100, 1, 100])
f7 = signal.remez(64, [0, 0.375, 0.377, 0.4375, 0.4377, 0.5], [0, 1, 0], [100, 1, 100])
#High pass
#f8 = signal.remez(numtaps, [0, cutoff - trans_width, cutoff, 0.5*fs],[0, 1], Hz=fs)
f8 = signal.remez(64, [0, 0.4375, 0.4377, 0.5], [0, 1], [100, 1])
# loading sound file as array
[s,rate] = snd.wavread('Track32.wav')

# Taking first chanel of audio
data = s[:, 0]


# filter implementation
filtered1 = signal.lfilter(f1,1,data)
filtered2 = signal.lfilter(f2,1,data)
filtered3 = signal.lfilter(f3,1,data)
filtered4 = signal.lfilter(f4,1,data)
filtered5 = signal.lfilter(f5,1,data)
filtered6 = signal.lfilter(f6,1,data)
filtered7 = signal.lfilter(f7,1,data)
filtered8 = signal.lfilter(f8,1,data)



filteredds1 = filtered1[::N]
filteredds2 = filtered2[::N]
filteredds3 = filtered3[::N]
filteredds4 = filtered4[::N]
filteredds5 = filtered5[::N]
filteredds6 = filtered6[::N]
filteredds7 = filtered7[::N]
filteredds8 = filtered8[::N]

# playing filtered sound of subband 1
#snd.sound(filtered1, 32000)

# playing filtered sound of subband 4
#snd.sound(filtered4, 32000)

# playing filtered sound after downsampling
#snd.sound(filteredds1, 4000)

# playing filtered sound after downsampling
#snd.sound(filteredds4, 4000)


# frequency response
w,H1 = signal.freqz(f1)
w,H4 = signal.freqz(f4)

# plotting Impulse response and frequency response
figure,(a1,a2) = plt.subplots(2)
a1.plot(f1)
a1.plot(f4)
a1.set_xlabel('Sample')
a1.set_ylabel('Value')
a1.legend(["f1", "f4"], loc ="lower right")
a1.set_title('Impulse Response ')

a2.plot(w,20*np.log10(np.abs(H1)))
a2.plot(w,20*np.log10(np.abs(H4)))
a2.set_xlabel('Normalized frequency')
a2.set_ylabel('Magnitude in dB')
a2.set_title('Magnitude Frequency response')

plt.show()


# up-sampling

filteredus1 = np.zeros(len(data))
filteredus2= np.zeros(len(data))
filteredus3 = np.zeros(len(data))
filteredus4 = np.zeros(len(data))
filteredus5 = np.zeros(len(data))
filteredus6 = np.zeros(len(data))
filteredus7 = np.zeros(len(data))
filteredus8 = np.zeros(len(data))

filteredus1[::N] = filteredds1
filteredus2[::N] = filteredds2
filteredus3[::N] = filteredds3
filteredus4[::N] = filteredds4
filteredus5[::N] = filteredds5
filteredus6[::N] = filteredds6
filteredus7[::N] = filteredds7
filteredus8[::N] = filteredds8



# synthesis filter
filteredsyn1 = signal.lfilter(f1,1,filteredus1)
filteredsyn2 = signal.lfilter(f2,1,filteredus2)
filteredsyn3 = signal.lfilter(f3,1,filteredus3)
filteredsyn4 = signal.lfilter(f4,1,filteredus4)
filteredsyn5 = signal.lfilter(f5,1,filteredus5)
filteredsyn6 = signal.lfilter(f6,1,filteredus6)
filteredsyn7 = signal.lfilter(f7,1,filteredus7)
filteredsyn8 = signal.lfilter(f8,1,filteredus8)

# playing reconstructed sound
recons = filteredsyn1+filteredsyn2+filteredsyn3+filteredsyn4+filteredsyn5+filteredsyn6+filteredsyn7+filteredsyn8
snd.sound(data, 32000)
snd.sound(recons, 32000)


#plt.plot(aud/max(aud))
#plt.plot(recons/(max(recons)))
#plt.show()





