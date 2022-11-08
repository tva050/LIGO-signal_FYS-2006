import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as s

plt.style.use("ggplot")

f_s = 4096 # Hz, sample rate
f = 31.5 # Hz, frequency of the signal

n = np.arange(4096) # number of samples

x_n = np.cos(2*np.pi*f*n/f_s) # signal
w_hann = s.hann(len(n)) # Hann window of length 4096 a)
w_x = w_hann*x_n # Hann windowed signal

""" ----------------- task 3b ----------------- """
# plot the signal and the Hann window signal
plt.plot(x_n, label = "signal x[n]")
plt.plot(w_x, label = "Hann window w[n]x[n]")
plt.legend(bbox_to_anchor=(0, 1.1), loc=2, borderaxespad=0, fancybox = True, shadow = True) 
plt.show()

""" ----------------- task 3c ----------------- """

f_k = np.fft.fftfreq(f_s, 1/f_s) # frequency in Hz, array of length n containing the sample frequencies.
f_shifted = np.fft.fftshift(f_k) # frequency
print("Frequencies corresponding to principal spectrum:", f_shifted, "Hz")

""" ----------------- task 3d ----------------- """

# function to find the value of k corresponding to the frequency f_k nearest to -31.5 Hz and 31.5 Hz
def find_nearest(array, value):
    array = np.asarray(array) # convert array to numpy array
    idx = (np.abs(array - value)).argmin() # find index of the value in the array closest to the value
    return idx

print("k=", find_nearest(f_k, 31.5), "Hz, corresponds to the frequency 31.5 Hz")
print("k=",find_nearest(f_k, -31.5), "Hz, corresponds to the frequency -31.5 Hz")

""" ----------------- task 3e ----------------- """

plt.vlines(x=-31.5*2*np.pi/f_s, ymin=-150, ymax=55, linestyle = "--", color="black", lw = 1, label = r"$\pm31.5 $Hz")
plt.vlines(x=31.5*2*np.pi/f_s, ymin=-150, ymax=55, linestyle = "--", color="black", lw = 1)
plt.plot(f_shifted*2*np.pi/f_s, 10.0*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x_n)))**2.0), label = "DFT", color = "teal") 
plt.plot(f_shifted*2*np.pi/f_s, 10.0*np.log10(np.abs(np.fft.fftshift(np.fft.fft(w_x)))**2.0), label="Windowed DFT", color = "orangered")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power [dB]")
plt.legend(bbox_to_anchor=(0, 1.05),loc = "upper left", fancybox = True, shadow = True)
plt.show()



