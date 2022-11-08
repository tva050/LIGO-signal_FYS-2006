import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.signal as sc
import sklearn as sk
import matplotlib.gridspec as gridspec

plt.style.use("ggplot") # use ggplot style for plots

f_s = 4096.0 # Hz, sample rate
N = 131072 # number of samples

def read_data(file_name):
    h = h5py.File(file_name,"r")
    data = h["strain/Strain"][()]
    detector_name = h["meta/Detector"][()]
    start_time = h["meta/UTCstart"][()]
    h.close()
    return detector_name, start_time, data

# read hanford measurement
h1_name, h1_start_time, h1_strain = read_data("H-H1_LOSC_4_V2-1126259446-32.hdf5")
# read livingston measurement
l1_name, l1_start_time, l1_strain = read_data("L-L1_LOSC_4_V2-1126259446-32.hdf5")

""" ----------- task 1 ----------- """

def task_1():
    """ ----------- task 1b ----------- """
    # Number of samples in the signal
    print("----------- task 1b -----------")
    print("Number of samples in H1 data:", len(h1_strain), "samples")
    print("Number of samples in L1 data:", len(l1_strain), "samples")

    """ ----------- task 1c ----------- """
    # Length of the signals in seconds
    print("----------- task 1c -----------")
    print("Duration of H1 data:", len(h1_strain)/f_s, "seconds")
    print("Duration of L1 data:", len(l1_strain)/f_s, "seconds")

    """ ----------- task 1e ----------- """
    #sample spacing in units of seconds (inverse of sample rate)
    print("----------- task 1e -----------")
    print("Sample spacing in H1 data:", round(1/f_s, 6), "seconds")
    print("Sample spacing in L1 data:", round(1/f_s, 6), "seconds")


""" ----------- task 2 ----------- """

def task_2():
    """ ----------- task 2a ----------- """
    t = np.arange(0, 32 , 1/f_s) # seconds of each sample of the array signal
    # Plot the data
    plt.figure(1)
    plt.subplot(211)
    plt.plot(t, h1_strain, label = "Hanford")
    plt.ylabel("Strain")
    plt.title("Hanford and Livingston data")
    plt.legend(fancybox = True, shadow = True)
    plt.subplot(212)
    plt.plot(t, l1_strain, color = "teal", label = "Livingston")
    plt.xlabel("Time [s]")
    plt.ylabel("Strain")
    plt.legend(loc = "upper left", fancybox = True, shadow = True)
    plt.show()
    
    """ ----------- task 2b ----------- """
    print("----------- task 2b -----------")
    print("Minimum value of H1 data:", min(h1_strain))
    print("Maximum value of H1 data:", max(h1_strain))
    print("Mean value of H1 data:", np.mean(h1_strain))
    print("")
    print("Minimum value of L1 data:", min(l1_strain))
    print("Maximum value of L1 data:", max(l1_strain))
    print("Mean value of L1 data:", np.mean(l1_strain))


""" ---------------- Task 4a ---------------- """
# windowed FFT of data
def windowed(data):
    x_hat = np.fft.rfft(sc.hann(len(data))*data)
    return x_hat
def freqs(data):
    freqs = np.fft.rfftfreq(len(data), 1/f_s)
    return freqs


""" ---------------- Task 4b ---------------- """
# Plot of the power spectrum of the LIGO signals
def task_4b():
    plt.figure(1)
    plt.subplot(211)
    plt.plot(np.fft.fftshift(freqs(h1_strain)), np.fft.fftshift(10.0*np.log10(np.abs(windowed(h1_strain))**2.0)), label = "Hanford")
    plt.xlim(0, 2100)
    plt.ylabel("Power [dB]")
    plt.legend(loc = "upper left",fancybox = True, shadow = True)
    plt.subplot(212)
    plt.plot(np.fft.fftshift(freqs(l1_strain)), np.fft.fftshift(10.0*np.log10(np.abs(windowed(l1_strain))**2.0)), color = "teal", label = "Livingston")
    plt.xlim(0, 2100)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power [dB]")
    plt.legend(loc = "upper left", fancybox = True, shadow = True)
    plt.suptitle("Power spectrum of the LIGO signals", fontsize=14)
    plt.show()


""" ---------------- Task 5b ---------------- """
# whitening filter
def whitening_filter(data):
    h_hat = 1.0/np.abs(windowed(data))
    return h_hat


""" ---------------- Task 5c ---------------- """
# Whitened and filtered signal, gives the whitened filtered signal
def whitened_signal(data):
    h_hat = whitening_filter(data)
    x_hat = windowed(data)
    x_whitened = np.fft.irfft(h_hat*x_hat)
    return x_whitened


""" ---------------- Task 5d ---------------- """

def task_5d():
    t = np.arange(0, 32, (1/f_s)) 

    fig, axs = plt.subplots(3, sharex=False, sharey=True)
    axs[0].plot(t, whitened_signal(h1_strain) , label = "Hanford")
    axs[0].legend(loc="upper left")
    axs[1].plot(t, whitened_signal(l1_strain), color = "teal", label = "Livingston")
    axs[1].legend(loc="upper left")
    axs[2].plot(t, whitened_signal(h1_strain), label = "Hanford")
    axs[2].plot(t, whitened_signal(l1_strain), color = "teal", label = "Livingston")
    axs[2].set_xlim(16.2,16.5)
    axs[2].set_xlabel("Time [s]")
    fig.text(0.009, 0.5, "Whitened strain y[n]", va='center', rotation='vertical', color = "dimgray", fontsize = 12)
    fig.suptitle("Whitened signals of " r"$y_H[n]$ and $y_L[n]$", fontsize=14)
    plt.show()



""" ---------------- Task 6a ---------------- """

omhat = np.linspace(-np.pi,  np.pi , num=1000) # Angular frequency

f_hertz = omhat/(2*np.pi*(1/f_s)) # Convert to Hz

frequency_response = np.zeros(len(omhat), dtype = complex) # Initialize frequency response

# Calculate frequency response
L = 8 
for k in range(0,L):  
    frequency_response = frequency_response + (1/L)*np.exp(-1j*omhat*k)

""" ---------------- Task 6b ---------------- """
def task_6b():
    ax1 = plt.subplot(211)
    ax1.margins(0.05)           # Default margin is 0.05, value 0 means fit
    ax1.plot(f_hertz, 10.0*np.log10(np.abs(frequency_response)**2.0), label = "Frequency response")
    ax1.scatter([310], [-6])
    ax1.axvline(x = 300, color = "dimgray", linestyle = "--")
    ax1.axhline(y = -6, color = "dimgray", linestyle = "--")
    ax1.set_ylabel("Power [dB]")
    ax1.set_title("Power spectral response of the filter")

    ax2 = plt.subplot(212)
    ax2.plot(f_hertz, 10.0*np.log10(np.abs(frequency_response)**2.0), label = "Frequency response")
    ax2.scatter([310], [-6])
    ax2.axvline(x = 300, color = "dimgray", linestyle = "--")
    ax2.axhline(y = -6, color = "dimgray", linestyle = "--")
    ax2.set_ylabel("Power [dB]")
    ax2.set_xlabel("Frequnecy [Hz]")
    plt.show()

""" ---------------- Task 6d ----------------   """

tau = 8.5e-4 # s ,time delay

# Running mean average low-pass filter
def average_filter(data):
    # Filter signal using an averaging filter (y[n] = 1/L * sum_{k=0}^{L-1} x[n-k])
    # for the whitened signal.
    filtered_signal = np.convolve(np.repeat(1.0/L, L), data, mode = "same")
    return filtered_signal

""" ---------------- Task 6e ---------------- """

t = np.arange(0, 32, (1/f_s)) # Time vector
time = t - tau # Shifted time

""" ---------------- Task 6f ---------------- """

# Plot the filtered signals
def task_6f():
    fig, axs = plt.subplots(3, sharex=True, sharey=True)
    axs[0].plot(time, average_filter(whitened_signal(h1_strain)), label = "Hanford")
    axs[0].legend(loc="upper left", fancybox = True, shadow = True)
    axs[1].plot(time, average_filter(whitened_signal(l1_strain)), color = "teal", label = "Livingston")
    axs[1].legend(loc="upper left", fancybox = True, shadow = True)
    axs[2].plot(time, average_filter(whitened_signal(h1_strain)), label = "Hanford")
    axs[2].plot(time, average_filter(whitened_signal(l1_strain)), color = "teal", label = "Livingston")
    axs[2].set_xlim(16.1,16.6)
    axs[2].set_xlabel("Time [s]")
    fig.text(0.009, 0.5, "Strain", va='center', rotation='vertical', color = "dimgray", fontsize = 12)
    fig.suptitle("Low pass filtered whitened Hanford and Livingston signals", fontsize=14)
    plt.show()
    
    fig, axs = plt.subplots(2, sharex=True, sharey=True)
    axs[0].plot(t, whitened_signal(h1_strain), label = "Hanford")
    axs[0].plot(t, whitened_signal(l1_strain), color = "teal", label = "Livingston")
    axs[0].set_title("Whitened Hanford and Livingston signals")
    axs[0].legend(loc="upper left", fancybox = True, shadow = True)
    axs[1].plot(time, average_filter(whitened_signal(h1_strain)), label = "Hanford")
    axs[1].plot(time, average_filter(whitened_signal(l1_strain)), color = "teal", label = "Livingston")
    axs[1].set_xlim(16.1,16.6)
    axs[1].set_xlabel("Time [s]")
    axs[1].set_title(r"$\downarrow$Low pass filtered$\downarrow$")
    fig.text(0.009, 0.5, "Strain", va='center', rotation='vertical', color = "dimgray", fontsize = 12)
    plt.show()

    plt.plot(time, average_filter(whitened_signal(h1_strain)), label = "Hanford")
    plt.show()


""" ---------------- Task 7a ---------------- """
def time_delay(data1, data2):
    # Finding the time value for the highest peak in the two signals
    # than finding the corresponding index in the time vector.
    time_value_s1 = time[np.abs(average_filter(whitened_signal(h1_strain))).argmax()]
    time_value_s2 = time[np.abs(average_filter(whitened_signal(l1_strain))).argmax()]

    n_0 = time_value_s1 - time_value_s2 # Time delay
    New_time = time + n_0 # Shifted time vector with the time delay
    return New_time, print("The time delay is: ", n_0, "s")

def task_7a():
    plt.plot(time, np.abs(average_filter(whitened_signal(h1_strain))), label = "Hanford")
    plt.plot(time_delay(h1_strain, l1_strain), np.abs(average_filter(whitened_signal(l1_strain))), color = "teal", label = "Livingston")
    plt.xlim(16.4,16.425)
    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude (strain)")
    plt.legend(loc="upper left", fancybox = True, shadow = True)
    plt.show()



""" ---------------- Task 8a ---------------- """
# Defining each of the LGIO signal with Running mean average low-pass filter with whitened signal. 
filtered_l1= average_filter(whitened_signal(l1_strain))
filtered_h1 = average_filter(whitened_signal(h1_strain))
# Calculation for the dynamic spectrum of the low-pass filtered whitened signal.
f, t, filtered_h1 = sc.spectrogram(filtered_h1, fs=f_s, window="hann", nperseg=400, noverlap=380, nfft=4096)
f, t, filtered_l1 = sc.spectrogram(filtered_l1, fs=f_s, window="hann", nperseg=400, noverlap=380, nfft=4096)

""" ---------------- Task 8b ---------------- """

def task_8b():
    plt.pcolormesh(t, f, 10.0*np.log10(np.abs(filtered_h1)**2.0), vmin=-180, cmap="viridis", shading="auto")
    #plt.pcolormesh(t, f, 10.0*np.log10(np.abs(filtered_l1)**2.0), shading = "auto", cmap = "viridis", vmin= -180)
    plt.xlim(15.5,17)
    plt.ylim([0, 400.0])
    plt.colorbar(label="Power [dB]")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title("Dynamic spectrum Hann window")
    plt.show()


def task_8b_comp():
    # Create 2x2 sub plots
    gs = gridspec.GridSpec(2, 2)
    plt.figure()
    plt.subplot(gs[0, 0]) # row 0, col 0
    plt.pcolormesh(t, f, 10.0*np.log10(np.abs(filtered_h1)**2.0), vmin=-180, cmap="viridis", shading="auto")
    plt.xlim(15.5,17)
    plt.ylim([0, 400.0])
    plt.ylabel("Frequency [Hz]")
    plt.title("Hanford, Washington (H1)")

    plt.subplot(gs[0, 1]) # row 0, col 1
    plt.pcolormesh(t, f, 10.0*np.log10(np.abs(filtered_l1)**2.0), shading = "auto", cmap = "viridis", vmin= -180)
    plt.xlim(15.5,17)
    plt.ylim([0, 400.0])
    plt.colorbar(label="Power [dB]")
    plt.title("Livingston, Louisiana (L1)")

    plt.subplot(gs[1, :]) # row 1, span all columns
    plt.pcolormesh(t, f, 10.0*np.log10(np.abs(filtered_h1)**2.0), vmin=-180, cmap="viridis", shading="auto")
    plt.pcolormesh(t, f, 10.0*np.log10(np.abs(filtered_l1)**2.0), shading = "auto", cmap = "viridis", vmin= -150, alpha=0.5)
    plt.xlim(15.5,17)
    plt.ylim([0, 400.0])
    plt.ylabel("Frequency [Hz]")
    plt.show()


""" ---------------- Task 9 ---------------- """
def kaiser_window(data):
     x_hat = np.fft.rfft(sc.kaiser(len(data), beta=9)*data, 2*N)
     return x_hat


def bandpass_filter(data):
    
    lowcut = 25.0 # Hz
    highcut = 300.0 # Hz
    
    nyq = 0.5*f_s # Nyquist frequency
    low = lowcut/nyq # Normalized lowcut frequency
    high = highcut/nyq # Normalized highcut frequency

    order = 3 # order of the filter

    b, a = sc.butter(order, [low, high], btype="bandpass", analog=False) # Butterworth filter
    y = sc.filtfilt(b, a, data, axis = 0) # This function applies a linear digital filter twice, once forward and once backwards.
    return y # Returns a bandpass filtered signal 

# Defining each of the LGIO signal with Running mean average band-pass filter with whitened signal.
bandpass_filtered_h1 = bandpass_filter(whitened_signal(h1_strain))
# Calculation for the dynamic spectrum of the band-pass filtered whitened signal.
f, t, bandpass_filtered_h1 = sc.spectrogram(bandpass_filtered_h1, fs=f_s, window=("kaiser", 8.6), nperseg=400, noverlap=380, nfft=4096)

def power_spectrum_Bandfilt():
    plt.pcolormesh(t, f, 10.0*np.log10(np.abs(bandpass_filtered_h1)**2.0), vmin=-180, cmap="viridis", shading="auto")
    plt.xlim(15.5,17)
    plt.ylim([0, 400.0])
    plt.colorbar(label="Power [dB]")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title("Dynamic spectrum bandpass filtered signal")
    plt.show()

def comp_between_bandlow():
    # Plot of the low pass filtered whitened and band pass filered whitened LIGO signals
    plt.plot(time,  average_filter(whitened_signal(h1_strain)), label = "Low pass filtered signal")
    plt.plot(time, bandpass_filter(whitened_signal(h1_strain)), label = "Band pass filtered signal", color = "teal")
    plt.xlim(15.5,17)
    plt.xlabel("Time [s]")
    plt.ylabel("Strain")
    plt.title("Time domain signal of Hanford")
    plt.legend(loc="upper left", fancybox=True, shadow=True, borderpad=1)
    plt.show()
    
    # Power spectrum plot of the low pass filtered whitened and band pass filered whitened LIGO signals
    # Low pass filtered  Hanford signal
    plt.subplot(1, 2, 1) # row 1, col 2 index 1
    plt.pcolormesh(t, f, 10.0*np.log10(np.abs(filtered_h1)**2.0), vmin=-180, cmap="viridis", shading="auto")
    plt.title("Low pass filtered")
    plt.xlim(15.5,17)
    plt.ylim([0, 400.0])
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")

    # Band pass filtered Hanford signal
    plt.subplot(1, 2, 2) # index 2
    plt.pcolormesh(t, f, 10.0*np.log10(np.abs(bandpass_filtered_h1)**2.0), vmin=-180, cmap="viridis", shading="auto")
    plt.title("Band pass filtered")
    plt.xlim(15.5,17)
    plt.ylim([0, 400.0])
    plt.colorbar(label="Power [dB]")
    plt.xlabel("Time [s]")
    plt.savefig("bandpass_lowpass_filtered.png")
    plt.show()

if __name__ == "__main__":
    #task_1()   # Data
    #task_2()   # Plotting the data
    #task_4b()  # Plot of the power spectrum of the LIGO signals
    #task_5d()  # Plot of the whitened LIGO signals
    #task_6b()  # Plot of power spectral response of the filter
    #task_6f()  # Plot of the low pass filtered whitened LIGO signals
    #task_7a()  # Plot of the corrected time delay between the two signals 
    #task_8b()  # Plot of the dynamic spectrum of the LIGO signal 
    #task_8b_comp() # Plot of the dynamic spectrum of the LIGO signals and comparisson 
    #power_spectrum_Bandfilt() # Plot of the dynamic spectrum of the band-pass filtered whitened signal 
    #comp_between_bandlow() # Plot of the low pass filtered whitened and band pass filered whitened LIGO signals
    pass