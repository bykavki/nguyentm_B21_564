import numpy as np
import matplotlib.pyplot as plt
import itertools


from scipy.signal import savgol_filter
from scipy.io import wavfile
from scipy import signal


dpi = 1000

def spectrogram(samples, sample_rate, filename):
    freq, t, spec = signal.spectrogram(samples, sample_rate, scaling = 'spectrum', window = ('hann'))

    spec = np.log10(spec+1)
    print(freq)
    plt.pcolormesh(t, freq, spec, shading='gouraud', vmin=spec.min(), vmax=spec.max())
    plt.ylabel('Частота [Гц]')
    plt.xlabel('Время [с]')

    plt.savefig(filename)

    return freq, t, spec


def butter_filter(samples, order):
    b, a = signal.butter(order, 0.1, btype='lowpass')
    filtered_signal = signal.filtfilt(b, a, samples)

    return filtered_signal

def get_peaks(freq, t, spec):

    peaks = set()
    delta_t = 0.1
    delta_freq = 200

    for i in range(len(freq)):
        for j in range(len(t)):
            index_t = np.asarray(abs(t-t[j]) < delta_t).nonzero()[0]
            index_freq = np.asarray(abs(freq-freq[i]) < delta_freq).nonzero()[0]
            indexes = np.array([x for x in itertools.product(index_freq, index_t)])
            flag = True
            for a, b in indexes:
                if spec[i, j] <= spec[a, b] and i != a and i != b:
                    flag = False
                    break
            
            if flag:
                peaks.add(t[j])
    
    return peaks
                                                





if __name__ == '__main__':
    
    sample_rate, samples = wavfile.read('input/def_not_rick_roll.wav')
    samples = samples[:, 1]
    denoised_butter = butter_filter(samples, 10)
    denoised_savgol = signal.savgol_filter(samples, 75, 5)


    wavfile.write('output/butter.wav', sample_rate, denoised_butter.astype(np.int16))
    wavfile.write('output/savgol.wav', sample_rate, denoised_savgol.astype(np.int16))

    freq, t, spec = spectrogram(samples, sample_rate, 'output/input.png')
    spectrogram(denoised_butter, sample_rate, 'output/butter.png')
    spectrogram(denoised_savgol, sample_rate, 'output/savgol.png')
    
    peaks = get_peaks(freq, t, spec)
    with open('output/peaks.txt', 'w') as f:
        f.write(str(peaks))
        f.write('\n')
        f.write(str(len(peaks)))


