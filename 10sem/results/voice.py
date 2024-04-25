import librosa
import numpy as np
import matplotlib.pyplot as plt
import itertools


from scipy.io import wavfile
from scipy import signal


def get_max_tembr(filename):
    data, sample_rate = librosa.load(filename)   

    chroma = librosa.feature.chroma_stft(y=data, sr=sample_rate)    

    f0 = librosa.piptrack(y=data, sr=sample_rate, S=chroma)[0]
    
    max_f0 = np.argmax(f0)

    return max_f0




def spectrogram(samples, sample_rate, filename):
    freq, t, spec = signal.spectrogram(samples, sample_rate, window = ('hann'))

    spec = np.log10(spec+1)
    plt.pcolormesh(t, freq, spec, shading='gouraud', vmin=spec.min(), vmax=spec.max())
    plt.ylabel('Частота [Гц]')
    plt.xlabel('Время [с]')

    plt.savefig(filename)

    return freq, t, spec

def get_peaks(freq, t, spec):

    peaks = dict()
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
                peaks[freq[i]] = max(peaks.get(freq[i], 0), spec[i, j])
    
    max_3 = sorted(peaks.items(), key=lambda x: x[1], reverse=True)[:3]

    return list(map(lambda x: x[0], max_3))

def get_max_min(freq, t, spec):

    max_freq = -1000
    min_freq = 10**10

    for i in range(len(freq)):
        if any(spec[i, :] != 0):
            max_freq = max(max_freq, freq[i])
            min_freq = min(min_freq, freq[i])
    
    return max_freq, min_freq


if __name__ == "__main__":
    files = ['input/voice_a.wav', 'input/voice_i.wav', 'input/voice_gav.wav']
    rate_a, samples_a = wavfile.read(files[0])
    rate_i, samples_i = wavfile.read(files[1])
    rate_gav, samples_gav = wavfile.read(files[2])

    freq_a, t_a, spec_a = spectrogram(samples_a, rate_a, 'output/voice_a.png')
    freq_i, t_i, spec_i = spectrogram(samples_i, rate_i, 'output/voice_i.png')
    freq_gav, t_gav, spec_gav = spectrogram(samples_gav, rate_gav, 'output/voice_gav.png')

    with open('output/res.txt', 'w') as f:
        max_freq, min_freq = get_max_min(freq_a, t_a, spec_a)
        f.write(f"Минимальная и максимальная частота для а: {min_freq},  {max_freq}\n")
        max_freq, min_freq = get_max_min(freq_i, t_i, spec_i)
        f.write(f"Минимальная и максимальная частота для и: {min_freq},  {max_freq}\n")
        max_freq, min_freq = get_max_min(freq_gav, t_gav, spec_gav)
        f.write(f"Минимальная и максимальная частота для гав: {min_freq},  {max_freq}\n")
                                                                                 
        f.write(f"Наиболее тембрально окрашенный основной тон для а: {get_max_tembr(files[0])}\n")
        f.write(f"Наиболее тембрально окрашенный основной тон для и: {get_max_tembr(files[1])}\n")
        f.write(f"Наиболее тембрально окрашенный основной тон для гав: {get_max_tembr(files[2])}\n")

        f.write(f"Три самые сильные форманты для а: {get_peaks(freq_a, t_a, spec_a)}\n")
        f.write(f"Три самые сильные форманты для и: {get_peaks(freq_i, t_i, spec_i)}\n")
                







