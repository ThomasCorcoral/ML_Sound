import librosa.display
import matplotlib.pyplot as plt

PATH = "../Data/train_bis/pipistrellus_pipistrellus/pipistr3.wav"
# PATH = "./Data/train_bis/pipistrellus_pipistrellus/Pipistrellus_pipistrellus_2.wav"


audio, sample_rate = librosa.load(PATH, res_type='kaiser_fast')
mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=100, hop_length=1024, htk=True)

spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128,
                                     fmax=11000, power=0.5)

print(len(spec[1]))
print(len(spec))

plt.figure(figsize=(10, 4))
librosa.display.specshow(spec, x_axis='time')
plt.colorbar()
plt.title('SPECTROGRAMME')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read

samplerate, data = read('../Data/train_bis/pipistrellus_pipistrellus/Pipistrellus_pipistrellus_2.wav')
duration = len(data)/samplerate
time = np.arange(0,duration,1/samplerate) #time vector

plt.plot(time,data)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('./Data/train_bis/pipistrellus_pipistrellus/Pipistrellus_pipistrellus_2.wav')
plt.show()