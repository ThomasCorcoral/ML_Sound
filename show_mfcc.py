import librosa.display
import matplotlib.pyplot as plt

# PATH = "./Data/train_bis/pipistrellus_pipistrellus/pipistr3.wav"
PATH = "./Data/train_bis/pipistrellus_pipistrellus/pipistr3.wav"


audio, sample_rate = librosa.load(PATH, res_type='kaiser_fast')
mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40, hop_length=1024, htk=True)

spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128,
                                     fmax=11000, power=0.5)

print(len(spec))
print(len(mfccs))

plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()