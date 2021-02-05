import pandas as pd
import os
import numpy as np
import librosa
from pydub import AudioSegment
import matplotlib.pyplot as plt

NMFCC_MFCC = 50
NMELS_SPEC = 128

# Returns NMELS_SPEC (128)
def get_nmels_spec():
    return NMELS_SPEC


# Returns NMFCC_MFCC (50)
def get_nmfcc_mfcc():
    return NMFCC_MFCC


# Is used to get the audio files in the folder indicated in the parameters
def get_audio_files(ip_dir):
    matches = []
    for root, dirnames, filenames in os.walk(ip_dir):
        for filename in filenames:
            if filename.lower().endswith('.wav'):
                matches.append(os.path.join(root, filename))
    return matches


# Is used to get the mfcc of the sound file
def extract_features_mfcc(file_name):
    if not (os.path.isfile(file_name)):
        return None
    try:
        if file_name.endswith('.mp3'):
            audio, sample_rate = read_mp3(file_name)
        else:
            audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=NMFCC_MFCC)
        mfccsscaled = np.mean(mfccs.T, axis=0)

    except Exception:
        print("Error encountered while parsing file")
        return None

    return mfccsscaled


# Is used to get the spectrogram of the sound file
def extract_features_spec(file_name):
    if not (os.path.isfile(file_name)):
        return None
    try:
        if file_name.lower().endswith('.mp3'):
            audio, sample_rate = read_mp3(file_name)
        else:
            audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=NMELS_SPEC,
                                              fmax=11000, power=0.5)
        specsscaled = np.mean(spec.T, axis=0)
    except Exception:
        print("Error encountered while parsing file")
        return None
    return specsscaled


# Is used to transform a .mp3 into a .wav
def read_mp3(f):
    sound = AudioSegment.from_mp3(f)
    dst = '../local_saves/current.wav'
    sound.export(dst, format="wav")
    audio, sample_rate = librosa.load(dst, res_type='kaiser_fast')
    return audio, sample_rate


# Is used get the spectrogram or the mfcc of the indicated sound file
def feature_extraction(path, file_label, spec):
    res = []
    train_labels = []
    plt.ion()
    fig = plt.figure(num=None, figsize=(7, 2), dpi=80, facecolor='w', edgecolor='k')
    fig.canvas.set_window_title('Loading')
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.2)
    name = ["Loading"]
    ax.barh(name, [0], align='center', color='orange')
    ax.set_xlim([0, 100])
    ax.set_yticks(name)
    ax.set_yticklabels(name)
    plt.draw()

    for i in range(len(file_label)):

        file_path = path + "/" + str(file_label[i][1]) + "/" + str(file_label[i][0])
        percent = i / len(file_label) * 100
        actual = [percent]
        ax.clear()
        ax.barh(name, actual, align='center', color='orange')
        ax.set_xlim([0, 100])
        ax.set_yticks(name)
        ax.set_yticklabels(name)
        plt.draw()
        plt.pause(0.1)  # is necessary for the plot to update for some reason

        if spec:
            data = extract_features_spec(file_path)
        else:
            data = extract_features_mfcc(file_path)
        if data is None:
            return -1, file_path
        res.append([data, file_label[i][2]])
        train_labels.append(file_label[i][2])

    # Convert into a Panda dataframe
    featuresdf = pd.DataFrame(res, columns=['feature', 'class_label'])

    print('Extraction terminé de ', len(featuresdf), ' fichiers')
    plt.close()
    # print('Taille des extractions : ', len(featuresdf['feature'][0]))

    return featuresdf, train_labels