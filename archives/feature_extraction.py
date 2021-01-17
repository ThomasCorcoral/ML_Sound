# Load various imports
import pandas as pd
import os
import numpy as np
import librosa


def get_audio_files(ip_dir):
    matches = []
    for root, dirnames, filenames in os.walk(ip_dir):
        for filename in filenames:
            if filename.lower().endswith('.wav'):
                matches.append(os.path.join(root, filename))
    return matches


def extract_features_mfcc(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        # 128 est la taille max pour n_mfcc
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=50)
        # print("mfccs row : ", len(mfccs), " / col : ", len(mfccs[0]), " / mfccs.T : ", len(mfccs.T))
        mfccsscaled = np.mean(mfccs.T, axis=0)
        # print("mfccsscaled size : ", len(mfccsscaled))

    except Exception as e:
        print("Error encountered while parsing file")
        return None

    return mfccsscaled


def extract_features_spec(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128,
                                              fmax=11000, power=0.5)

        specsscaled = np.mean(spec.T, axis=0)

    except Exception as e:
        print("Error encountered while parsing file")
        return None

    return specsscaled


def feature_extraction(path, file_label):
    # Iterate through each sound file and extract the features
    audio_files = get_audio_files(path)

    res = []

    for file_cnt, file_name in enumerate(audio_files):
        print(file_name)
        data = extract_features_spec(file_name)
        res.append([data, file_label[file_cnt]])

    # Convert into a Panda dataframe
    featuresdf = pd.DataFrame(res, columns=['feature', 'class_label'])

    print('Extraction terminé de ', len(featuresdf), ' fichiers')
    # print('Taille des extractions : ', len(featuresdf['feature'][0]))

    return featuresdf
