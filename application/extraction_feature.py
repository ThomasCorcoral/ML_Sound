import pandas as pd
import os
import numpy as np
import librosa

NMFCC_MFCC = 50
NMELS_SPEC = 128


def get_nmels_spec():
    return NMELS_SPEC


def get_nmfcc_mfcc():
    return NMFCC_MFCC


def get_audio_files(ip_dir):
    matches = []
    for root, dirnames, filenames in os.walk(ip_dir):
        for filename in filenames:
            if filename.lower().endswith('.wav'):
                matches.append(os.path.join(root, filename))
    return matches


def extract_features_mfcc(file_name):
    if not (os.path.isfile(file_name)):
        return None
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=NMFCC_MFCC)
        mfccsscaled = np.mean(mfccs.T, axis=0)

    except Exception as e:
        print("Error encountered while parsing file")
        return None

    return mfccsscaled


def extract_features_spec(file_name):
    if not (os.path.isfile(file_name)):
        return None
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=NMELS_SPEC,
                                              fmax=11000, power=0.5)
        specsscaled = np.mean(spec.T, axis=0)
    except Exception as e:
        print("Error encountered while parsing file")
        return None
    return specsscaled


def feature_extraction(path, file_label, spec):
    res = []
    train_labels = []

    for i in range(len(file_label)):
        file_path = path + "/" + str(file_label[i][1]) + "/" + str(file_label[i][0])
        # print(file_path)
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
    # print('Taille des extractions : ', len(featuresdf['feature'][0]))

    return featuresdf, train_labels

#
# def feature_extraction(path, file_label):
#     res = []
#     train_labels = []
#
#     with open(file_label) as f:
#         reader = csv.DictReader(f, delimiter=';')
#         for row in reader:
#             name = row['name']
#             fd = row['folder']
#             classe = row['class']
#             file_path = path + "/" + fd + "/" + name
#             data = extract_features_mfcc(file_path)
#             res.append([data, classe])
#             train_labels.append(classe)
#
#     # Convert into a Panda dataframe
#     featuresdf = pd.DataFrame(res, columns=['feature', 'class_label'])
#
#     print('Extraction terminé de ', len(featuresdf), ' fichiers')
#
#     return featuresdf, train_labels
