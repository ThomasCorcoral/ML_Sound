# Load various imports
import pandas as pd
import os
import csv
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from numpy import save

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
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=50)
        mfccsscaled = np.mean(mfccs.T, axis=0)

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

SIZE = 8732

def feature_extraction(path, file_label):
    # Iterate through each sound file and extract the features
    # audio_files = get_audio_files(path)

    res = []
    train_labels = []

    for i in range(len(file_label)):
        file_path = path + "/fold" + str(file_label[i][1]) + "/" + str(file_label[i][0])
        data = extract_features_mfcc(file_path)
        res.append([data, file_label[i][2]])
        train_labels.append(file_label[i][2])
        if i % 873 == 0 :
            print(str(i/8730*100), " %")

    # Convert into a Panda dataframe
    featuresdf = pd.DataFrame(res, columns=['feature', 'class_label'])

    print('Extraction terminé de ', len(featuresdf), ' fichiers')
    # print('Taille des extractions : ', len(featuresdf['feature'][0]))

    return featuresdf, train_labels

PATH_CSV = "./UrbanSound8K/metadata/UrbanSound8K.csv"
PATH_TRAIN = "./UrbanSound8K/audio"

def get_infos():
    data = []
    with open(PATH_CSV) as csvDataFile:
        to_add = []
        cmpt=0
        csv_reader = csv.reader(csvDataFile)
        for row in csv_reader:
            if cmpt != 0:
                to_add.append(row[0])
                to_add.append(row[5])
                to_add.append(row[6])
                data.append(to_add)
            cmpt = cmpt + 1
            to_add = []
    return data

def conv_data():
    infos = get_infos()
    featuresdf, train_labels = feature_extraction(PATH_TRAIN, infos)

    # Convert features and corresponding classification labels into numpy arrays
    train_audio = np.array(featuresdf.feature.tolist())
    train_labels = np.asarray(train_labels).astype(np.float32)

    train_audio, test_audio, train_labels, test_labels = train_test_split(train_audio, train_labels, test_size=0.2, random_state = 42)

    print(len(test_audio))
    print(len(train_audio))

    return train_audio, train_labels, test_audio, test_labels

train_audio, train_labels, test_audio, test_labels = conv_data()

save('./arrays/train_audio.npy', train_audio)
save('./arrays/train_labels.npy', train_labels)
save('./arrays/test_audio.npy', test_audio)
save('./arrays/test_labels.npy', test_labels)
