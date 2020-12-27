# Load various imports
import pandas as pd
import os
import csv
import numpy as np
import librosa
import tensorflow as tf

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

SIZE = 8732

def feature_extraction(path, file_label):
    # Iterate through each sound file and extract the features
    # audio_files = get_audio_files(path)

    res = []
    train_labels = []

    for i in range(len(file_label)):
        file_path = path + "/fold" + str(file_label[i][1]) + "/" + str(file_label[i][0])
        data = extract_features_spec(file_path)
        res.append([data, file_label[i][2]])
        train_labels.append(file_label[i][2])
        if i == SIZE/4:
            print("25 %")
        if i == SIZE/2:
            print("50 %")
        if i == SIZE/4 + SIZE/2:
            print("75 %")

    # for file_cnt, file_name in enumerate(audio_files):
    #     data = extract_features_spec(file_name)
    #     res.append([data, file_label[file_cnt]])

    # Convert into a Panda dataframe
    featuresdf = pd.DataFrame(res, columns=['feature', 'class_label'])

    print('Extraction terminé de ', len(featuresdf), ' fichiers')
    # print('Taille des extractions : ', len(featuresdf['feature'][0]))

    return featuresdf, train_labels

# PATH_CSV = "./Data/dataset.csv"
# PATH_TRAIN = "./Data/train"
# Fichiers d'entrainement sans les warning
PATH_CSV = "./UrbanSound8K/metadata/UrbanSound8K.csv"
PATH_TRAIN = "./UrbanSound8K/audio"

# name, class_id, fold

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
                # print("taille to_add : ", len(to_add))
                data.append(to_add)
            cmpt = cmpt + 1
            to_add = []
        # print("taille element :", len(data[0]))
    return data

def conv_data():
    infos = get_infos()
    featuresdf, train_labels = feature_extraction(PATH_TRAIN, infos)

    # Convert features and corresponding classification labels into numpy arrays
    train_audio = np.array(featuresdf.feature.tolist())
    train_labels = np.asarray(train_labels).astype(np.float32)

    # # Encode the classification labels
    # le = LabelEncoder()
    # yy = to_categorical(le.fit_transform(y))
    #
    # # split the dataset
    from sklearn.model_selection import train_test_split

    train_audio, test_audio, train_labels, test_labels = train_test_split(train_audio, train_labels, test_size=0.2, random_state = 42)

    print(len(test_audio))
    print(len(train_audio))

    return train_audio, train_labels, test_audio, test_labels

train_audio, train_labels, test_audio, test_labels = conv_data()

corresp_labels = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling", "engine_idling",
                    "gun_shot", "jackhammer", "siren", "street_music"]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128),
    tf.keras.layers.Dense(128),
    tf.keras.layers.Dense(64),
    tf.keras.layers.Dense(32)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

model.fit(train_audio, train_labels, epochs=100)

score = model.evaluate(test_audio, test_labels, verbose=1)
accuracy = 100*score[1]

print("Pre-training accuracy: %.4f%%" % accuracy)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128),
    tf.keras.layers.Dense(128),
    tf.keras.layers.Dense(64),
    tf.keras.layers.Dense(32)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

model.fit(train_audio, train_labels, epochs=10)

score = model.evaluate(test_audio, test_labels, verbose=1)
accuracy = 100*score[1]

print("Pre-training accuracy: %.4f%%" % accuracy)