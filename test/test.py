from math import *
import os
import numpy as np
import librosa
from pydub import AudioSegment
import csv
from sklearn.model_selection import train_test_split
from numpy import save
import tensorflow as tf
from numpy import load
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

NMFCC_MFCC = 50
NMELS_SPEC = 128
SIZE_SEC = 43


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


# Convert a .mp3 to a .wav using pydub (with ffmpeg)
def read_mp3(f):
    sound = AudioSegment.from_mp3(f)
    dst = '../local_saves/current.wav'
    sound.export(dst, format="wav")
    audio, sample_rate = librosa.load(dst, res_type='kaiser_fast')
    return audio, sample_rate


def feature_extraction(file_name):
    if not (os.path.isfile(file_name)):
        return None
    try:
        # Check if the file is a .mp3
        if file_name.endswith('.mp3'):
            # Convert it with pydub
            audio, sample_rate = read_mp3(file_name)
        else:
            audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=NMFCC_MFCC)
    except Exception:
        print("Error encountered while parsing file")
        return None
    return mfccs, sample_rate, len(audio)


def process_the_audio(audio_path):
    # get the mfcc of the audio
    extr, sr, size = feature_extraction(audio_path)
    # values per second
    num_frame = len(extr[0])
    # duration of the audio
    audio_sec = size/sr
    # loss = audio_sec // (floor(audio_sec))
    # with this parameters 43
    var_per_sec = int(floor(num_frame/audio_sec))
    # loss = int(ceil(ceil((num_frame / audio_sec - var_per_sec) * audio_sec) / var_per_sec))
    loss = int(ceil(num_frame/var_per_sec) - ceil(audio_sec))
    # print(audio_sec)
    # print(num_frame/var_per_sec)
    # print(loss)
    # Initialisation of the results array
    prepared_audio = []
    # Add an initialise array for all
    if audio_sec < 1:
        prepared_audio.append([])
    # elif audio_sec == ceil(audio_sec):
    #     for i in range(int(audio_sec)):
    #         prepared_audio.append([])
    else:
        for i in range(int(ceil(audio_sec)+loss)):
            # print(i)
            prepared_audio.append([])
    # Change the NMFCC_MFCC if uses spectrogram
    for i in range(NMFCC_MFCC):
        for num, value in enumerate(extr[i], start=1):
            # Initialise the new array for the next values
            # print(num)
            if floor((num-1)/var_per_sec) == (num-1)/var_per_sec:
                # print((num-1)/var_per_sec)
                try:
                    prepared_audio[int((num-1)/var_per_sec)].append([])
                except IndexError:
                    print("Error with the file : " + audio_path + " / loss : " + str(loss) + " / indice : " +
                          str(int((num-1)/var_per_sec)))
            prepared_audio[int(floor((num-1)/var_per_sec))][i].append(value)
    # First we need to check if the audio is very short
    if len(prepared_audio[0][0]) != var_per_sec:
        print(len(prepared_audio[0][0]))
        local_size = len(prepared_audio[0][0])
        for i in range(len(prepared_audio[0])):
            for j in range(var_per_sec - local_size):
                prepared_audio[len(prepared_audio) - 1][i].append(0)
        return prepared_audio
    # Remove the last element if his length is lower than the half of the normal length
    if len(prepared_audio[len(prepared_audio)-1]) != 0:
        if len(prepared_audio[len(prepared_audio)-1][0]) < var_per_sec/2:
            prepared_audio.pop()
        else:# Else we add 0 at the end of each arrays
            local_size = len(prepared_audio[len(prepared_audio)-1][0])
            for i in range(len(prepared_audio[len(prepared_audio)-1])):
                for j in range(var_per_sec-local_size):
                    prepared_audio[len(prepared_audio)-1][i].append(0)
    # print(prepared_audio[len(prepared_audio)-1][0])
    return prepared_audio


def get_infos(path_csv):
    data = []
    with open(path_csv, newline='') as f:
        to_add = []
        reader = csv.DictReader(f)
        cmpt = 0
        for row in reader:
            name = row['name']
            fd = row['folder']
            label = row['class']
            to_add.append(name)
            to_add.append(fd)
            to_add.append(label)
            data.append(to_add)
            cmpt += 1
            to_add = []
    return data


def generate_labels(path_csv, path_txt='../local_saves/data_format/class_label.txt'):
    class_label = []
    with open(path_csv, newline='') as f:
        reader = csv.DictReader(f)
        with open(path_txt, 'w') as filehandle:
            for row in reader:
                filehandle.write('%s\n' % row["class_name"])
                class_label.append(row["class_name"])


def get_the_data(data_path, csv_path, label_text_path, ratio=0.1, rs=42):
    infos = get_infos(csv_path)
    res = []
    labels = []
    for i in range(len(infos)):
        file_path = data_path + "/" + str(infos[i][1]) + "/" + str(infos[i][0])
        current_label = infos[i][2]
        current_cut_audio = process_the_audio(file_path)
        for cutted in current_cut_audio:
            try:
                if len(cutted[0]) == SIZE_SEC:
                    res.append(cutted)
                    labels.append(current_label)
            except IndexError:
                print("Erreur avec fichier : " + file_path)
    # Conversion into numpy arrays
    labels = np.asarray(labels).astype(np.float32)
    generate_labels(csv_path, label_text_path)
    train_audio, test_audio, train_labels, test_labels = train_test_split(res, labels,
                                                                          test_size=ratio, random_state=rs)
    # save('./data_format/train_audio.npy', train_audio)
    # save('./data_format/train_labels.npy', train_labels)
    # save('./data_format/test_audio.npy', test_audio)
    # save('./data_format/test_labels.npy', test_labels)

    save('./data_format/res.npy', res)
    save('./data_format/labels.npy', labels)

    return res, labels


def my_model(size):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(50, 43)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, kernel_initializer='lecun_normal', activation='selu'),
        tf.keras.layers.Dense(128, kernel_initializer='lecun_normal', activation='elu'),
        tf.keras.layers.Dense(64, kernel_initializer='lecun_normal', activation='elu'),
        tf.keras.layers.Dense(32, kernel_initializer='lecun_normal', activation='selu'),
        tf.keras.layers.Dense(size, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def read_labels():
    class_label = []
    with open('./label_txt.txt', 'r') as filehandle:
        for line in filehandle:
            current_place = line[:-1]
            class_label.append(current_place)
    return class_label


def run_model(epoch=10):
    train_audio = load('../local_saves/data_format/train_audio.npy')
    train_labels = load('../local_saves/data_format/train_labels.npy')
    test_audio = load('../local_saves/data_format/test_audio.npy')
    test_labels = load('../local_saves/data_format/test_labels.npy')
    print("test audio size : " + str(len(test_audio)))
    print("test labels size : " + str(len(test_labels)))
    print("train audi size : " + str(len(train_audio)))
    print("train labels size : " + str(len(train_labels)))

    class_label = read_labels()
    class_label = list(dict.fromkeys(class_label))

    model = my_model(len(class_label))
    model.fit(train_audio, train_labels, epochs=epoch, verbose=0)
    score = model.evaluate(test_audio, test_labels, verbose=1)
    accuracy = 100 * score[1]
    json_file = model.to_json()
    with open("../local_saves/model/model.json", "w") as file:
        file.write(json_file)
    with open("../local_saves/accuracy.txt", "w") as file:
        file.write(str(accuracy))
    model.save_weights("../local_saves/model/model.h5")

    return accuracy, model


# ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type list).
if __name__ == "__main__":
    d_path = "../UrbanSound8K/audio"
    c_path = "../UrbanSound8K/metadata/UrbanSound8K.csv"
    lt_path = "./label_txt.txt"
    # data, lab = get_the_data(d_path, c_path, lt_path, ratio=0.1, rs=42)
    # prep = process_the_audio("./erreur_2.wav")

    # train_audio = load('./data_format/train_audio.npy', allow_pickle=True)
    # train_labels = load('./data_format/train_labels.npy', allow_pickle=True)
    # test_audio = load('./data_format/test_audio.npy', allow_pickle=True)
    # test_labels = load('./data_format/test_labels.npy', allow_pickle=True)

    data = load('./data_format/res.npy')
    lab = load('./data_format/labels.npy')

    features = []

    print(len(lab))

    features.append([data, lab])

    featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])

    X = np.array(featuresdf.feature.tolist())
    y = np.asarray(lab).astype(np.float32)

    print(len(y))

    # Encode the classification labels
    le = LabelEncoder()
    yy = to_categorical(le.fit_transform(y))

    print(len(X[0]))

    X = X[0]

    print(X.shape)

    # split the dataset
    x_train, x_test, y_train, y_test = train_test_split(data, lab, test_size=0.2, random_state=42)

    num_rows = 50
    num_columns = 43
    num_channels = 1

    class_label = read_labels()
    class_label = list(dict.fromkeys(class_label))
    epoch = 20
    model = my_model(len(class_label))
    model.fit(x_train, y_train, epochs=epoch)

    json_file = model.to_json()
    if not os.path.exists('./model'):
        os.mkdir('./model')
    with open("./model/model.json", "w") as file:
        file.write(json_file)
    model.save_weights("./model/model.h5")

    score = model.evaluate(x_test, y_test, verbose=1)
    accuracy = 100 * score[1]

    print("accuracy is : " + str(accuracy))

    prediction_feature = process_the_audio("./marteau_piqueur.mp3")
    class_label = read_labels()

    prediction_feature = np.array(prediction_feature)

    print("taille class_label : " + str(len(class_label)))

    class_label = list(dict.fromkeys(class_label))

    print(prediction_feature.shape)

    res = model.predict(prediction_feature, verbose=1, max_queue_size=len(class_label))

    res_for_each = []

    le = LabelEncoder()
    to_categorical(le.fit_transform(class_label))
    the_classes = list(le.classes_)
    print(the_classes)

    # for i in range(len(res[0])):
    #     s = 0
    #     for j in range(len(res)):
    #         s = s + res[j][i]
    #     s = s / len(res)
    #     res_for_each.append(s)

    for i in range(len(res[0])):
        s = []
        for j in range(len(res)):
            s.append(res[j][i])
        res_for_each.append(max(s))

    for i in range(len(res_for_each)):
        print("Prediction for class " + str(the_classes[i]) + " is " + str(floor(res_for_each[i] * 100)) + "%")
