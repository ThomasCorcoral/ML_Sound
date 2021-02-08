import tensorflow as tf
import librosa
import noisereduce as nr
from pydub import AudioSegment
from math import *
import numpy as np
import librosa.display
from numpy import save
from sklearn.model_selection import train_test_split
import csv
from numpy import load
import matplotlib.pyplot as plt
from keras.layers import Conv1D, AveragePooling1D, Conv2D, AveragePooling2D, MaxPooling2D, \
    Dropout, GlobalAveragePooling2D
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import model_from_json


NMFCC_MFCC = 50
NMELS_SPEC = 128
SIZE_SEC = 43


# Extract audio values and process noise reduction
def preprocess_audio(audio_path):
    try:
        dst = audio_path
        if audio_path.endswith('.mp3'):
            sound = AudioSegment.from_mp3(audio_path)
            dst = '../local_saves/current.wav'
            sound.export(dst, format="wav")
        audio_data, sample_rate = librosa.load(dst)
        noisy_part = audio_data[8000:10000]
        reduced_noise = nr.reduce_noise(audio_clip=audio_data,
                                        noise_clip=noisy_part, verbose=False)
        reduced_noise = np.asarray(reduced_noise)
    except Exception:
        return [], [], -1
    return reduced_noise, audio_data, sample_rate


# Get the mfccs, cut into 1 second length and select most noisy parts
def process_audio(audio_path):
    reduced_noise, basic_noise, sample_rate = preprocess_audio(audio_path)
    if sample_rate == -1:
        return np.asarray([])
    mfccs = librosa.feature.mfcc(y=reduced_noise, sr=sample_rate, n_mfcc=NMFCC_MFCC)
    # values per second
    num_frame = len(mfccs[0])

    # duration of the audio
    audio_sec = len(basic_noise)/sample_rate
    # with this parameters 43
    var_per_sec = int(floor(num_frame/audio_sec))
    loss = int(ceil(num_frame/var_per_sec) - ceil(audio_sec))
    all_file_mean = np.mean(mfccs)

    prepared_audio = []

    for i in range(int(ceil(audio_sec) + loss)):
        # print(i)
        prepared_audio.append([])
    # Change the NMFCC_MFCC if uses spectrogram
    for i in range(NMFCC_MFCC):
        for num, value in enumerate(mfccs[i], start=1):
            # Initialise the new array for the next values
            # print(num)
            if floor((num - 1) / var_per_sec) == (num - 1) / var_per_sec:
                # print((num-1)/var_per_sec)
                try:
                    prepared_audio[int((num - 1) / var_per_sec)].append([])
                except IndexError:
                    print("Error with the file : " + audio_path + " / loss : " + str(loss) + " / indice : " +
                          str(int((num - 1) / var_per_sec)))
            prepared_audio[int(floor((num - 1) / var_per_sec))][i].append(value)
    # First we need to check if the audio is very short
    if len(prepared_audio[0][0]) != var_per_sec:
        print(len(prepared_audio[0][0]))
        local_size = len(prepared_audio[0][0])
        for i in range(len(prepared_audio[0])):
            for j in range(var_per_sec - local_size):
                prepared_audio[len(prepared_audio) - 1][i].append(0)
        return prepared_audio
    # Remove the last element if his length is lower than the half of the normal length
    if len(prepared_audio[len(prepared_audio) - 1]) != 0:
        if len(prepared_audio[len(prepared_audio) - 1][0]) < var_per_sec / 2:
            prepared_audio.pop()
        else:  # Else we add 0 at the end of each arrays
            local_size = len(prepared_audio[len(prepared_audio) - 1][0])
            for i in range(len(prepared_audio[len(prepared_audio) - 1])):
                for j in range(var_per_sec - local_size):
                    prepared_audio[len(prepared_audio) - 1][i].append(0)
    res = []
    for i in range(len(prepared_audio)):
        if np.mean(prepared_audio[i]) > all_file_mean * 1.2:
            res.append(prepared_audio[i])
    return np.asarray(res)


# Get all the infos from the csv file through our norm
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


# Generate the class_label file with all the labels
def generate_labels(path_csv, path_txt='./class_label_u8.txt'):
    class_label = []
    with open(path_csv, newline='') as f:
        reader = csv.DictReader(f)
        with open(path_txt, 'w') as filehandle:
            for row in reader:
                filehandle.write('%s\n' % row["class_name"])
                class_label.append(row["class_name"])


# Get the data from a specific path with a csv
def get_the_data(data_path, csv_path, label_text_path, ratio=0.1, rs=42):
    infos = get_infos(csv_path)
    res = []
    labels = []
    ax, name = create_loading()
    for i in range(len(infos)):
        file_path = data_path + "/" + str(infos[i][1]) + "/" + str(infos[i][0])
        percent = i / len(infos) * 100
        update_loading(ax, name, percent)
        current_label = infos[i][2]
        current_cut_audio = process_audio(file_path)
        for cutted in current_cut_audio:
            try:
                if len(cutted[0]) == SIZE_SEC:
                    res.append(cutted)
                    labels.append(current_label)
            except IndexError:
                print("Erreur avec fichier : " + file_path)
    # Conversion into numpy arrays
    labels = np.asarray(labels).astype(np.float32)
    res = np.asarray(res)
    generate_labels(csv_path, label_text_path)
    train_audio, test_audio, train_labels, test_labels = train_test_split(res, labels,
                                                                          test_size=ratio, random_state=rs)
    save('./data_format/train_audio.npy', train_audio)
    save('./data_format/train_labels.npy', train_labels)
    save('./data_format/test_audio.npy', test_audio)
    save('./data_format/test_labels.npy', test_labels)

    # save('./data_format/res.npy', res)
    # save('./data_format/labels.npy', labels)

    return train_audio, test_audio, train_labels, test_labels


# Read line by line a .txt file
def read_labels(file_path):
    class_label = []
    with open(file_path, 'r') as filehandle:
        for line in filehandle:
            current_place = line[:-1]
            class_label.append(current_place)
    return class_label


def model_builder(size):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(50, 43)),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
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


def model_builder_bis(size):
    model = tf.keras.Sequential([
        Conv2D(32, 3, activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, activation='relu'),
        MaxPooling2D(),
        Conv2D(128, 3, activation='relu'),
        MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(size, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


def create_loading():
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
    return ax, name


def update_loading(ax, name, percent):
    actual = [percent]
    ax.clear()
    ax.barh(name, actual, align='center', color='orange')
    ax.set_xlim([0, 100])
    ax.set_yticks(name)
    ax.set_yticklabels(name)
    plt.draw()
    plt.pause(0.1)  # is necessary for the plot to update for some reason


# ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type list).
if __name__ == "__main__":
    d_path = "../UrbanSound8K/audio"
    c_path = "../UrbanSound8K/metadata/UrbanSound8K.csv"
    lt_path = "./class_label_u8.txt"
    # test line to only check the audio processing
    # process_audio("./Rougegorge familier.mp3")

    # line to create the .npy files
    # train_audio, test_audio, train_labels, test_labels = get_the_data(d_path, c_path, lt_path)

    train_audio = load('./data_format/train_audio.npy', allow_pickle=True)
    train_labels = load('./data_format/train_labels.npy', allow_pickle=True)
    test_audio = load('./data_format/test_audio.npy', allow_pickle=True)
    test_labels = load('./data_format/test_labels.npy', allow_pickle=True)

    print(train_audio.shape)

    num_rows = 50
    num_columns = 43
    num_channels = 1

    # train_audio = train_audio.reshape(train_audio.shape[0], num_rows, num_columns, num_channels)
    # test_audio = test_audio.reshape(test_audio.shape[0], num_rows, num_columns, num_channels)

    class_label = read_labels(lt_path)
    class_label = list(dict.fromkeys(class_label))
    epoch = 13
    model = model_builder(len(class_label))
    print("START FIT")
    model.fit(train_audio, train_labels, epochs=epoch, verbose=1)
    score = model.evaluate(test_audio, test_labels, verbose=1)
    accuracy = 100 * score[1]
    print(accuracy)

    path_json = "./model/model.json"
    path_h5 = "./model/model.h5"

    file = open(path_json, 'r')
    model_json = file.read()
    model = model_from_json(model_json)
    file.close()
    model.load_weights(path_h5)

    # prev = 0
    # prev_of_prev = 0
    # cmpt = 0
    # while prev_of_prev < accuracy:
    #     prev_of_prev = prev
    #     prev = accuracy
    #     model.fit(train_audio, train_labels, epochs=epoch, verbose=0)
    #     score = model.evaluate(test_audio, test_labels, verbose=1)
    #     accuracy = 100 * score[1]
    #     cmpt += 1
    # epoch = abs(cmpt - 2)
    # model = model_builder(len(class_label))
    # model.fit(train_audio, train_labels, epochs=epoch, verbose=1)
    # score = model.evaluate(test_audio, test_labels, verbose=1)
    # accuracy = 100 * score[1]
    #
    # print("Final accuracy : " + str(accuracy) + " / With " + str(epoch) + " epochs")

    prediction_feature = process_audio("./marteau_piqueur.mp3")

    le = LabelEncoder()
    to_categorical(le.fit_transform(class_label))
    print(list(le.classes_))
    predicted_vector = np.argmax(model.predict(prediction_feature), axis=-1)
    print("predicted_vector : " + str(predicted_vector[0]))
    predicted_class = le.inverse_transform(predicted_vector)
    print("The predicted class is:", predicted_class[0], '\n')
    predicted_proba_vector = model.predict(prediction_feature, verbose=1, max_queue_size=len(class_label))
    test = model.predict_on_batch(prediction_feature)
    print("test len : " + str(test[0]))
    predicted_proba = predicted_proba_vector[0]
    print("taille class_label : " + str(len(class_label)))
    print("taille predicted_proba_vector : " + str(len(predicted_proba_vector)))
    print("taille predicted_proba : " + str(len(predicted_proba)))

    all_prob = ""
    for i in range(len(predicted_proba)):
        category = le.inverse_transform(np.array([i]))
        current_classe = category[0]
        if type(current_classe) is tuple:
            current_classe = str(current_classe[0])
        prob = int(float(predicted_proba[i]) * 100)
        all_prob += current_classe + " : " + str(prob) + " %\n"
    print(all_prob)

