import pyaudio
import numpy as np
import pylab
import time
import librosa
from keras.models import model_from_json
from math import *
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical


RATE = 22500
CHUNK = int(RATE/43)
# RATE / number of updates per second
NMFCC_MFCC = 50


def feature_extraction(audio, sample_rate):
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=NMFCC_MFCC)
    return mfccs, sample_rate, len(audio)


def process_the_audio(audio):
    sr = RATE
    extr, sr, size = feature_extraction(audio, sr)
    num_frame = len(extr[0])
    audio_sec = size/sr
    # print("durée : " + str(audio_sec))
    var_per_sec = int(floor(num_frame/audio_sec))
    # print("v / sec : " + str(var_per_sec))
    var_per_sec = 43
    loss = int(ceil(num_frame/var_per_sec) - ceil(audio_sec))
    # print("loss : " + str(loss))
    prepared_audio = []
    for i in range(int(ceil(audio_sec)+loss)):
        prepared_audio.append([])
    for i in range(NMFCC_MFCC):
        for num, value in enumerate(extr[i], start=1):
            if floor((num-1)/var_per_sec) == (num-1)/var_per_sec:
                try:
                    prepared_audio[int((num-1)/var_per_sec)].append([])
                except IndexError:
                    print("Error with the stream, index : " + str(int((num-1)/var_per_sec)))
            prepared_audio[int(floor((num-1)/var_per_sec))][i].append(value)
    if len(prepared_audio[0][0]) != var_per_sec:
        # print(len(prepared_audio[0][0]))
        local_size = len(prepared_audio[0][0])
        for i in range(len(prepared_audio[0])):
            for j in range(var_per_sec - local_size):
                prepared_audio[len(prepared_audio) - 1][i].append(0)
        return prepared_audio
    if len(prepared_audio[len(prepared_audio)-1]) != 0:
        if len(prepared_audio[len(prepared_audio)-1][0]) < var_per_sec/2:
            prepared_audio.pop()
        else:
            local_size = len(prepared_audio[len(prepared_audio)-1][0])
            for i in range(len(prepared_audio[len(prepared_audio)-1])):
                for j in range(var_per_sec-local_size):
                    prepared_audio[len(prepared_audio)-1][i].append(0)
    return prepared_audio


def soundanalyse(stream, a_stream):
    data = np.frombuffer(stream.read(CHUNK), dtype=np.float32)
    # print("Longueur de data courant : " + str(len(data)))
    res = np.concatenate((data, a_stream), axis=None)
    return res


def read_labels():
    class_label = []
    with open('./label_txt.txt', 'r') as filehandle:
        for line in filehandle:
            current_place = line[:-1]
            class_label.append(current_place)
    return class_label


if __name__=="__main__":
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
    # Load model
    file = open("../local_saves/model/model.json", 'r')
    model_json = file.read()
    model = model_from_json(model_json)
    file.close()
    model.load_weights("../local_saves/model/model.h5")

    class_label = read_labels()
    class_label = list(dict.fromkeys(class_label))

    i = 0
    audio_stream = np.empty(0)
    while i < 10:
        audio_stream = soundanalyse(stream, audio_stream)
        i += 1
        prep_audio = process_the_audio(audio_stream)
        prep_audio = np.array(prep_audio)
        res = model.predict(prep_audio, verbose=1, max_queue_size=len(class_label))
        res_for_each = []
        le = LabelEncoder()
        to_categorical(le.fit_transform(class_label))
        the_classes = list(le.classes_)
        for i in range(len(res[0])):
            s = []
            for j in range(len(res)):
                s.append(res[j][i])
            res_for_each.append(max(s))
        for i in range(len(res_for_each)):
            print("Prediction for class " + str(the_classes[i]) + " is " + str(floor(res_for_each[i] * 100)) + "%")

    stream.stop_stream()
    stream.close()
    p.terminate()
