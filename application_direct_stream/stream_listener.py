import pyaudio
import numpy as np
import pylab
import time
import librosa
from keras.models import model_from_json
from math import *
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import new_audio_process as nap


RATE = 22500
CHUNK = int(RATE/43)
# RATE / number of updates per second
NMFCC_MFCC = 50


def soundanalyse(stream, a_stream):
    data = np.frombuffer(stream.read(CHUNK), dtype=np.float32)
    print("Longueur de data courant : " + str(len(data)))
    res = np.concatenate((data, a_stream), axis=None)
    return res


def read_labels():
    class_label = []
    with open('./label_txt.txt', 'r') as filehandle:
        for line in filehandle:
            current_place = line[:-1]
            class_label.append(current_place)
    return class_label


def listen_stream():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
    i = 0
    audio_stream = np.empty(0)

    while i < 10:
        audio_stream = soundanalyse(stream, audio_stream)
        i += 1
        prep_audio = nap.process_the_audio(audio_stream)
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
