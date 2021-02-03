import librosa
from math import *


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