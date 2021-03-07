import numpy as np
import librosa
from pydub import AudioSegment
from math import *
import noisereduce as nr

NMFCC_MFCC = 50


def preprocess_audio(audio_path):
    """Extract audio values and process noise reduction"""
    try:
        dst = audio_path
        if audio_path.endswith('.mp3'):
            sound = AudioSegment.from_mp3(audio_path)
            dst = './local_saves/current.wav'
            sound.export(dst, format="wav")
        audio_data, sample_rate = librosa.load(dst)
        noisy_part = audio_data[8000:10000]
        reduced_noise = nr.reduce_noise(audio_clip=audio_data,
                                        noise_clip=noisy_part, verbose=False)
        reduced_noise = np.asarray(reduced_noise)
    except Exception:
        return [], [], -1
    return reduced_noise, audio_data, sample_rate


def process_audio(audio_path, mfcc=True):
    """Get the mfccs, cut into 1 second length and select most noisy parts"""
    reduced_noise, basic_noise, sample_rate = preprocess_audio(audio_path)
    if sample_rate == -1:
        return np.asarray([]), np.asarray([])

    mfccs = librosa.feature.mfcc(y=reduced_noise, sr=sample_rate, n_mfcc=NMFCC_MFCC)
    spec = librosa.feature.melspectrogram(y=reduced_noise, sr=sample_rate, n_mels=NMFCC_MFCC, fmax=11000, power=0.5)
    # values per second
    num_frame = len(mfccs[0])

    # duration of the audio
    audio_sec = len(basic_noise)/sample_rate
    # with this parameters 43
    var_per_sec = int(floor(num_frame/audio_sec))
    loss = int(ceil(num_frame/var_per_sec) - ceil(audio_sec))
    all_file_mean = np.mean(mfccs)
    all_file_mean_spec = np.mean(spec)

    prepared_audio = []
    prepared_audio_spec = []

    for i in range(int(ceil(audio_sec) + loss)):
        # print(i)
        prepared_audio.append([])
        prepared_audio_spec.append([])
    # Change the NMFCC_MFCC if uses spectrogram
    for i in range(NMFCC_MFCC):
        for n in range(len(mfccs[i])):
            value = mfccs[i][n]
            value_spec = spec[i][n]
            num = n+1
            if floor((num - 1) / var_per_sec) == (num - 1) / var_per_sec:
                try:
                    prepared_audio[int((num - 1) / var_per_sec)].append([])
                    prepared_audio_spec[int((num - 1) / var_per_sec)].append([])
                except IndexError:
                    print("Error with the file : " + audio_path + " / loss : " + str(loss) + " / indice : " +
                          str(int((num - 1) / var_per_sec)))
            prepared_audio[int(floor((num - 1) / var_per_sec))][i].append(value)
            prepared_audio_spec[int(floor((num - 1) / var_per_sec))][i].append(value_spec)
    # First we need to check if the audio is very short
    if len(prepared_audio[0][0]) != var_per_sec:
        local_size = len(prepared_audio[0][0])
        for i in range(len(prepared_audio[0])):
            for j in range(var_per_sec - local_size):
                prepared_audio[len(prepared_audio) - 1][i].append(0)
                prepared_audio_spec[len(prepared_audio) - 1][i].append(0)
        return prepared_audio, prepared_audio_spec
    # Remove the last element if his length is lower than the half of the normal length
    if len(prepared_audio[len(prepared_audio) - 1]) != 0:
        if len(prepared_audio[len(prepared_audio) - 1][0]) < var_per_sec / 2:
            prepared_audio.pop()
            prepared_audio_spec.pop()
        else:  # Else we add 0 at the end of each arrays
            local_size = len(prepared_audio[len(prepared_audio) - 1][0])
            for i in range(len(prepared_audio[len(prepared_audio) - 1])):
                for j in range(var_per_sec - local_size):
                    prepared_audio[len(prepared_audio) - 1][i].append(0)
                    prepared_audio_spec[len(prepared_audio) - 1][i].append(0)
    res = []
    res_spec = []
    for i in range(len(prepared_audio)):
        if np.mean(prepared_audio[i]) > all_file_mean * 1.2:
            res.append(prepared_audio[i])
        if np.mean(prepared_audio_spec[i]) > all_file_mean_spec * 1.2:
            res.append(prepared_audio_spec[i])
    return np.asarray(res), np.asarray(prepared_audio_spec)
