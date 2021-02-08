import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from pydub import AudioSegment

NMFCC_MFCC = 50
NMELS_SPEC = 128


# Is used to get an array containing the results after they are compared (with mfcc or spectrogram)
def extract_feature(file_name, mfcc=True):
    try:
        if file_name.endswith('.mp3'):
            audio_data, sample_rate = read_mp3(file_name)
        else:
            audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        if mfcc:
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=NMFCC_MFCC)
            result = np.mean(mfccs.T, axis=0)
        else:
            spec = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=NMELS_SPEC,
                                                  fmax=11000, power=0.5)
            result = np.mean(spec.T, axis=0)

    except Exception:
        print("Error encountered while parsing file: ", file_name)
        return None, None

    return np.array([result])


# Is used to get the name of the animal in class_label.txt
def read_labels():
    class_label = []
    with open('../../local_saves/data_format/class_label.txt', 'r') as filehandle:
        for line in filehandle:
            current_place = line[:-1]
            class_label.append(current_place)
    return class_label


# Is used to transform a .mp3 into a.wav and get the needed data to have the mfcc or spectrogram
def read_mp3(f):
    sound = AudioSegment.from_mp3(f)
    dst = '../../local_saves/current.wav'
    sound.export(dst, format="wav")
    audio, sample_rate = librosa.load(dst, res_type='kaiser_fast')
    return audio, sample_rate


# Prints the estimated specie for the sound and the general percentages
def print_prediction(file_name, model, mfcc):
    prediction_feature = extract_feature(file_name, mfcc)
    class_label = read_labels()
    print("taille class_label : " + str(len(class_label)))
    class_label = list(dict.fromkeys(class_label))
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
    return predicted_class[0], le, predicted_proba