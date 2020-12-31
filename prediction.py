# import extraction_feature as ef
#
#
# def create_prediction(filename, model):
#     data = ef.extract_features_mfcc(filename)
#     predicted_vector = model.predict_classes(data)
#

import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from numpy import load
from keras.utils import to_categorical
import csv


def extract_feature(file_name):
    try:
        audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=50)
        mfccsscaled = np.mean(mfccs.T, axis=0)

    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None, None

    return np.array([mfccsscaled])


def read_labels():
    class_label = []
    with open('./local_npy_files/class_label.txt', 'r') as filehandle:
        for line in filehandle:
            currentPlace = line[:-1]
            class_label.append(currentPlace)
    return class_label


def print_prediction(file_name, model):
    prediction_feature = extract_feature(file_name)
    class_label = read_labels()
    print("taille class_label : " + str(len(class_label)))
    class_label = list(dict.fromkeys(class_label))
    le = LabelEncoder()
    yy = to_categorical(le.fit_transform(class_label))
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
