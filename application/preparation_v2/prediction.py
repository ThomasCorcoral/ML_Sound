import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from application.preparation_v2 import extract_infos as ei, extraction_feature as ef

NMFCC_MFCC = 50


def print_prediction(file_name, model, mfcc):
    """Prints the estimated specie for the sound and the general percentages"""
    prediction_feature_mfcc, prediction_feature_spec = ef.process_audio(file_name)

    if mfcc:
        prediction_feature = prediction_feature_mfcc
    else:
        prediction_feature = prediction_feature_spec

    if len(prediction_feature) == 0:
        return

    class_label = ei.read_labels("local_saves/data_format/class_label.txt")

    class_label = list(dict.fromkeys(class_label))
    le = LabelEncoder()
    to_categorical(le.fit_transform(class_label))

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
