from numpy import load
import os as os
from application.preparation_v2 import extract_infos as ei, model_builder as mb


def run_model(epoch=10):
    """Load all the data from local saves, read the labels of the prepared audio, then build the model, fit and
    evaluate it. Finaly the function save the model in local files"""
    train_audio = load('local_saves/data_format/train_audio.npy')
    train_labels = load('local_saves/data_format/train_labels.npy')
    test_audio = load('local_saves/data_format/test_audio.npy')
    test_labels = load('local_saves/data_format/test_labels.npy')
    print("test audio size : " + str(len(test_audio)))
    print("test labels size : " + str(len(test_labels)))
    print("train audi size : " + str(len(train_audio)))
    print("train labels size : " + str(len(train_labels)))

    class_label = ei.read_labels('local_saves/data_format/class_label.txt')
    class_label = list(dict.fromkeys(class_label))

    model = mb.builder(len(class_label))
    model.fit(train_audio, train_labels, epochs=epoch, verbose=0)
    score = model.evaluate(test_audio, test_labels, verbose=1)
    accuracy = 100 * score[1]
    json_file = model.to_json()
    if not os.path.exists('local_saves/model'):
        os.mkdir('local_saves/model')
    with open("local_saves/model/model.json", "w") as file:
        file.write(json_file)
    with open("local_saves/accuracy.txt", "w") as file:
        file.write(str(accuracy))
    model.save_weights("local_saves/model/model.h5")

    return accuracy, model
