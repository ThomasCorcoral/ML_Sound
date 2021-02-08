from numpy import load
from application.preparation_v2 import model_builder as mb, extract_infos as ei


def get_best():
    epoch = 1
    train_audio = load('../local_saves/data_format/train_audio.npy')
    train_labels = load('../local_saves/data_format/train_labels.npy')
    test_audio = load('../local_saves/data_format/test_audio.npy')
    test_labels = load('../local_saves/data_format/test_labels.npy')

    class_label = ei.read_labels('../local_saves/data_format/class_label.txt')
    class_label = list(dict.fromkeys(class_label))

    model = mb.builder(len(class_label))
    model.fit(train_audio, train_labels, epochs=epoch, verbose=1)
    score = model.evaluate(test_audio, test_labels, verbose=1)
    accuracy = 100 * score[1]
    prev = 0
    prev_of_prev = 0
    cmpt = 0
    while prev_of_prev < accuracy:
        prev_of_prev = prev
        prev = accuracy
        model.fit(train_audio, train_labels, epochs=epoch, verbose=0)
        score = model.evaluate(test_audio, test_labels, verbose=1)
        accuracy = 100 * score[1]
        cmpt += 1
    epoch = abs(cmpt - 2)
    return epoch
