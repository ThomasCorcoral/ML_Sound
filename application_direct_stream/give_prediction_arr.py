from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical


def give_pred(model, prep_audio, class_label, maxi=True):
    res = model.predict(prep_audio, verbose=1, max_queue_size=len(class_label))
    res_for_each = []
    le = LabelEncoder()
    to_categorical(le.fit_transform(class_label))
    the_classes = list(le.classes_)
    for i in range(len(res[0])):
        s = []
        for j in range(len(res)):
            s.append(res[j][i])
        if maxi:
            res_for_each.append(max(s))
        else:
            res_for_each.append(sum(s) / len(s))
            print(sum(s) / len(s))
    return res_for_each
