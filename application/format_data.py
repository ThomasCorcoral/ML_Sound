import numpy as np
import csv
import extraction_feature as ef
from sklearn.model_selection import train_test_split
from numpy import save


# Is used to get the informations contained in the .csv indicatedd by the user
def get_infos(path_csv):
    data = []
    with open(path_csv, newline='') as f:
        to_add = []
        reader = csv.DictReader(f)
        cmpt = 0
        for row in reader:
            name = row['name']
            fd = row['folder']
            label = row['class']
            # print("name : " + name + " / folder : " + fd + " / label : " + label)
            to_add.append(name)
            to_add.append(fd)
            to_add.append(label)
            data.append(to_add)
            # print("name : " + data[cmpt][0] + " / folder : " + data[cmpt][1] + " / label : " + data[cmpt][2])
            cmpt += 1
            to_add = []
    return data


# Is used to get the name (labels) of the bats : birds and store them in a .txt (used by the programm)
def get_labels(path_csv, path_txt='../local_saves/data_format/class_label.txt'):
    class_label = []
    with open(path_csv, newline='') as f:
        reader = csv.DictReader(f)
        with open(path_txt, 'w') as filehandle:
            for row in reader:
                filehandle.write('%s\n' % row["class_name"])
                class_label.append(row["class_name"])


# Is used to get the data in the .csv, get the sound data and names, transform them in .npy, then get train_audio,
# train_label, test_audio and test_label and get the labels
def conv_data(path_data, path_csv, ratio=0.1, rs=42, spec=False):
    infos = get_infos(path_csv)
    featuresdf, train_labels = ef.feature_extraction(path_data, infos, spec)

    if type(featuresdf) is int:
        print("Error : file not found : " + train_labels)
        return -1

    # Conversion des tableaux en tableaux Numpy
    train_audio = np.array(featuresdf.feature.tolist())
    train_labels = np.asarray(train_labels).astype(np.float32)

    train_audio, test_audio, train_labels, test_labels = train_test_split(train_audio, train_labels,
                                                                          test_size=ratio, random_state=rs)

    # print(len(test_audio))
    # print(len(train_audio))

    save('../local_saves/data_format/train_audio.npy', train_audio)
    save('../local_saves/data_format/train_labels.npy', train_labels)
    save('../local_saves/data_format/test_audio.npy', test_audio)
    save('../local_saves/data_format/test_labels.npy', test_labels)
    get_labels(path_csv)

    return 0
