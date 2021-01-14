import numpy as np
import csv
import feature_extraction as fe

# PATH_CSV = "./Data/dataset.csv"
# PATH_TRAIN = "./Data/train"
# Fichiers d'entrainement sans les warning
PATH_CSV = "../Data/dataset_bis.csv"
PATH_TRAIN = "../Data/train_bis"
TRAIN_SIZE = 3

def get_ids():
    data = []
    with open(PATH_CSV) as csvDataFile:
        csv_reader = csv.reader(csvDataFile)
        for row in csv_reader:
            if row[2] != 'id':
                data.append(int(row[2]))
    return data

def conv_data():
    train_labels = get_ids()
    featuresdf = fe.feature_extraction(PATH_TRAIN, train_labels)

    # Convert features and corresponding classification labels into numpy arrays
    train_audio = np.array(featuresdf.feature.tolist())
    train_labels = np.asarray(train_labels).astype(np.float32)

    # # Encode the classification labels
    # le = LabelEncoder()
    # yy = to_categorical(le.fit_transform(y))
    #
    # # split the dataset
    # from sklearn.model_selection import train_test_split
    #
    # x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)
    #
    # #print(len(x_test))

    print(len(train_audio[0]))

    return train_audio, train_labels