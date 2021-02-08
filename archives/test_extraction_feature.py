from application.preparation_v1 import extraction_feature as ef
import csv
import numpy as np
import warnings


def test_extract_mfcc():
    test_file = '../test/gunshot.wav'
    data = ef.extract_features_mfcc(test_file)
    size = ef.get_nmfcc_mfcc()
    # Tableau 1D
    assert data.shape == (size,)
    assert len(data) == size


def test_extract_mfcc_bad():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        test_file = './test/bad.wav'
        data = ef.extract_features_mfcc(test_file)
        assert data is None


def test_extract_spec():
    test_file = '../test/gunshot.wav'
    data = ef.extract_features_spec(test_file)
    size = ef.get_nmels_spec()
    # Tableau 1D
    assert data.shape == (size,)
    assert len(data) == size


def test_extract_spec_bad():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        test_file = './test/bad.wav'
        data = ef.extract_features_spec(test_file)
        assert data is None


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
            to_add.append(name)
            to_add.append(fd)
            to_add.append(label)
            data.append(to_add)
            cmpt += 1
            to_add = []
    return data


# def write_erreur(wrt):
#     with open('./test/class_label.txt', 'w') as filehandle:
#         filehandle.write('%s\n' % wrt)


def test_extract_multiple_files_mfcc():
    folder_path = '../test/extraction'
    csv_path = '../test/test.csv'
    infos = get_infos(csv_path)
    size = len(infos)
    featuresdf, train_labels = ef.feature_extraction(folder_path, infos, False)
    train_audio = np.array(featuresdf.feature.tolist())
    assert len(train_audio) == size
    assert len(featuresdf) == size
    assert len(train_labels) == size


def test_extract_multiple_files_spec():
    folder_path = '../test/extraction'
    csv_path = '../test/test.csv'
    infos = get_infos(csv_path)
    size = len(infos)
    featuresdf, train_labels = ef.feature_extraction(folder_path, infos, True)
    train_audio = np.array(featuresdf.feature.tolist())
    assert len(train_audio) == size
    assert len(featuresdf) == size
    assert len(train_labels) == size


def test_extract_class_label():
    folder_path = '../test/extraction'
    csv_path = '../test/test.csv'
    infos = get_infos(csv_path)
    name_class_one = infos[0][2]
    featuresdf, train_labels = ef.feature_extraction(folder_path, infos, True)
    assert train_labels[0] == name_class_one
