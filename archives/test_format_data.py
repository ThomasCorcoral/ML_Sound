from application.preparation_v1 import format_data as fd


def test_get_infos():
    csv_path = '../test/test.csv'
    infos = fd.get_infos(csv_path)
    assert len(infos) == 13


def test_get_infos_bad():
    csv_path = './test/bad.csv'
    check = False
    try:
        fd.get_infos(csv_path)
    except FileNotFoundError:
        check = True
    assert check


def read_labels(path):
    class_label = []
    with open(path, 'r') as filehandle:
        for line in filehandle:
            currentPlace = line[:-1]
            class_label.append(currentPlace)
    return class_label


def test_get_labels():
    csv_path = '../test/test.csv'
    file_txt = "../test/test.txt"
    infos = fd.get_labels(csv_path, file_txt)
    class_labels = read_labels(file_txt)
    assert len(class_labels) == 13


def test_get_labels_bad():
    csv_path = './test/bad.csv'
    check = False
    try:
        fd.get_labels(csv_path)
    except FileNotFoundError:
        check = True
    assert check
