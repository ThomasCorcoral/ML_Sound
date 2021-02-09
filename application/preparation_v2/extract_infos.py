import csv


def generate_labels(path_csv, path_txt):
    """Generate the class_label file with all the labels"""
    class_label = []
    with open(path_csv, newline='') as f:
        reader = csv.DictReader(f)
        with open(path_txt, 'w') as filehandle:
            for row in reader:
                filehandle.write('%s\n' % row["class_name"])
                class_label.append(row["class_name"])


def read_labels(file_path):
    """Read line by line a .txt file"""
    class_label = []
    with open(file_path, 'r') as filehandle:
        for line in filehandle:
            current_place = line[:-1]
            class_label.append(current_place)
    return class_label


def get_infos(path_csv):
    """Get all the infos from the csv file through our norm"""
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