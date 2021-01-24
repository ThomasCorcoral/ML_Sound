import csv
import os


def deep_search(full_path, local_path):
    res = []
    list_of_dir = os.listdir(full_path)
    for to_add_path in list_of_dir:
        if os.path.isdir(full_path + '/' + to_add_path):
            to_append = deep_search(full_path + '/' + to_add_path, local_path + "/" + to_add_path)
            for rec_to_add in to_append:
                res.append(rec_to_add)
        else:
            res.append([to_add_path, local_path])
    return res


def generate(path):
    with open('../local_saves/auto_generate.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        # spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
        spamwriter.writerow(['name', 'folder', 'class', 'class_name'])
        list_of_dir = os.listdir(path)
        id_current = 0
        for to_add_path in list_of_dir:
            if os.path.isdir(path + '/' + to_add_path):
                to_add_rec = deep_search(path + '/' + to_add_path, to_add_path)
                for name, path_rec in to_add_rec:
                    spamwriter.writerow([name, path_rec, id_current, path_rec])
            else:
                spamwriter.writerow([to_add_path, '.'])
            id_current = id_current + 1