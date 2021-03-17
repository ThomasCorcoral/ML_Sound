import os
import shutil
from keras.models import model_from_json
import zipfile as zf


def local_unzip(full_path, tmp_path):
    """Is used to un-zip a compressed folder in a tmp_folder"""
    with zf.ZipFile(full_path, 'r') as zip_ref:
        return zip_ref.extractall(tmp_path)


def get_model(full_path):
    """Is used to get all the folders inside the indicated folder, to get the species of the animals"""
    if zf.is_zipfile(full_path):
        os.mkdir("./local_unzip")
        tmp_path = "./local_unzip"
        local_unzip(full_path, tmp_path)
    else:
        tmp_path = full_path
    if os.path.isdir(tmp_path):
        files = os.listdir(tmp_path)
        h5_file = ""
        json_file = ""
        acc_file = ""
        if len(files) != 3:
            print("Error : There can only be 3 files in the model directory.")
            return None
        for name in files:
            print(name)
            if name.lower().endswith('.h5'):
                h5_file = name
            elif name.lower().endswith('.json'):
                json_file = name
            elif name.lower().endswith('.txt'):
                acc_file = name
            else:
                print("Error : Incorrect model, the model should only have a .h5 and a .json in it")
                return None

        try:
            # load json and create model
            file = open(tmp_path + "/" + json_file, 'r')
        except IOError:
            print("File not accessible")
            return None
        model_json = file.read()
        model_local = model_from_json(model_json)
        file.close()
        # load weights
        model_local.load_weights(tmp_path + "/" + h5_file)

        acc_file = open(tmp_path + "/" + acc_file, 'r')
        acc_str = acc_file.read()
        accuracy = int(float(acc_str))
        acc_file.close()

        shutil.rmtree(tmp_path)

        return model_local, accuracy
    else:
        print("Error : Indicated path is not a directory")
        shutil.rmtree(tmp_path)
        return None
