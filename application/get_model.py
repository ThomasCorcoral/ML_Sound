import os
import shutil
from keras.models import model_from_json
import zipfile as zf


# Is used to un-zip a compressed folder in a tmp_folder
def local_unzip(full_path, tmp_path):
    with zf.ZipFile(full_path, 'r') as zip_ref:
        return zip_ref.extractall(tmp_path)


# Is used to get all the folders inside the indicated folder, to get the species of the animals
def get_model(full_path):
    if zf.is_zipfile(full_path):
        os.mkdir("../local_unzip")
        tmp_path = "../local_unzip"
        local_unzip(full_path, tmp_path)
    else:
        tmp_path = full_path
    if os.path.isdir(tmp_path):
        files = os.listdir(tmp_path)
        h5_file = ""
        json_file = ""
        if len(files) != 2:
            print("Error : There can only be 2 files in the model directory.")
            return None
        for name in files:
            if name.lower().endswith('.h5') or name.lower().endswith('.json'):
                if name.lower().endswith('.h5'):
                    h5_file = name
                elif name.lower().endswith('.json'):
                    json_file = name
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
        shutil.rmtree(tmp_path)
        return model_local
    else:
        print("Error : Indicated path is not a directory")
        shutil.rmtree(tmp_path)
        return None
