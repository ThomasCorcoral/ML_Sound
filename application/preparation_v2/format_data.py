import numpy as np
from application.preparation_v2 import extraction_feature as ef, progress_bar as pb, extract_infos as ei
from sklearn.model_selection import train_test_split
from numpy import save
import os


SIZE_SEC = 43


def get_the_data(data_path, csv_path, label_text_path, ratio=0.1, rs=42, mfcc=True):
    """Get the data from a specific path with a csv"""
    infos = ei.get_infos(csv_path)
    res, res_spec, labels, labels_spec = [], [], [], []
    ax, name, fig = pb.create_loading()
    for i in range(len(infos)):
        file_path = data_path + "/" + str(infos[i][1]) + "/" + str(infos[i][0])
        percent = i / len(infos) * 100
        if not pb.update_loading(ax, name, percent):
            return [], [], [], []
        current_label = infos[i][2]
        current_cut_audio, current_cut_audio_spec = ef.process_audio(file_path)
        for cutted in current_cut_audio:
            try:
                if len(cutted[0]) == SIZE_SEC:
                    res.append(cutted)
                    labels.append(current_label)
            except IndexError:
                print("Erreur avec fichier : " + file_path)
        for cutted in current_cut_audio_spec:
            try:
                if len(cutted[0]) == SIZE_SEC:
                    res_spec.append(cutted)
                    labels_spec.append(current_label)
            except IndexError:
                print("Erreur avec fichier : " + file_path)
    # Conversion into numpy arrays
    pb.close_loading(fig) # close progress bar
    labels, labels_spec = np.asarray(labels).astype(np.float32), np.asarray(labels_spec).astype(np.float32)
    res, res_spec = np.asarray(res), np.asarray(res_spec)
    ei.generate_labels(csv_path, label_text_path)
    train_audio_mfcc, test_audio_mfcc, train_labels_mfcc, test_labels_mfcc = train_test_split(res, labels,
                                                                                test_size=ratio, random_state=rs)
    train_audio_spec, test_audio_spec, train_labels_spec, test_labels_spec = train_test_split(res_spec, labels_spec,
                                                                                test_size=ratio, random_state=rs)
    if not os.path.exists('local_saves/data_format'):
        os.mkdir('local_saves/data_format')

    save('local_saves/data_format/train_audio_mfcc.npy', train_audio_mfcc)
    save('local_saves/data_format/train_labels_mfcc.npy', train_labels_mfcc)
    save('local_saves/data_format/test_audio_mfcc.npy', test_audio_mfcc)
    save('local_saves/data_format/test_labels_mfcc.npy', test_labels_mfcc)

    save('local_saves/data_format/train_audio_spec.npy', train_audio_spec)
    save('local_saves/data_format/train_labels_spec.npy', train_labels_spec)
    save('local_saves/data_format/test_audio_spec.npy', test_audio_spec)
    save('local_saves/data_format/test_labels_spec.npy', test_labels_spec)

    return train_audio_mfcc, train_labels_mfcc, test_audio_mfcc, test_labels_mfcc, train_audio_spec, train_labels_spec, test_audio_spec, test_audio_spec, test_labels_spec
