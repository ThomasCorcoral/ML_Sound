import numpy as np
from application.preparation_v2 import extraction_feature as ef, progress_bar as pb, extract_infos as ei
from sklearn.model_selection import train_test_split
from numpy import save


SIZE_SEC = 43


# Get the data from a specific path with a csv
def get_the_data(data_path, csv_path, label_text_path, ratio=0.1, rs=42):
    infos = ei.get_infos(csv_path)
    res = []
    labels = []
    ax, name, fig = pb.create_loading()
    for i in range(len(infos)):
        file_path = data_path + "/" + str(infos[i][1]) + "/" + str(infos[i][0])
        percent = i / len(infos) * 100
        pb.update_loading(ax, name, percent)
        current_label = infos[i][2]
        current_cut_audio = ef.process_audio(file_path)
        for cutted in current_cut_audio:
            try:
                if len(cutted[0]) == SIZE_SEC:
                    res.append(cutted)
                    labels.append(current_label)
            except IndexError:
                print("Erreur avec fichier : " + file_path)
    # Conversion into numpy arrays
    pb.close_loading(fig)
    labels = np.asarray(labels).astype(np.float32)
    res = np.asarray(res)
    ei.generate_labels(csv_path, label_text_path)
    train_audio, test_audio, train_labels, test_labels = train_test_split(res, labels,
                                                                          test_size=ratio, random_state=rs)
    save('./data_format/train_audio.npy', train_audio)
    save('./data_format/train_labels.npy', train_labels)
    save('./data_format/test_audio.npy', test_audio)
    save('./data_format/test_labels.npy', test_labels)

    return train_audio, test_audio, train_labels, test_labels
