from __future__ import print_function
import numpy as np
import os
import modules.mywavfile as mywavfile
import modules.cpu_detection as detector


def get_audio_files(ip_dir):
    matches = []
    for root, dirnames, filenames in os.walk(ip_dir):
        for filename in filenames:
            if filename.lower().endswith('.wav'):
                matches.append(os.path.join(root, filename))
    return matches


def read_audio(file_name, do_time_expansion, chunk_size, win_size):
    # try to read in audio file
    try:
        samp_rate_orig, audio = mywavfile.read(file_name)
    except:
        print('  Error reading file')
        return True, None, None, None, None

    # convert to mono if stereo
    if len(audio.shape) == 2:
        print('  Warning: stereo file. Just taking left channel.')
        audio = audio[:, 0]
    file_dur = audio.shape[0] / float(samp_rate_orig)
    print('  dur', round(file_dur, 3), '(secs) , fs', samp_rate_orig)

    # original model is trained on time expanded data
    samp_rate = samp_rate_orig
    if do_time_expansion:
        samp_rate = int(samp_rate_orig / 10.0)
        file_dur *= 10

    # pad with zeros so we can go right to the end
    multiplier = np.ceil(file_dur / float(chunk_size - win_size))
    diff = multiplier * (chunk_size - win_size) - file_dur + win_size
    audio_pad = np.hstack((audio, np.zeros(int(diff * samp_rate))))
    print(audio_pad)

    return False, audio_pad, file_dur, samp_rate, samp_rate_orig


def running_detector(opt, tme):
    # detection_thresh = 0.95        # make this smaller if you want more calls
    # do_time_expansion = True       # if audio is already time expanded set this to False
    # detection_thresh = dt
    # Etend le fichier pour mieux capter et reconnaitre les sons (x10)
    do_time_expansion = tme
    save_individual_results = False  # if True will create an output for each file
    # save_summary_result = True  # if True will create a single csv file with all results

    # load data
    data_dir = './to_find'

    # this is the path to your audio files
    op_ann_dir = 'results'  # this where your results will be saved
    op_ann_dir_ind = os.path.join(op_ann_dir, 'individual_results')  # this where individual results will be saved
    # op_file_name_total = os.path.join(op_ann_dir, 'op_file.csv')
    if not os.path.isdir(op_ann_dir):
        os.makedirs(op_ann_dir)
    if save_individual_results and not os.path.isdir(op_ann_dir_ind):
        os.makedirs(op_ann_dir_ind)

    # récupère le fichier .wav du dossier ./to_find
    audio_files = get_audio_files(data_dir)

    # load and create the detector
    if opt:
        det_model_file = 'models/detector_192K.npy'
    else:
        det_model_file = 'models/detector.npy'

    det_params_file = det_model_file[:-4] + '_params.json'
    det = detector.CPUDetector(det_model_file, det_params_file)

    # loop through audio files
    # results = []
    for file_cnt, file_name in enumerate(audio_files):

        file_name_basename = file_name[len(data_dir):]
        print('\n', file_cnt + 1, 'of', len(audio_files), '\t', file_name_basename)

        # read audio file - skip file if can't read it
        read_fail, audio, file_dur, samp_rate, samp_rate_orig = read_audio(file_name,
                                                                           do_time_expansion, det.chunk_size,
                                                                           det.win_size)
        if read_fail:
            continue

        write_log(to_str(audio))


def to_str(var):
    return str(list(np.reshape(np.asarray(var), (1, np.size(var)))[0]))[1:-1]

def write_log(res):
    fin_res = ""
    cmpt = 0
    for el in res.split():
        if el != "0.0," and el != "-1.0," and el != "-2.0,":
            fin_res += el + " "
            cmpt += 1
            if cmpt > 20:
                cmpt = 0
                fin_res += "\n"
    log_path = "log/log"
    fname = log_path + ".txt"
    cmpt = 1
    while os.path.isfile(fname):
        fname = log_path + str(cmpt) + ".txt"
        cmpt += 1
    f = open(fname, "a")
    f.write(fin_res)
    f.close()

##########################################
# MAIN
##########################################


if __name__ == "__main__":
    do_time_expansion = False# S'il faut faire une expansion du temps
    high_perf_audio = True# Si l'on souahite utiliser le détécteur 192K
    running_detector(high_perf_audio, do_time_expansion)