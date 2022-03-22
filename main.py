import re
import numpy as np
import time
from sklearn.cluster import AffinityPropagation, KMeans
from scipy import stats
import argparse
from transformers import logging
import recasepunc
from recasepunc import CasePuncPredictor
from recasepunc import WordpieceTokenizer
from recasepunc import Config
from vosk import Model, KaldiRecognizer, SpkModel
import json
import wave
import streamlit as st
from pydub import AudioSegment


def do_itog():
    f1 = open('text_punct.txt', 'r')
    f2 = open('output_dia.txt', 'r')
    f = open('text_itog.txt', 'w')

    for line in f2:
        f.write(f1.read(len(line)) + '\n')
    f.close()
    f = open('text_itog.txt', 'r')
    text = f.readlines()
    st.write(text)
    f.close()
    f1.close()
    f2.close()
    #st.write("text is ready")


def punct_text():
    logging.set_verbosity_error()

    predictor = CasePuncPredictor('checkpoint', lang="ru")

    text = " ".join(open('text.txt', 'r').readlines())
    tokens = list(enumerate(predictor.tokenize(text)))

    results = ""
    for token, case_label, punc_label in predictor.predict(tokens, lambda x: x[1]):
        prediction = predictor.map_punc_label(predictor.map_case_label(token[1], case_label), punc_label)
        if token[1][0] != '#':
            results = results + ' ' + prediction
        else:
            results = results + prediction

    print(results.strip())
    f = open('text_punct.txt', 'w')
    for index in results.strip():
        f.write(index)
    f.close()
    #st.write("Punctuation is ready")


def base_text(wf, model):
    spk_model = SpkModel(spk_model_path)

    rcgn_fr = wf.getframerate() * wf.getnchannels()
    rec = KaldiRecognizer(model, rcgn_fr)
    result = ''
    last_n = False
    # read_block_size = 4000
    read_block_size = wf.getnframes()
    while True:
        data = wf.readframes(read_block_size)
        if len(data) == 0:
            break

        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())

            if res['text'] != '':
                result += f" {res['text']}"
                if read_block_size < 200000:
                    print(res['text'] + " \n")

                last_n = False
            elif not last_n:
                result += '\n'
                last_n = True

    res = json.loads(rec.FinalResult())
    result += f" {res['text']}"

    ft = open('text.txt', 'w')
    for index in result:
        ft.write(index)
    ft.close()

    print('\n'.join(line.strip() for line in re.findall(r'.{1,150}(?:\s+|$)', result)))
    #st.write("Text parsing is ready")
    # st.write('\n'.join(line.strip() for line in re.findall(r'.{1,150}(?:\s+|$)', result)))


def dia_text(wf, model, spk_model):
    spk_sig = [-1.110417, 0.09703002, 1.35658, 0.7798632, -0.305457, -0.339204, 0.6186931, -0.4521213, 0.3982236,
               -0.004530723, 0.7651616, 0.6500852, -0.6664245, 0.1361499, 0.1358056, -0.2887807, -0.1280468, -0.8208137,
               -1.620276, -0.4628615, 0.7870904, -0.105754, 0.9739769, -0.3258137, -0.7322628, -0.6212429, -0.5531687,
               -0.7796484, 0.7035915, 1.056094, -0.4941756, -0.6521456, -0.2238328, -0.003737517, 0.2165709, 1.200186,
               -0.7737719, 0.492015, 1.16058, 0.6135428, -0.7183084, 0.3153541, 0.3458071, -1.418189, -0.9624157,
               0.4168292, -1.627305, 0.2742135, -0.6166027, 0.1962581, -0.6406527, 0.4372789, -0.4296024, 0.4898657,
               -0.9531326, -0.2945702, 0.7879696, -1.517101, -0.9344181, -0.5049928, -0.005040941, -0.4637912,
               0.8223695, -1.079849, 0.8871287, -0.9732434, -0.5548235, 1.879138, -1.452064, -0.1975368, 1.55047,
               0.5941782, -0.52897, 1.368219, 0.6782904, 1.202505, -0.9256122, -0.9718158, -0.9570228, -0.5563112,
               -1.19049, -1.167985, 2.606804, -2.261825, 0.01340385, 0.2526799, -1.125458, -1.575991, -0.363153,
               0.3270262, 1.485984, -1.769565, 1.541829, 0.7293826, 0.1743717, -0.4759418, 1.523451, -2.487134,
               -1.824067, -0.626367, 0.7448186, -1.425648, 0.3524166, -0.9903384, 3.339342, 0.4563958, -0.2876643,
               1.521635, 0.9508078, -0.1398541, 0.3867955, -0.7550205, 0.6568405, 0.09419366, -1.583935, 1.306094,
               -0.3501927, 0.1794427, -0.3768163, 0.9683866, -0.2442541, -1.696921, -1.8056, -0.6803037, -1.842043,
               0.3069353, 0.9070363, -0.486526]

    # spk_sig =[-0.435445, 0.877224, 1.072917, 0.127324, -0.605085, 0.930205, 0.44148, -1.20399, 0.069384, 0.538427, 1.226569, 0.852291, -0.806415, -1.157439, 0.313101, 1.332273, -1.628154, 0.402829, 0.472996, -1.479501, -0.065581, 1.127467, 0.897095, -1.544573, -0.96861, 0.888643, -2.189499, -0.155159, 1.974215, 0.277226, 0.058169, -1.234166, -1.627201, -0.429505, -1.101772, 0.789727, 0.45571, -0.547229, 0.424477, -0.919078, -0.396511, 1.35064, -0.02892, -0.442538, -1.60219, 0.615162, 0.052128, -0.432882, 1.94985, -0.704909, 0.804217, 0.472941, 0.333696, 0.47405, -0.214551, -1.895343, 1.511685, -1.284075, 0.623826, 0.034828, -0.065535, 1.604209, -0.923321, 0.502624, -0.288166, 0.536349, -0.631745, 0.970297, 0.403614, 0.131859, 0.978622, -0.5083, -0.104544, 1.629872, 1.730207, 1.010488, -0.866015, -0.711263, 2.359106, 1.151348, -0.426434, -0.80968, -1.302515, -0.444948, 0.074877, 1.352473, -1.007743, 0.318039, -1.532761, 0.145248, 3.59333, -0.467264, -0.667231, -0.890853, -0.197016, 1.546726, 0.890309, -0.7503, 0.773801, 0.84949, 0.391266, -0.79776, 0.895459, -0.816466, 0.110284, -1.030472, -0.144815, 1.087008, -1.448755, 0.776005, -0.270475, 1.223657, 1.09254, -1.237237, 0.065166, 1.487602, -1.409871, -0.539695, -0.758403, 0.31941, -0.701649, -0.210352, 0.613223, 0.575418, -0.299141, 1.247415, 0.375623, -1.001396]
    def cosine_dist(x, y):
        nx = np.array(x)
        ny = np.array(y)
        return 1 - np.dot(nx, ny) / np.linalg.norm(nx) / np.linalg.norm(ny)

    f = open('output_dia.txt', 'w')

    read_block_size = wf.getnframes()

    rec = KaldiRecognizer(model, wf.getframerate() * wf.getnchannels(), spk_model)
    # rec = KaldiRecognizer(model, wf.getframerate() * wf.getnchannels())
    # rec.SetSpkModel(spk_model)

    # res={};
    wf.rewind()
    # while True:
    for i in range(1080):
        data = wf.readframes(4000)
        datalen = len(data)
        if datalen == 0:
            res = json.loads(rec.FinalResult())
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            # f.write('- ')
            for index in res['text']:
                f.write(index)
            f.write('\n')
            print("Text:", res['text'])
            if 'spk' in res:
                print("X-vector:", res['spk'])
                print("Speaker distance:", cosine_dist(spk_sig, res['spk']), end=' ')
                print("based on frames:", res['spk_frames'])
        if datalen == 0:
            break
    # Note that second distance is not very reliable because utterance is too short. Utterances longer than 4 seconds give better xvector
    f.close()
    #st.write("Diarization is ready")


def dia_text_v2(wf, model, spk_model):
    # rec = KaldiRecognizer(model, wf.getframerate(), spk_model)
    wf.rewind()
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetSpkModel(spk_model)
    f = open('output_dia.txt', 'w')
    # We compare speakers with cosine distance. We can keep one or several fingerprints for the speaker in a database
    # to distingusih among users.
    spk_sig = [-1.110417, 0.09703002, 1.35658, 0.7798632, -0.305457, -0.339204, 0.6186931, -0.4521213, 0.3982236,
               -0.004530723, 0.7651616, 0.6500852, -0.6664245, 0.1361499, 0.1358056, -0.2887807, -0.1280468, -0.8208137,
               -1.620276, -0.4628615, 0.7870904, -0.105754, 0.9739769, -0.3258137, -0.7322628, -0.6212429, -0.5531687,
               -0.7796484, 0.7035915, 1.056094, -0.4941756, -0.6521456, -0.2238328, -0.003737517, 0.2165709, 1.200186,
               -0.7737719, 0.492015, 1.16058, 0.6135428, -0.7183084, 0.3153541, 0.3458071, -1.418189, -0.9624157,
               0.4168292, -1.627305, 0.2742135, -0.6166027, 0.1962581, -0.6406527, 0.4372789, -0.4296024, 0.4898657,
               -0.9531326, -0.2945702, 0.7879696, -1.517101, -0.9344181, -0.5049928, -0.005040941, -0.4637912,
               0.8223695, -1.079849, 0.8871287, -0.9732434, -0.5548235, 1.879138, -1.452064, -0.1975368, 1.55047,
               0.5941782, -0.52897, 1.368219, 0.6782904, 1.202505, -0.9256122, -0.9718158, -0.9570228, -0.5563112,
               -1.19049, -1.167985, 2.606804, -2.261825, 0.01340385, 0.2526799, -1.125458, -1.575991, -0.363153,
               0.3270262, 1.485984, -1.769565, 1.541829, 0.7293826, 0.1743717, -0.4759418, 1.523451, -2.487134,
               -1.824067, -0.626367, 0.7448186, -1.425648, 0.3524166, -0.9903384, 3.339342, 0.4563958, -0.2876643,
               1.521635, 0.9508078, -0.1398541, 0.3867955, -0.7550205, 0.6568405, 0.09419366, -1.583935, 1.306094,
               -0.3501927, 0.1794427, -0.3768163, 0.9683866, -0.2442541, -1.696921, -1.8056, -0.6803037, -1.842043,
               0.3069353, 0.9070363, -0.486526]

    def cosine_dist(x, y):
        nx = np.array(x)
        ny = np.array(y)
        return 1 - np.dot(nx, ny) / np.linalg.norm(nx) / np.linalg.norm(ny)

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            for index in res['text']:
                f.write(index)
            f.write('\n')
            print("Text:", res['text'])
            if 'spk' in res:
                print("X-vector:", res['spk'])
                print("Speaker distance:", cosine_dist(spk_sig, res['spk']), "based on", res['spk_frames'], "frames")
    f.close()
    #st.write("Diarization is ready")


model_path = "vosk-model-ru-0.22"
spk_model_path = "vosk-model-spk-0.4"
punct_model_path = "vosk-recasepunc-ru-0.22"

st.title("РАСПОЗНАВАНИЕ РЕЧИ")

sleep_duration = 1
percent_complete = 0
progress_bar = st.progress(percent_complete)
aq = st.write("Loading models...")

mod = Model(model_path)

sleep_duration = 0.01
for percent in range(percent_complete, 51):
    time.sleep(sleep_duration)
    progress_bar.progress(percent)

spk_mod = SpkModel(spk_model_path)

for percent in range(percent_complete, 101):
    time.sleep(sleep_duration)
    progress_bar.progress(percent)

del aq

fileObject = st.sidebar.file_uploader('Выберите vaw файл:')
if fileObject is not None:
    wavefile = wave.open(fileObject, "rb")
    if wavefile.getnchannels() != 1:
        print("Audio file must be mono.")
        exit(1)
    if wavefile.getsampwidth() != 2:
        print("Audio file must be WAV format PCM. sampwidth=", wavefile.getsampwidth())
        exit(1)

    if wavefile.getcomptype() != "NONE":
        print("Audio file must be WAV format PCM. comptype=", wavefile.getcomptype())
        exit(1)
    progress_bar = st.progress(percent_complete)
    aq = st.write("Processing...")
    base_text(wavefile, mod)
    for percent in range(percent_complete, 33):
        time.sleep(sleep_duration)
        progress_bar.progress(percent)
    dia_text_v2(wavefile, mod, spk_mod)
    for percent in range(percent_complete, 66):
        time.sleep(sleep_duration)
        progress_bar.progress(percent)

    punct_text()
    for percent in range(percent_complete, 101):
        time.sleep(sleep_duration)
        progress_bar.progress(percent)
    st.subheader("РАСПОЗНАННЫЙ ТЕКСТ:")
    do_itog()
    # dia_text_rosa(wavefile)
