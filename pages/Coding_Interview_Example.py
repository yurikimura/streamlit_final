############################################################
# This program allows you to run our app through Streamlit.
############################################################

# imports
import streamlit as st
import numpy as np
import pandas as pd
import os
import warnings
import base64

import wave
import struct
import math
from datetime import timedelta
import requests

from speech_to_text import *

# ignores any warnings that pop up when running file
warnings.simplefilter('ignore')

sample_rate = 44100
threshold = 20
sample_length = 7680
batch_size = 16
epoch = 50
time = 3

# names of speakers
target_label = {0: "Speaker 1",
                1: "Speaker 2",
                2: "Speaker 3",
                3: "Speaker 4",
                4: "Speaker 5",
                5: 'Speaker 6',
                6: 'Speaker 7'}

# pictures of speakers
target_images = {0: 'speaker_images/black_sw.png',
                 1: 'speaker_images/blue_sw.png',
                 2: 'speaker_images/green_sw.png',
                 3: 'speaker_images/pink_sw.png',
                 4: 'speaker_images/red_sw.png',
                 5: 'speaker_images/yellow_sw.png',
                 6: 'speaker_images/rainbow_sw.jpg'}

# title
st.title('Speaker Annotation: Interview')
st.write('This model was trained on data from a coding interview and tells you who is speaking when.')
col1, col2, col3 = st.columns([1,6,1])
with col1:
    st.write("")
with col2:
    st.image('speaker_images/interview.png')
with col3:
    st.write("")

interview_url = "https://interview33-gxqshhfzea-an.a.run.app/getIntervewerPrediction"
time = 1


# create a sidebar for uploading wav files
uploaded_audio = st.sidebar.file_uploader("Upload your audio file below ⬇️", type=['wav'])

def target_cropper(wr, time):
    #waveファイルが持つ性質を取得
    ch = wr.getnchannels()#モノラルorステレオ
    width = wr.getsampwidth()
    fr = wr.getframerate()#サンプリング周波数
    fn = wr.getnframes()#フレームの総数
    total_time = 1.0 * fn / fr
    integer = math.floor(total_time)
    t = int(time)
    frames = int(ch * fr * t)
    num_cut = int(integer//t)

    # waveの実データを取得し数値化
    data = wr.readframes(wr.getnframes())
    wr.close()
    X = np.frombuffer(data, dtype='int16')

    for i in range(num_cut):
        #出力データを生成
        outf = str(i) + '.wav'
        start_cut = i*frames
        end_cut = i*frames + frames
        Y = X[start_cut:end_cut]
        outd = struct.pack("h" * len(Y), *Y)

        # 書き出し
        ww = wave.open(outf, 'w')
        ww.setnchannels(ch)
        ww.setsampwidth(width)
        ww.setframerate(fr)
        ww.writeframes(outd)
        ww.close()

    return num_cut

def get_predict(uploaded_file, time, interview_url):
    wr = wave.open(uploaded_file, 'r')
    num_cut = target_cropper(wr,time)
    for i in range(num_cut):
        with open(f'{i}.wav', "rb") as wavfile:
            input_wav = wavfile.read()
            wr_b64 = base64.b64encode(input_wav)
            payload = {"wrfile": wr_b64, "dur":time, "num":i}
            r = requests.post(interview_url, data = payload)
        os.remove(f'{i}.wav')
    return r.json()

def predict_timestamp_and_remove(num_cut):
    # starting time is 0 seconds
    start_sec = 0
    time_from = str(timedelta(seconds = 0))
    # no last speaker at time 0
    last_speaker = None
    # df placeholder
    df_placeholder = st.empty()
    # image placeholder
    image_placeholder = st.sidebar.empty()
    # df placeholder
    df_placeholder = st.empty()
    # image placeholder
    image_placeholder = st.sidebar.empty()
    # creates empty lists for data frame
    speaker_column = []
    start_time_column = []
    end_time_column = []
    probability_column = []
    text_column = []

    # displays timestamps and speaker info in real time
    for i in range(num_cut):
        # calculate seconds
        sec = (i + 1) * time
        # calculate ending time for audio cropping
        end_sec = sec
        time_to = str(timedelta(seconds = sec))

        path = f'{i}.wav'
        result = get_predict(uploaded_audio, time, interview_url)
        predicted_speaker = result['predlist']

        ### start ###
        # If the same person is still speaking, pass
        if predicted_speaker == last_speaker:
            pass

        # If this is the first speaker, make them the 'last' speaker
        elif last_speaker == None:
            last_speaker = predicted_speaker

        # If this is the last audio clip in the folder
        elif i + 1 == num_cut:

            # calculates time stamp
            timestamp = result['timestamp']

            # calculates speaker probability and label
            speaker_probability = round(max(result['predlist'][0]) * 100)
            speaker = target_label[predicted_speaker]

            # calculates speech to text
            text = audio_to_text(predfile, start_sec, end_sec)

            # update values
            start_time_column.append(time_from)
            end_time_column.append(time_to)
            speaker_column.append(speaker)
            probability_column.append(speaker_probability)
            text_column.append(text)

            # displays image in sidebar
            with image_placeholder.container():
                st.image(target_images[predicted_speaker], width = 300)
                st.subheader(f'{speaker}:   {timestamp}')

            # creates df
            df = pd.DataFrame({'speaker': speaker_column,
                                   'start_time': start_time_column,
                                   'end_time': end_time_column,
                                   'text': text_column,
                                   'probability': probability_column})

            # overwrites container
            with df_placeholder.container():
                # Displays df
                st.dataframe(df.tail(5))


        # If the speaker changes
        else:
            # This is for the first speaker only
            if start_sec == 0:
                # calculate time stamp
                timestamp = f'{time_from} --> {time_to}'

                # calculate probability of speaker
                speaker_probability = round(max(result['predlist'][0]) * 100)
                speaker = target_label[predicted_speaker]

                # calculates speech to text
                text = audio_to_text(predfile, start_sec, end_sec)

                # update values
                start_time_column.append(time_from)
                end_time_column.append(time_to)
                speaker_column.append(speaker)
                probability_column.append(speaker_probability)
                text_column.append(text)

                # displays image in sidebar
                with image_placeholder.container():
                    st.image(target_images[predicted_speaker], width = 300)
                    st.subheader(f'🗣️ {speaker}: {timestamp}')

                df = pd.DataFrame({'speaker': speaker_column,
                                   'start_time': start_time_column,
                                   'end_time': end_time_column,
                                   'text': text_column,
                                   'probability': probability_column})

                # overwrites current container
                with df_placeholder.container():
                    # Displays df
                    st.dataframe(df.tail(5))

                # updates information
                last_speaker = predicted_speaker
                time_from = time_to
                start_sec = end_sec

            else:

                # calculates time stamp
                timestamp = f'{time_from} --> {time_to}'

                # calculate probability of speaker
                speaker_probability = round(max(result['predlist'][0]) * 100)
                speaker = target_label[predicted_speaker]

                # calculate speech to text
                text = audio_to_text(predfile, start_sec, end_sec)

                # update values
                start_time_column.append(time_from)
                end_time_column.append(time_to)
                speaker_column.append(speaker)
                probability_column.append(speaker_probability)
                text_column.append(text)

                # displays image in sidebar
                with image_placeholder.container():
                    st.image(target_images[predicted_speaker], width = 300)
                    st.subheader(f'🗣️ {speaker}: {timestamp}')

                # creates data frame
                df = pd.DataFrame({'speaker': speaker_column,
                                   'start_time': start_time_column,
                                   'end_time': end_time_column,
                                   'text': text_column,
                                   'probability': probability_column})

                # overwrites current container
                with df_placeholder.container():
                    # Displays df
                    st.dataframe(df.tail(5))

                # updates information
                last_speaker = predicted_speaker
                time_from = time_to
                start_sec = end_sec

        # remove audio clips
        os.remove(path)

    # convert df to csv
    @st.cache
    def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')
    csv = convert_df(df)

    # display download button
    st.download_button(
        data = csv,
        label = "Download data as a CSV",
        file_name = 'dataframe.csv',
        mime = 'text/csv')

try:
    # Top of page
    st.write('Hit the button below to see how it works: ')

    # If the user clicks this button, the preloaded file will run
    if st.button('run example', type = 'primary'):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        audio_dir = os.path.join(current_dir, 'example_audio')
        predfile = os.path.join(audio_dir, 'trimmed_interview.wav')

    # If the user doesn't click the button, then they upload an audio file to run
    else:
        predfile = uploaded_audio
    wr = wave.open(predfile, 'r')
    n = target_cropper(wr, 3)
    predict_timestamp_and_remove(n)

except:
    pass
