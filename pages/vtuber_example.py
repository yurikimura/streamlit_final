# This file contains contents needed to run out app through streamlit

# imports
import streamlit as st
import numpy as np
import pandas as pd
import os
import random
import librosa
import warnings
from keras.models import load_model

import wave
import struct
import math
from datetime import timedelta

import io
from scipy.io import wavfile

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
target_label = {0: "Calliope", 
                1: "Ninomae", 
                2: "Watson", 
                3: "Gura", 
                4: "Kiara"}

# pictures of speakers
target_images = {0: 'vtuber_images/calliope.webp', 
                 1: 'vtuber_images/ninomae.webp', 
                 2: 'vtuber_images/watson.webp', 
                 3: 'vtuber_images/gura.webp', 
                 4: 'vtuber_images/kiara.webp'}

# loading model
reconstructed_model = load_model("vtuber_reco.h5")

# title
st.title('Speaker Annotation with 5 VTubers')
st.write('This model was trained on 5 VTubers and is now able to identify them.')
col1, col2, col3 = st.columns([1,6,1])
with col1:
    st.write("")
with col2:
    st.image('vtuber_images/everyone.jpeg')
with col3:
    st.write("")
    
# create a sidebar for uploading wav files
predfile = st.sidebar.file_uploader("Upload your audio file below ⬇️", type=['wav'])

# prepare audio for feeding into the model
def preprocess(audio,threshold,sample_length,sample_rate):
    audio, _ = librosa.effects.trim(audio, threshold)

    # すべての音声ファイルを指定した同じサイズに変換
    if threshold is not None:
        if len(audio) <= sample_length:
            # padding
            pad = sample_length - len(audio)
            audio = np.concatenate((audio, np.zeros(pad, dtype=np.float32)))
        else:
            # trimming
            start = random.randint(0, len(audio) - sample_length - 1)
            audio = audio[start:start + sample_length]
        stft = np.abs(librosa.stft(audio))
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40),axis=1)
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate),axis=1)
        mel = np.mean(librosa.feature.melspectrogram(audio, sr=sample_rate),axis=1)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate),axis=1)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate),axis=1)

        feature = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
        feature = np.expand_dims(feature, axis=1)

    return feature

# crop audio files into 3 second segments
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

    # play sound
    virtualfile = io.BytesIO()
    wavfile.write(virtualfile, rate = fr * 2, data = X)

    # display audio player
    st.sidebar.audio(virtualfile )

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
        
        x_batch = []  # feature
        path = f'{i}.wav'
        audio, _ = librosa.load(path, sr=sample_rate)
        mfccs = preprocess(audio,threshold,sample_length,sample_rate)
        x_batch.append(mfccs)
        x_batch = np.asarray(x_batch)
        
        # list of predictions for each speaker
        pred = reconstructed_model.predict(x_batch,verbose=0)
        
        # the number that corresponds to the predicted speaker
        predicted_speaker = pred.argmax()

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
            timestamp = f'{time_from} --> {time_to}'
            
            # calculates speaker probability and label
            speaker_probability = round(max(pred[0]) * 100)
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
                st.image(target_images[predicted_speaker], width = 200)
                st.subheader(f'{speaker}: {timestamp}')
            
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
                speaker_probability = round(max(pred[0]) * 100)
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
                    st.image(target_images[predicted_speaker], width = 200)
                    st.subheader(f'{speaker}: {timestamp}')
                
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
                speaker_probability = round(max(pred[0]) * 100)
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
                    st.image(target_images[predicted_speaker], width = 200)
                    st.subheader(f'{speaker}: {timestamp}')
                
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
    wr = wave.open(predfile, 'r')
    n = target_cropper(wr, 3)
    predict_timestamp_and_remove(n)
    
except:
    pass


