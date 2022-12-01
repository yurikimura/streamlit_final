###########################################
# Description: 
# - This program allows the user to input a file with a desired start and stop time. 
# - It returns text corresponding to the inputted audio. 
###########################################

# Imports
import speech_recognition as sr 
import os 
from pydub import AudioSegment
from pydub.silence import split_on_silence

# create a speech recognition object
r = sr.Recognizer()

# crop audio file
def crop_audio(audio, start : int = 0, stop : int = -1):
    '''
    Allows user to select a part of an audio file by specifying the seconds. 
    Returns uncropped version of audio by default. 

    audio: opened audio file in wav format
    start: The starting time in seconds
    stop: The stopping time in seconds
    '''

    start = 1000 * start
    stop = 1000 * stop
    return audio[start:stop]

# split cropped audio into sentences
def find_sentences(audio): 
    '''
    Searches for long pauses to indicate sentences within speech and returns 
    a list of audio chunks.
    '''
    # split audio sound where silence is half a second or more and get chunks
    chunks = split_on_silence(audio,
        # experiment with this value for target audio file
        min_silence_len = 500,
        # adjust this per requirement
        silence_thresh = audio.dBFS-14,
        # keep the silence for half a second
        keep_silence = 500,
    )
    return chunks
    
# saves audio chunks and converts them to text
def to_text(cropped_chunks : list, 
               foldername = 'cropped_audio_files', 
               name = 'audio_chunk'): 
    '''
    Saves cropped audio file chunks in folder and then converts them to text.
    
    cropped_chunks: must be in list format
    '''

    # create folder
    if not os.path.isdir(foldername):
        os.mkdir(foldername)
    
    whole_text = ''
    
    # process each chunk 
    for i, audio_chunk in enumerate(cropped_chunks):
        
        # export audio chunk and save it in a folder
        chunk_filename = os.path.join(foldername, f"{name}{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")

        # recognize the chunk
        with sr.AudioFile(chunk_filename) as source:
            audio_listened = r.record(source)
            
            # try converting it to text
            try:
                text = r.recognize_google(audio_listened)
            # if audio cannot be translated, print an empty string
            except sr.UnknownValueError as e:
                print('')
            # convert audio to text
            else:
                text = f"{text.capitalize()}. "
                whole_text += text
    return whole_text
        
# convert audio to text
def audio_to_text(path, start, stop): 
    '''
    Converts a wav audio file to text. 
    
    path: file path
    start: start time in seconds
    stop: stop time in seconds
    '''
    
    # open audio file
    audio = AudioSegment.from_file(path, format = "wav")
    
    # crop audio to desired length
    cropped_audio = crop_audio(audio, start, stop)
    
    # separate cropped audio into sentences
    chunks = find_sentences(cropped_audio)
    
    # save audio chunks and converts them to text
    text = to_text(chunks)
    
    return text