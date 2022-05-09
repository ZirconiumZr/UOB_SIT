# Import necessary library
# For managing audio file
import librosa

#Importing Pytorch
import torch

#Importing Wav2Vec
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Loading model
processor = Wav2Vec2Processor.from_pretrained("./your_path/facebook_wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("./your_path/facebook_wav2vec2-base-960h")

def transcription(audio_path):
    audio, rate = librosa.load(audio_path, sr = 16000)
    # audio file is decoded on the fly
    inputs = processor(audio, sampling_rate=rate, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)

    # transcribe speech
    transcription = processor.batch_decode(predicted_ids)
    transcription[0].lower()
    return transcription[0].lower()

print(transcription('/your_audio_path/0.wav'))

# if you have amount of audio files needed to transcribe in one folder:
import os

def wavFilesPath(path):
    '''
    path: directory folder address

    Return value: list, full path of wav file
    '''
    filePaths = [] # All file names in the storage directory, including paths
    for root,dirs,files in os.walk(path):
        for file in files:
            filePaths.append(os.path.join(root,file))
    return filePaths

# file folder
filepath = r'/content/singlish-test/singlish-test'
WavFilesPath=wavFilesPath(filepath)
print(WavFilesPath)

stt_list=[]
for i in WavFilesPath:
    stt_list.append(transcription(i))

print(stt_list)
c={"WavFilesPath" : WavFilesPath,
   "stt" : stt_list}
from pandas import DataFrame
data_1=DataFrame(c)#Convert dictionary to dataframe
from os.path import basename
for i in range(len(data_1)):
    data_1['new_path'][i] = int(os.path.splitext(basename(data_1['WavFilesPath'][i]))[0])

print(data_1['new_path'])
data_1.sort_values(by="new_path" , inplace=True, ascending=True)#Reorder the entire dataframe by a column


# if you have correct text and want to do the WER
import pandas as pd
transcript = pd.read_csv('/content/singlish.csv') #correct text
print(transcript)
stt=list(data_1['stt'])# get the predicted text
transcript['stt']=stt # add the predicted text to the correct dataframe, make sure the order is matched
example=transcript
example.to_csv("example.csv",index=False,header=True)

# get the accuracy of each audio and the average accuracy of all audios in the folder
import pandas as pd
import Levenshtein

"""# Load data with stt results and real transcript columns."""

data=pd.read_csv("example.csv")

"""# WER function"""

def get_edit(transcript,stt):
    result_dict={'S':0,'D':0,'I':0,'N':0,'ser':0}
    transcript=str(transcript).replace('NaN','')
    stt=str(stt).replace('NaN','')
    result=Levenshtein.editops(transcript,stt)
    for row in result:
        if 'replace' in row:
            result_dict['S']+=1
        if 'delete' in row:
            result_dict['D']+=1
        if 'insert' in row:
            result_dict['I']+=1
    result_dict['N']=len(transcript)
    return result_dict

data['I']=data.apply(lambda x:get_edit(x['transcript'],x['stt'])['I'],axis=1)
data['D']=data.apply(lambda x:get_edit(x['transcript'],x['stt'])['D'],axis=1)
data['S']=data.apply(lambda x:get_edit(x['transcript'],x['stt'])['S'],axis=1)
data['N']=data.apply(lambda x:get_edit(x['transcript'],x['stt'])['N'],axis=1)

data['Accuracy']=data.apply(lambda x:(x['N']-x['I']-x['D']-x['S'])/x['N'],axis=1)

data.to_csv("example_results.csv",index=False,header=True)

print('Finish! Please check the output csv.')
print('number of audios:',len(data))
print('average accuracy:',sum(data['Accuracy'])/len(data))