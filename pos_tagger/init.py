import logging
import os

pretrained_model_path = './pretrained-model/'

AUDIO_NAME = 'MOCK CALL_Bank Customer Service_139-284.wav' #'Bdb001_interaction_first60s.wav'
AUDIO_PATH = './wav/'
AUDIO_FILE = os.path.join(AUDIO_PATH,AUDIO_NAME)
SAMPLE_RATE = 44100
STT_SAMPLERATE = 16000
FLG_REDUCE_NOISE:bool = False
FLG_SPEECH_ENHANCE:bool = False
FLG_SPEECH_ENHANCE_NEW:bool = False
FLG_SUPER_RES:bool = False

label_stop_words = 'nan'
label_checklist = 'checklist.txt'

dbHost = 'localhost'
dbName = 'UOBtests'
dbUser = 'uobadmin'
dbPwd = 'p@ssw0rd'

sd_global_starttime = 0.0
sttModel = 'malaya-speech' # ? ['malaya-speech', 'vosk']
sdModel = 'resemblyzer' # ? [pyannoteaudio, malaya, resemblyzer]

flg_slice_orig = True
stt_replace_template = 'stt_replace_template.csv' # template file path(path+filename(.csv))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                              Initialization  Processes                              #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# * Audo Convert FLG_SPEECH_ENHANCE for better effect
if FLG_REDUCE_NOISE == False and FLG_SPEECH_ENHANCE==True:
    logging.warning('Auto turn off Speech Enhancement for better effect since Noise Reduction is off.')
    FLG_SPEECH_ENHANCE = False