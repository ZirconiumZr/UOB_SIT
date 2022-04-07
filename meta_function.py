# 5.读取audio本身文件信息，存成JSON string

import wave
import os

def get_audio_params(audioname,audiopath):
    '''
    Note: only works on .wav file
    '''
    audiofile = os.path.join(audiopath, audioname)
    f=wave.open(audiofile)
    nframes = f.getnframes()
    rate = f.getframerate()
    duration = nframes / float(rate)#也就是音频时长 = 采样点数/采样率，单位为s
    bytes_size=os.path.getsize(os.path.join(audiopath, audioname))
    params = f.getparams()
    nchannels = f.getnchannels()
    samplerate = f.getframerate()
    sampwidth = f.getsampwidth()
    bit_type = f.getsampwidth() * 8#比特(Precision)，也称为位深
    bit_rate=samplerate * bit_type * nchannels#【比特率】(kbps)=【量化采样点】(kHz)*【位深】(bit/采样点)*声道数量
    f.close()

    print('params: ',params)
    # print('channels: ',channels)
    # print('sample rate: ', samplerate)
    # print('sample width: ', sampwidth)
    # print('nframes: ', nframes)

    return {'nchannels':nchannels,'samplerate':samplerate,'sampwidth':sampwidth,'nframes':nframes, 'duration':duration, 'bytes_size':bytes_size, 'bit_type':bit_type,'bit_rate':round(bit_rate)}

#test:
get_audio_params(audioname='TPO_C1.wav',audiopath = "/Users/ruiqianli/Desktop/UOB_SIT-main/test_data/audios/audios/")
#output:
# {'nchannels': 2,
#  'samplerate': 44100,
#  'sampwidth': 2,
#  'nframes': 7187328,
#  'duration': 162.97795918367348,
#  'bytes_size': 28749356,
#  'bit_type': 16,
#  'bit_rate': 1411200}