# Fine-tune STT model

## 1.deepspeech

### For test data:

#### Need one csv with 3 column:

wav_filename   ###absolute path

wav_filesize   ###audio size(kb)

transcript   ###text

#### Audio requirement:

channel: 1 (mono)

sample rate: 16000

### For training in colab:

Need to configure gpu and pip install tensorflow-gpu==1.15.4

## 2.vosk


