# Fine-tune STT model

## 1.deepspeech

SoX is used to adjust the audio format and will not be used in the process of training the model, so it is not necessary to install it at present.

### run STT

Not yet merged into main

run in cmd：

E.

python client.py --model deepspeech-0.9.3-models.pbmm --scorer deepspeech-0.9.3-models.scorer --audio bank.wav

### For test data:

#### Need one csv with 3 column: (example: testdata.csv)

wav_filename   ###absolute path

wav_filesize   ###audio size(kb)

transcript   ###text

#### Audio requirement: (example: sample.zip)

channel: 1 (mono)

sample rate: 16000

### For training in colab:

Need to configure gpu and pip install tensorflow-gpu==1.15.4

## 2.vosk


