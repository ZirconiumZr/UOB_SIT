@echo off
echo model download start...
echo ###################################################################################
wget https://alphacephei.com/kaldi/models/vosk-model-en-us-0.22.zip --no-check-certificate
echo ###################################################################################
echo model download success!
echo ###################################################################################
echo unzip model...
tar -xf "../vosk-model-en-us-0.22.zip"
echo ###################################################################################
echo move model to folder...
MOVE vosk-model-en-us-0.22 model
move model "../pretrained-model/stt/model"
echo success!
pause