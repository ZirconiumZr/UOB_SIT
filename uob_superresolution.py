import malaya_speech
import numpy as np
from malaya_speech import Pipeline

import uob_extractmodel


def load_superresolution_model(model = 'srgan-128', quantized = False): 
    ## List available deep model:
    # malaya_speech.speech_enhancement.available_deep_enhance()
    #? model=['unet','resnet-unet']
    
    ## Load deep model
    model = malaya_speech.super_resolution.deep_model(model = model, quantized = quantized)
    return model

def load_superresolution_model_local(model = 'srgan-256', quantized = False): 
    model = uob_extractmodel.get_model_superresolution(model = model, quantized = quantized)
    return model

def get_sr_output(y, sr, model):
    print(y)
    output = model(y)
    return output

