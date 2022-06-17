import re
import os
from text2digits import text2digits
from flair.data import Sentence
from flair.models import SequenceTagger
from analysis.uob_init import(
    pretrained_model_path
)

def personalInfoDetector(sttDf):
    t2d = text2digits.Text2Digits()
    sttDf['text'] = [t2d.convert(x) for x in sttDf['text']]
    # kyc detector
    kycList = [' ']
    kycWordsList = ['dob','date of birth','birth','nric', 'id', 'pass', 'passport']
    for i in range(1,len(sttDf)):
        if re.findall(r'\d+\b', sttDf['text'][i]) and (any(x in sttDf['text'][i].split() for x in kycWordsList) or any(x in sttDf['text'][i-1].split() for x in kycWordsList)):
            kycList.append('Yes')
        else:
            kycList.append(' ')
    sttDf['is_kyc'] = kycList
    
    # pii detector
    # load the trained model
    modelPath = os.path.join(pretrained_model_path, 'ner/ner-model.pt')
    model = SequenceTagger.load(modelPath)
    piiList=[ ]
    Label = [ ]
    list_of_sentences = sttDf['text'].values
    for i in range(len(list_of_sentences)):
        sentence = Sentence(list_of_sentences[i])
        model.predict(sentence)
        foo = sentence.to_dict(tag_type='ner')
        if len(foo['ner']) > 0:       
            Label.append(foo['ner'][0]['value'])
            # PII.append(foo['ner'][0]['text'])
            piiList.append('Yes')
        else:
            Label.append(' ')
            piiList.append(' ')
    sttDf['is_pii'] = piiList
    return sttDf
