# -*- coding: utf-8 -*-
"""WER_on _prem.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1O2B1v6swC02RuGKzhJ5Id6yKfTNTKQbp
"""

import pandas as pd
import Levenshtein
import re

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