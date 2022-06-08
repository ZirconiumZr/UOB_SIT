!pip install datasets==1.18.3
!pip install transformers==4.11.3
!pip install torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
!pip install jiwer

# check the information for avaliable GPU
gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
    print('Not connected to a GPU')
else:
    print(gpu_info)

# get the fine-tune training and testing data
from datasets import load_dataset, load_metric, Audio
# "load_dataset" is a function using script to loading data, put the path to your data_loading script(should be a python script with .py format)
uob_voice_train = load_dataset("/content/drive/MyDrive/MrBrown/uob_loadingdata.py", split="train", cache_dir="path/to/the/place/you/want/to/storing/the/loaded/dataset")# path/to/your/loading_data/transcript
uob_voice_test = load_dataset("/content/drive/MyDrive/MrBrown/uob_loadingdata.py", split="test", cache_dir="path/to/the/place/you/want/to/storing/the/loaded/dataset")# path/to/your/loading_data/transcript

# check whether the Dataset format data have been prepaired, and the information of the training and testing datasets
print(uob_voice_train)
print(uob_voice_test)

# View specific information for each piece of data
print(uob_voice_test[0])
print(uob_voice_test[0]['id'])
print(uob_voice_train[0]['audio'])
print(uob_voice_train[0]['text'])#the corresponding correct text for each slice of audio
print(uob_voice_train[0]['file'])#path to audio file
print(type(uob_voice_train))# make sure the type of training data is "Dataset"

from datasets import ClassLabel
import random
import pandas as pd
from IPython.display import display, HTML

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    display(HTML(df.to_html()))

# have a quick look of formated training data
print(show_random_elements(uob_voice_train.remove_columns(["audio"]), num_examples=5))

# remove these characters inside the text
import re
chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\'\[\]\(\)]'

def remove_special_characters(batch):
    batch["text"] = re.sub(chars_to_remove_regex, '', batch["text"]).lower()
    return batch

# use the "map" function do the remove in batch
uob_voice_train = uob_voice_train.map(remove_special_characters)
uob_voice_test = uob_voice_test.map(remove_special_characters)

print(show_random_elements(uob_voice_train.remove_columns(["audio"]), num_examples=5))

# extract all distinct letters of the training and test data and build our vocabulary from this set of letters.
def extract_all_chars(batch):
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

# mapping function -- concatenate all transcriptions into one long transcription and then transforms the string into a set of chars
# use the "map" function extract characters inside the data in batch (batched=True->mapping function has access to all transcriptions at once)
vocab_train = uob_voice_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=uob_voice_train.column_names)
vocab_test = uob_voice_test.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=uob_voice_test.column_na

vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))

# make vocab_dict base on the fine-tune training set
# note: the vocab_dict made here should be same as the vocab_dict of the pretrained model(the model we what to base on it do the fine-tune),
# so make sure your fine-tune training set's text should just contains a-z,0-9 ,no more, no less.
vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
print(vocab_dict)

vocab_dict[""] = vocab_dict[" "]
del vocab_dict[" "]

vocab_dict["0"] = len(vocab_dict)
vocab_dict["1"] = len(vocab_dict)
vocab_dict["2"] = len(vocab_dict)
vocab_dict["3"] = len(vocab_dict)
vocab_dict["4"] = len(vocab_dict)
vocab_dict["5"] = len(vocab_dict)
vocab_dict["6"] = len(vocab_dict)
vocab_dict["7"] = len(vocab_dict)
vocab_dict["8"] = len(vocab_dict)
vocab_dict["9"] = len(vocab_dict)
vocab_dict["|"] = len(vocab_dict)
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
len(vocab_dict)

# have a check, should be same as the malaya-speech's vocab_list
vocab_dict

# save the vocabulary as a json file.
import json
with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

# use the json file to instantiate an object of the Wav2Vec2CTCTokenizer clas
from transformers import Wav2Vec2CTCTokenizer

tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

# feature_size (int, defaults to 1) — The feature dimension of the extracted features.
# sampling_rate (int, defaults to 16000) — The sampling rate at which the audio files should be digitalized expressed in Hertz per second (Hz).
# padding_value (float, defaults to 0.0) — The value that is used to fill the padding values.
# do_normalize (bool, optional, defaults to True) — Whether or not to zero-mean unit-variance normalize the input. Normalizing can help to significantly improve the performance for some models, e.g., wav2vec2-lv60.
# return_attention_mask (bool, optional, defaults to False) — Whether or not call() should return attention_mask.
# Wav2Vec2 models that have set config.feat_extract_norm == "group", such as wav2vec2-base, have not been trained using attention_mask. For such models, input_values should simply be padded with 0 and no attention_mask should be passed.
from transformers import Wav2Vec2FeatureExtractor
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)

# Constructs a Wav2Vec2 processor which wraps a Wav2Vec2 feature extractor and a Wav2Vec2 CTC tokenizer into a single processor.
# plus: CTC(Connectionist Temporal Classification) can ensure that the speed of the speaker does not affect the recognition of STT. For the specific principle, please refer to this article:
# https://distill.pub/2017/ctc/
from transformers import Wav2Vec2Processor
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# have a check before re-sample to 16000
print(uob_voice_train[0]["file"])
print(uob_voice_train[0]["audio"])

# re-sample all the audios to 16000 Hz
uob_voice_train = uob_voice_train.cast_column("audio", Audio(sampling_rate=16_000))
uob_voice_test = uob_voice_test.cast_column("audio", Audio(sampling_rate=16_000))
# have a check after re-sampling to 16000
# the "sampling_rate" should be 16000 now
print(uob_voice_train[0]["audio"])

# import IPython.display as ipd
import numpy as np
import random

rand_int = random.randint(0, len(uob_voice_train)-1)

print(uob_voice_train[rand_int]["text"])
# if use the jupyter notebook, can listen to the slice of audio as well.
# ipd.Audio(data=uob_voice_train[rand_int]["audio"]["array"], autoplay=True, rate=16000)

rand_int = random.randint(0, len(uob_voice_train)-1)

print("Target text:", uob_voice_train[rand_int]["text"])
print("Input array shape:", uob_voice_train[rand_int]["audio"]["array"].shape)
print("Sampling rate:", uob_voice_train[rand_int]["audio"]["sampling_rate"])

def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched"
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch

uob_voice_train = uob_voice_train.map(prepare_dataset, remove_columns=uob_voice_train.column_names)
uob_voice_test = uob_voice_test.map(prepare_dataset, remove_columns=uob_voice_test.column_names)

print(uob_voice_train)

# if the memory of the computer is not large, can set this threshold to drop some of the long auios inside the training set.
max_input_length_in_sec = 5.0
uob_voice_train = uob_voice_train.filter(lambda x: x < max_input_length_in_sec * processor.feature_extractor.sampling_rate, input_columns=["input_length"])

import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
                labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
                batch["labels"] = labels
                return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

wer_metric = load_metric("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

from transformers import Wav2Vec2ForCTC

model = Wav2Vec2ForCTC.from_pretrained(
    "malay-huggingface/wav2vec2-xls-r-300m-mixed", #here put the path to the pre-trained model package
    attention_dropout=0.0,
    hidden_dropout=0.0,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.0,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
)

# The first component of Wav2Vec2 consists of a stack of CNN layers that are used to extract acoustically meaningful - but contextually independent - features from the raw speech signal.
# This part of the model has already been sufficiently trained during pretrainind and as stated in the paper does not need to be fine-tuned anymore.
# Thus, we can set the requires_grad to False for all parameters of the feature extraction part.
model.freeze_feature_extractor()

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='/content/drive/MyDrive/MrBrown/malaya-finetuned-easting', # path to store the fine-tuned model
    group_by_length=True,# group_by_length makes training more efficient by grouping training samples of similar input length into one batch. This can significantly speed up training time by heavily reducing the overall number of useless padding tokens that are passed through the model
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=30, #
    gradient_checkpointing=True,
    fp16=True,
    save_steps=400,
    eval_steps=400,
    logging_steps=400,
    learning_rate=0.001,# usually the smaller the learning_rate, the longer time will cost but the training will be better
    warmup_steps=500,
    save_total_limit=2,
    push_to_hub=False, #If do not want to upload the model checkpoints to the hub, simply set push_to_hub=False.
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=uob_voice_train,
    eval_dataset=uob_voice_test,
    tokenizer=processor.feature_extractor,
)

# import torch
# torch.cuda.empty_cache()

trainer.train()



#################################################
# Here is the way if you want to reuse the fine-tuned model(model after fine-tune) to do the speech-to-text on any audio.
#################################################

# Import necessary library
# For managing audio file

import librosa

# Importing Pytorch
import torch

# Importing Wav2Vec
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Loading model
processor = Wav2Vec2Processor.from_pretrained("RuiqianLi/wav2vec2-large-xls-r-300m-singlish-colab")#here put the path to the fine-tuned model
model = Wav2Vec2ForCTC.from_pretrained("RuiqianLi/wav2vec2-large-xls-r-300m-singlish-colab")#here put the path to the fine-tuned model

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

print(transcription('/content/drive/MyDrive/MrBrown/Easting/chunk46.wav'))
print(transcription('/content/drive/MyDrive/MrBrown/Easting/chunk47.wav'))
print(transcription('/content/drive/MyDrive/MrBrown/Easting/chunk48.wav'))



