# coding=utf-8
# Copyright 2021 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Singlish automatic speech recognition dataset."""


import csv
import os

import datasets
from datasets.tasks import AutomaticSpeechRecognition


_CITATION = """\
@misc{RuiqianLi,
  author       = {Ruiqian LI},
  title        = {The Singlish Speech Dataset},
  year         = 2022
}
"""

_DESCRIPTION = """\
This is a public domain speech dataset consisting of 3579 short audio clips of singlish
"""

_DL_URL = "https://docs.google.com/uc?export=download&id=1BtaSyRAOfNU9fZmEFQmy3hpNPaVH5ziS"

class UOBSinglish(datasets.GeneratorBasedBuilder):
    """Singlish dataset."""

    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="main", version=VERSION, description="The Singlish dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=16000),
                    "file": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    #"normalized_text": datasets.Value("string"),
                }
            ),
            supervised_keys=("file", "text"),
            #homepage=_URL,
            citation=_CITATION,
            #task_templates=[AutomaticSpeechRecognition(audio_column="audio", transcription_column="text")],
        )

    def _split_generators(self, dl_manager):
        root_path = dl_manager.extract("./drive/MyDrive/MrBrown/Easting.zip")
        #root_path = dl_manager.download_and_extract(_DL_URL)
        root_path = os.path.join(root_path, "Easting")
        train_wav_path = os.path.join(root_path, "train")
        train_csv_path = os.path.join(root_path, "train.csv")
        test_wav_path = os.path.join(root_path, "test")
        test_csv_path = os.path.join(root_path, "test.csv")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"wav_path": train_wav_path, "csv_path": train_csv_path}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"wav_path": test_wav_path, "csv_path": test_csv_path}
            ),
        ]

    def _generate_examples(self, wav_path, csv_path):
        """Generate examples from an Singlish archive_path."""

        with open(csv_path, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                uid, text = row
                filename = f"{uid}.wav"
                example = {
                    "id": uid,
                    "file": os.path.join(wav_path, filename),
                    "audio": os.path.join(wav_path, filename),
                    "text": text,
                    #"normalized_text": norm_text,
                }
                yield uid, example
