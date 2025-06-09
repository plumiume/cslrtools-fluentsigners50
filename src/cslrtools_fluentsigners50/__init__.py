import csv
from parse import parse, Result
import pickle
from itertools import chain
from typing import TypedDict, cast, Self
from pathlib import Path
from cslrtools.dataset.pytorch import Metadata, Dataset
import numpy as np
import torch

def raise_exception(exception: Exception):
    raise exception

_PathLike = Path | str

# (root)
# ├── KRSL_1723_17_08
# │   ├── 000
# │   │   ├── P0_S000_00.mp4
# │   │   ├── P0_S000_01.mp4
# │   │   ├── P0_S000_02.mp4
# |   :   :
# ├── <KRSL_1723_17_08_landmarks>/
# │   ├── 000
# │   │   ├── P0_S000_00.npy (or .csv)
# │   │   ├── P0_S000_01.npy
# │   │   ├── P0_S000_02.npy
# |   :   :
# ├── gloss_annotation.csv
# └── russian_translation.csv

class GlossAnnotation(TypedDict):
    ID: str
    Gloss: str

class RussianTranslation(TypedDict):
    ID: str
    Translation: str

class FluentSiners50Metadata(Metadata):
    person: int
    variation: int

class FluentSigners50Item(TypedDict):
    input: torch.Tensor
    label: list[str]
    metadata: FluentSiners50Metadata

class FluentSigners50:

    def __init__(self, root: _PathLike, landmarks: str, use_translation: bool = False):

        root = Path(root)

        self.gloss_annotation: list[GlossAnnotation] = cast(
            list[GlossAnnotation],
            csv.DictReader((root / "gloss_annotation.csv").open(), delimiter=",")
        )

        self.russian_translation: list[RussianTranslation] = cast(
            list[RussianTranslation],
            csv.DictReader((root / "russian_translation.csv").open(), delimiter=",")
        )

        sequences = list(chain.from_iterable(
            (
                FluentSigners50Item({
                    'input': torch.from_numpy(
                        np.genfromtxt(
                            sample,
                            delimiter=',',
                            dtype=np.float32,
                            encoding='utf-8'
                        )
                        if sample.suffix == '.csv' else
                        torch.from_numpy(np.load(sample, allow_pickle=True))
                        if sample.suffix == '.npv' else
                        raise_exception(
                            ValueError(f"Unsupported file format: {sample.suffix}")
                        )
                    ),
                    'label': (ann['Translation'] if 'Translation' in ann else ann['Gloss']).split(),
                    'metadata': FluentSiners50Metadata({
                        'person': (pr := cast(Result, parse(
                            'P{person:d}_S{sentence:d}_{variation:d}', sample.name
                        ))).named['person'],
                        'variation': pr.named['variation'],
                    })
                })
                for sample in sentence.iterdir()
            )
            for ann in (self.russian_translation if use_translation else self.gloss_annotation)
            if (sentence := root / ann['ID']).exists()
        ))

        self.dataset = Dataset[FluentSiners50Metadata].from_sequences(
            inputs = [item['input'] for item in sequences],
            labels = [item['label'] for item in sequences],
            blank_label=' ',
            metas=[item['metadata'] for item in sequences],
        )

    def save(self, dst: _PathLike):
        with open(dst, 'wb') as f:
            pickle.dump(self.dataset, f)

    @classmethod
    def load(cls, src: _PathLike) -> Self:
        with open(src, 'rb') as f:
            return pickle.load(f)
        
