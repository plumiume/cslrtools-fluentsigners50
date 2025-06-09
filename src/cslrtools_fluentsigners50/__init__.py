import csv
from parse import parse, Result
import pickle
from itertools import chain
from typing import TypedDict, cast, Self, Iterable
from pathlib import Path
from cslrtools.dataset.pytorch import Metadata, Dataset
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, Future, wait
from tqdm import tqdm
from halo import Halo
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
    input: Future[torch.Tensor]
    label: list[str]
    metadata: FluentSiners50Metadata

class FluentSigners50:

    def _load_input(self, sample: Path) -> torch.Tensor:
        if sample.suffix == '.csv':
            return torch.from_numpy(np.genfromtxt(sample, delimiter=',', encoding='utf8'))
        elif sample.suffix == '.npy':
            return torch.from_numpy(np.load(sample, allow_pickle=True))
        else:
            raise ValueError(f"Unsupported file format: {sample.suffix}")

    @property
    def dataset(self) -> Dataset[FluentSiners50Metadata]:
        return self._dataset

    def __init__(self, root: _PathLike, landmarks: str, use_translation: bool = False, quiet: bool = False):

        root = Path(root)

        self.gloss_annotation: list[GlossAnnotation] = cast(
            list[GlossAnnotation],
            list(csv.DictReader((root / "gloss_annotation.csv").open(encoding='utf8'), delimiter=","))
        )

        self.russian_translation: list[RussianTranslation] = cast(
            list[RussianTranslation],
            list(csv.DictReader((root / "russian_translation.csv").open(encoding='utf8'), delimiter=","))
        )

        pool = ThreadPoolExecutor(8)

        with tqdm(
            list(self.russian_translation if use_translation else self.gloss_annotation),
            desc="Loading annotations",
            disable=quiet,
            ) as progress:

            sequences = list(chain.from_iterable(
                (
                    FluentSigners50Item({
                        # 'input': self._load_input(sample),
                        'input': pool.submit(self._load_input, sample),
                        'label': (ann['Translation'] if 'Translation' in ann else ann['Gloss']).split(),
                        'metadata': FluentSiners50Metadata({
                            'person': (pr := cast(Result, parse(
                                'P{person:d}_S{sentence:d}_{variation:d}', sample.stem
                            ))).named['person'],
                            'variation': pr.named['variation'],
                        })
                    })
                    for sample in sentence.iterdir()
                )
                for ann in cast(Iterable[GlossAnnotation | RussianTranslation], progress)
                if (sentence := root / landmarks / f'{ann["ID"]:0>3}').exists()
            ))

        dataset_init_args = {
            'inputs': list(tqdm(
                (wait([item['input']]) and item['input'].result() for item in sequences),
                total=len(sequences),
                desc="Loading inputs",
                disable=quiet
            )),
            'labels': [item['label'] for item in sequences],
            'blank_label': ' ',
            'metas': [item['metadata'] for item in sequences],
        }

        with Halo(text="Creating dataset ...", spinner='dots', enabled=not quiet):
            self._dataset = Dataset[FluentSiners50Metadata].from_sequences(
                **dataset_init_args
            )
        print('✅ Creating dataset finished.')

    def save(self, dst: _PathLike, spiner_enabled: bool = True):
        with Halo(
                text=f'Saving dataset to {dst} ...', spinner='dots', enabled=spiner_enabled
            ), open(dst, 'wb') as f:
            pickle.dump(self, f)
        print(f'✅ Saving dataset to {dst} finished.')

    @classmethod
    def load(cls, src: _PathLike, spiner_enabled: bool = True) -> Self:
        with Halo(
                text=f'Loading dataset from {src} ...', spinner='dots', enabled=spiner_enabled
            ), open(src, 'rb') as f: 
            ret = pickle.load(f)
        print(f'✅ Loading dataset from {src} finished.')
        return ret
