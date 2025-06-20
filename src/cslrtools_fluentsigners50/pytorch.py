# Copyright 2025 plumiume.com
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

import csv
from parse import parse, Result
from itertools import chain
from typing import TypedDict, Literal, cast, Self, Iterable
from pathlib import Path
from cslrtools.dataset.pytorch import Metadata, Dataset
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, Future
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
    _tag: Literal['GlossAnnotation']
    ID: str
    Gloss: str

class RussianTranslation(TypedDict):
    _tag: Literal['RussianTranslation']
    ID: str
    Translation: str

class FluentSigners50Metadata(Metadata):
    person: int
    variation: int

class FluentSigners50:

    def _load_input(self, sample: Path, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if sample.suffix == '.csv':
            ret = torch.from_numpy(np.genfromtxt(sample, delimiter=',', encoding='utf8'))
        elif sample.suffix == '.npy':
            ret = torch.from_numpy(np.load(sample))
        else:
            raise ValueError(f"Unsupported file format: {sample.suffix}")
        return ret.to(dtype=dtype)

    @property
    def dataset(self) -> Dataset[FluentSigners50Metadata]:
        return self._dataset

    def __init__(
        self,
        root: _PathLike,
        landmarks: str,
        use_translation: bool = False,
        quiet: bool = False,
        dtype: torch.dtype = torch.float32,
        ):

        root = Path(root)

        self.gloss_annotation = [
            cast(GlossAnnotation, line | {'_tag': 'GlossAnnotation'})
            for line in
            csv.DictReader((root / "gloss_annotation.csv").open(encoding='utf8'), delimiter=",")
        ]

        self.russian_translation = [
            cast(RussianTranslation, line | {'_tag': 'RussianTranslation'})
            for line in
            csv.DictReader((root / "russian_translation.csv").open(encoding='utf8'), delimiter=",")
        ]

        futures: list[Future[torch.Tensor]] = []
        labels: list[list[str]] = []
        metadata: list[FluentSigners50Metadata] = []

        pool = ThreadPoolExecutor(8)

        with (
            tqdm(
                iterable=self.russian_translation if use_translation else self.gloss_annotation,
                desc="Loading annotations",
                disable=quiet,
                position=0,
            ) as annotation_progress,
            tqdm(
                desc="Loading inputs",
                position=1,
            ) as input_progress,
            ):

            input_progress.total = 0

            try:
                for ann in cast(Iterable[GlossAnnotation | RussianTranslation], annotation_progress):
                    sentence = root / landmarks / f'{ann["ID"]:0>3}'
                    if not sentence.exists():
                        continue

                    for sample in sentence.iterdir():
                        ftr = pool.submit(self._load_input, sample, dtype)
                        ftr.add_done_callback(
                            lambda ftr:
                                input_progress.update()
                        )
                        lbl = (
                            ann['Translation']
                            if ann['_tag'] == 'RussianTranslation'
                            else ann['Gloss']
                        ).split()
                        meta = FluentSigners50Metadata({
                            'person': (pr := cast(Result, parse(
                                'P{person:d}_S{sentence:d}_{variation:d}', sample.stem
                            ))).named['person'],
                            'variation': pr.named['variation'],
                        })
                        
                        input_progress.total += 1
                        futures.append(ftr)
                        labels.append(lbl)
                        metadata.append(meta)

                input_progress.pos = 0
                pool.shutdown()

            except KeyboardInterrupt as e:
                pool.shutdown(wait=False, cancel_futures=True)
                raise e

        with Halo(text="Creating dataset ...", spinner='dots', enabled=not quiet):
            self._dataset = Dataset[FluentSigners50Metadata].from_sequences(
                inputs=[ftr.result() for ftr in futures],
                labels=labels,
                blank_label=' ',
                metas=metadata,
            )
        print('✅ Creating dataset finished.')

    def save(self, dst: _PathLike, spiner_enabled: bool = True):
        with Halo(
                text=f'Exporting dataset to {dst} ...', spinner='dots', enabled=spiner_enabled
            ), open(dst, 'wb') as f:
            torch.save(self, f)
        print(f'✅ Exporting dataset to {dst} finished.')

    @classmethod
    def load(cls, src: _PathLike, spiner_enabled: bool = True) -> Self:
        torch.serialization.add_safe_globals([
            cls,
            FluentSigners50Metadata, RussianTranslation, GlossAnnotation,
            Metadata, Dataset
            ])
        with Halo(
                text=f'Loading dataset from {src} ...', spinner='dots', enabled=spiner_enabled
            ), open(src, 'rb') as f:
            ret = torch.load(f, weights_only=True)
        print(f'✅ Loading dataset from {src} finished.')
        torch.serialization.clear_safe_globals()
        return ret
