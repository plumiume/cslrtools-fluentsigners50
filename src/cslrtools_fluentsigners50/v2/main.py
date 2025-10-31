from typing import cast, TypeVar, Hashable, Any, Literal
from dataclasses import dataclass
import re
import csv
import torch
from pathlib import Path
from torch import Tensor

from cslrtools.dataset.v2.torch.dataset import Dataset, DatasetItem, JSON
from cslrtools.dataset.v2.torch import loader

FS50_SAMPLE_REGEX_PATTERN = re.compile(
    r'^P(?P<signer>\d{2})_S(?P<sentence>\d{3})_(?P<take>\d{2})$'
)
FS50_SAMPLE_GLOB_PATTERN = 'P??_S???_??*'
GLOSS_ANNOTATION_CSV = 'gloss_annotation.csv'
RUSSIAN_TRANSLATION_CSV = 'russian_translation.csv'
BLANK = ' '

_Kvid = TypeVar('_Kvid', bound=str, covariant=True)
_Klm = TypeVar('_Klm', bound=str, covariant=True)

_A = TypeVar('_A', bound=Hashable)
_B = TypeVar('_B', bound=Hashable)

class InversibleDict(dict[_A, _B]):
    """A dictionary that can be inverted to swap keys and values."""
    def inverse(self) -> 'InversibleDict[_B, _A]':
        """Return an inverted dictionary with keys and values swapped."""
        return InversibleDict({
            v: k for k, v in self.items()
        })

@dataclass
class Metadata:
    """Metadata for a single FS50 dataset item."""
    signer: int
    sentence: int
    take: int

TargetKey = Literal['gloss_indices', 'russian_translation']

class FS50Item(DatasetItem[_Kvid, _Klm, TargetKey]):
    """A single item from the FS50 dataset."""
    metadata: Metadata

class FS50Dataset(Dataset[_Kvid, _Klm, TargetKey]):
    """FluentSigners50 Dataset.

    The FluentSigners50 dataset consists of videos of sign language sentences performed by multiple signers,
    along with their corresponding gloss annotations and translations.

    The dataset is organized into a directory structure that reflects the signer, sentence, and take information.

    Args:
        base_root (Path): Path to the base root directory containing the FS50 dataset.
        processed_root (Path): Path to the processed root directory containing annotations and processed files.
        original_video_key (_Kvid): Key name for the original video files.
        use_original_video (bool): Whether to use the original video files from the base root directory.
        extra_video_files (dict[_Kvid, str]): Dictionary mapping extra video keys to their file names.
        landmark_files (dict[_Klm, str]): Dictionary mapping landmark keys to their file names.
        self_connections (dict[_Klm, Tensor]): Dictionary mapping landmark keys to their self-connection tensors.
        other_connections (dict[tuple[_Klm, _Klm], Tensor]): Dictionary mapping landmark key pairs to their connection tensors.
        enable_cache (bool): Whether to enable caching for data loaders.
    """

    def __init__(
        self,
        base_root: Path,
        # (base_root) / "KSLR_173_17_08" / "{sentence:03d}" / "P{signer:02d}_S{sentence:03d}_{take:02d}.mp4"
        # (base_root -> processed_root) / "gloss_annotation.csv"
        #     Id, Gloss
        # (base_root -> processed_root) / "rusian_translation.csv"
        #     Id, Translation
        processed_root: Path,
        # (processed_root) / "{sentence:03d}" / "P{signer:02d}_S{sentence:03d}_{take:02d}" / some files ... {guessable task-specific structure},
        original_video_key: _Kvid = 'original',
        use_original_video: bool = False,
        extra_video_files: dict[_Kvid, str] = {},
        landmark_files: dict[_Klm, str] = {},
        self_connection_files: dict[_Klm, str] = {},
        other_connection_files: dict[tuple[_Klm, _Klm], str] = {},
        enable_cache: bool = True,
        ):

        self.base_root = base_root
        self.processed_root = processed_root
        self.enable_cache = enable_cache

        # metadata
        self.gloss_annotation = self._load_gloss_annotation()
        # metadata
        self.russian_translation = self._load_russian_translation()

        unique_labels = set[str]()
        for labels in self.gloss_annotation.values():
            unique_labels.update(labels)

        self.index_to_label = InversibleDict(enumerate([
            BLANK, *sorted(unique_labels)
        ]))
        # metadata
        self.label_to_index = self.index_to_label.inverse()

        self.sentence_to_gloss_indices = self._create_sentence_to_gloss_indices()

        self.original_video_key = original_video_key
        self.use_original_video = use_original_video

        self.extra_video_files = extra_video_files
        self.landmark_files = landmark_files
        self.self_connections: dict[_Klm, Tensor] = self._load_self_connections(
            self_connection_files
        )
        self.other_connections: dict[tuple[_Klm, _Klm], Tensor] = self._load_other_connections(
            other_connection_files
        )

        self.entries = list(base_root.glob(FS50_SAMPLE_GLOB_PATTERN))

        self.lm_loaders: dict[str, loader.ArrayLoader[str, Any]] = self._configure_lm_loaders(
            self.landmark_files
        )

        # VideoLoaderインスタンスを__init__で生成
        self.video_loader = loader.VideoLoader(enable_cache=enable_cache)

    def _load_gloss_annotation(self) -> dict[int, list[str]]:
        """Load gloss annotations from the CSV file."""
        path = self.base_root / GLOSS_ANNOTATION_CSV
        if not path.exists():
            # フォールバック: processed_rootも探す
            path = self.processed_root / GLOSS_ANNOTATION_CSV
        with path.open() as f:
            records = csv.DictReader(f)
            return {
                int(r['Id']): r['Gloss'].split()
                for r in records
            }

    def _load_russian_translation(self) -> dict[int, list[str]]:
        """Load Russian translations from the CSV file."""
        path = self.base_root / RUSSIAN_TRANSLATION_CSV
        if not path.exists():
            # フォールバック: processed_rootも探す
            path = self.processed_root / RUSSIAN_TRANSLATION_CSV
        with path.open() as f:
            records = csv.DictReader(f)
            return {
                int(r['Id']): r['Translation'].split()
                for r in records
            }

    def _create_sentence_to_gloss_indices(self) -> dict[int, Tensor]:
        """Create a mapping from sentence IDs to tensors of gloss indices."""
        return {
            sentence_id: torch.tensor([
                self.label_to_index[label] for label in labels
            ])
            for sentence_id, labels in self.gloss_annotation.items()
        }

    def _load_self_connections(
        self,
        self_connection_files: dict[_Klm, str],
        ) -> dict[_Klm, Tensor]:

        connections: dict[_Klm, Tensor] = {}
        for klm, fname in self_connection_files.items():

            fpath = self.processed_root / fname

            match fpath.suffix:
                case '.npy':
                    connections[klm] = loader.NpyLoader(
                        fpath.stem
                    ).get_tensor(fpath, fpath.stem)
                case '.npz':
                    connections[klm] = loader.NpzLoader(
                        fpath.stem
                    ).get_tensor(fpath, fpath.stem)
                case '.safetensor':
                    connections[klm] = loader.SafetensorLoader(
                        fpath.stem
                    ).get_tensor(fpath, fpath.stem)
                case ext:
                    raise ValueError(f"Unsupported self-connection file extension: {ext}")

        return connections

    def _load_other_connections(
        self,
        other_connection_files: dict[tuple[_Klm, _Klm], str],
        ) -> dict[tuple[_Klm, _Klm], Tensor]:

        connections: dict[tuple[_Klm, _Klm], Tensor] = {}
        for (klm_a, klm_b), fname in other_connection_files.items():

            fpath = self.processed_root / fname

            match fpath.suffix:
                case '.npy':
                    connections[(klm_a, klm_b)] = loader.NpyLoader(
                        fpath.stem
                    ).get_tensor(fpath, fpath.stem)
                case '.npz':
                    connections[(klm_a, klm_b)] = loader.NpzLoader(
                        fpath.stem
                    ).get_tensor(fpath, fpath.stem)
                case '.safetensor':
                    connections[(klm_a, klm_b)] = loader.SafetensorLoader(
                        fpath.stem
                    ).get_tensor(fpath, fpath.stem)
                case ext:
                    raise ValueError(f"Unsupported other-connection file extension: {ext}")

        return connections

    def _configure_lm_loaders(
        self,
        landmark_files: dict[_Klm, str],
        ) -> dict[str, loader.ArrayLoader[str, Any]]:
        """Configure landmark loaders based on the file extensions."""
        lm_loaders: dict[str, loader.ArrayLoader[str, Any]] = {}
        for fname in landmark_files.values():
            fpath = Path(fname)
            match fpath.suffix:
                case '.npy':
                    lm_loaders[fname] = loader.NpyLoader(
                        fpath.stem,
                        enable_cache=self.enable_cache
                    )
                case '.npz':
                    lm_loaders[fname] = loader.NpzLoader(
                        fpath.stem,
                        enable_cache=self.enable_cache
                    )
                case '.safetensor':
                    lm_loaders[fname] = loader.SafetensorLoader(
                        fpath.stem,
                        enable_cache=self.enable_cache
                    )
                case ext:
                    raise ValueError(f"Unsupported landmark file extension: {ext}")
        return lm_loaders

    @property
    def metadata(self) -> dict[str, JSON]:
        """dataset metadata as a JSON-serializable dictionary"""
        return {
            'gloss_annotation': {
                str(k): v
                for k, v in self.gloss_annotation.items()
            },
            'russian_translation': {
                str(k): v
                for k, v in self.russian_translation.items()
            },
            'label_to_index': dict(self.label_to_index),
        }

    @metadata.setter
    def metadata(self, value: dict[str, JSON]):
        self.gloss_annotation = {
            int(k): v
            for k, v in cast(dict[str, list[str]], value['gloss_annotation']).items()
        }
        self.russian_translation = {
            int(k): v
            for k, v in cast(dict[str, list[str]], value['russian_translation']).items()
        }
        self.label_to_index = InversibleDict(
            cast(dict[str, int], value['label_to_index'])
        )
        self.index_to_label = self.label_to_index.inverse()
        self.sentence_to_gloss_indices = self._create_sentence_to_gloss_indices()

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> DatasetItem[_Kvid, _Klm, TargetKey]:

        entry_path = self.entries[index]

        match_obj = FS50_SAMPLE_REGEX_PATTERN.match(entry_path.stem)
        if not match_obj:
            raise ValueError(f"Invalid sample name: {entry_path.stem}")

        metadata = Metadata(
            signer=int(match_obj.group('signer')),
            sentence=int(match_obj.group('sentence')),
            take=int(match_obj.group('take')),
        )

        processed_sample_root = (
            self.processed_root /
            f"{metadata.sentence:03d}" /
            f"P{metadata.signer:02d}_S{metadata.sentence:03d}_{metadata.take:02d}"
        )

        videos: dict[_Kvid, Tensor] = {}
        if self.use_original_video:
            video_path = entry_path.with_suffix('.mp4')
            videos[self.original_video_key] = self.video_loader.get_tensor(video_path)

        for kvid, video_file in self.extra_video_files.items():
            video_path = processed_sample_root / video_file
            videos[kvid] = self.video_loader.get_tensor(video_path)

        landmarks: dict[_Klm, tuple[Tensor, Tensor]] = {}
        for klm, lm_file in self.landmark_files.items():
            lm_path = processed_sample_root / lm_file
            lm_loader = self.lm_loaders[lm_path.name]
            tensor = lm_loader.get_tensor(lm_path, lm_path.stem)
            landmarks[klm] = (tensor, self.self_connections[klm])
        items = FS50Item(
            videos=videos,
            landmarks=landmarks,
            connections=self.other_connections,
            targets={
                'gloss_indices': self.sentence_to_gloss_indices[metadata.sentence],
            }, # TODO: implement targets（gloss_indices以外は未実装）
        )
        items.metadata = metadata
        return items

