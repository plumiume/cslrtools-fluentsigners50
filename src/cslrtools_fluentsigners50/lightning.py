try:
    import lightning
except ImportError:
    raise ImportError(
        "lightning.pytorch is required to use LightningDataModule. "
        f"Please install it with 'pip install {__package__}[lightning]'"
    )

import pickle
from pathlib import Path
import torch
from cslrtools.dataset.pytorch import Dataset
from .pytorch import FluentSigners50Metadata
from cslrtools.dataset.lightning import LightningDataModule, StageString

class FluentSigners50DataModule(LightningDataModule[FluentSigners50Metadata]):

    def __init__(
        self,
        dataset: Dataset[FluentSigners50Metadata],
        stages: list[list[StageString]],
        common_kwargs: LightningDataModule.DataLoaderCommonKwargs = {}
        ):
        super().__init__(dataset, stages, common_kwargs)

    @classmethod
    def load_or_create_cross_validataion_groups(
        cls,
        groups_pkl: Path,
        dataset: Dataset[FluentSigners50Metadata],
        num_splits: int = 5,
        ):

        if groups_pkl.exists():
            groups: torch.Tensor = pickle.load(groups_pkl.open('rb'))
        else:
            people = set(m['person'] for m in dataset._metas)
            groups = torch.rand((len(people),)).argsort() % num_splits
            pickle.dump(groups, groups_pkl.open('wb'))

        return groups

    @classmethod
    def generate_cross_validation_splits_by_person(
        cls,
        dataset: Dataset[FluentSigners50Metadata],
        num_splits: int = 5,
        common_kwargs: LightningDataModule.DataLoaderCommonKwargs = {},
        groups: torch.Tensor | None = None
        ):

        if groups is None:
            people = set(m['person'] for m in dataset._metas)
            groups = torch.rand((len(people),)).argsort() % num_splits

        for valid_idx in range(num_splits):
            yield cls(
                dataset=dataset,
                stages=[
                    ['val', 'test']
                    if groups[mi['person']] == valid_idx
                    else ['train']
                    for mi in dataset._metas
                ],
                common_kwargs=common_kwargs
            )

        return groups
