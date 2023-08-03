"""
"""
import os
from functools import partial
from typing import Tuple, Optional

from typing_extensions import Self
from datasets import Dataset as HuggingFaceDataset, DatasetDict as HuggingFaceDatasetDict

from emodels.config import EMODELS_ITEMS_DIR
from emodels.datasets.utils import DatasetFilename, get_random_dataset, DatasetBucket, DEFAULT_DATASET_RATIO


class ItemsDatasetFilename(DatasetFilename):
    @classmethod
    def build_from_items(
        cls,
        name: str,
        project: str,
        classes: Optional[Tuple[str]] = None,
        dataset_ratio: Tuple[float, ...] = DEFAULT_DATASET_RATIO,
    ) -> Self:
        """
        Build a dataset dict from extracted items in user dataset folder.
        - name is a name for the dataset. It will determine the storing filename.
        - project is the name of the project the dataset belongs to. It will determine the storing filename.
        - If classes is a tuple of strings, select only the specified
        item subfolders.
        - dataset_ratio is the same for get_random_dataset() and determines how samples are distributed
          among train, test and validation buckets.
        """
        result = cls.local_by_name(name, project)
        if os.path.exists(result):
            raise ValueError(
                "Output file already exists. "
                f'open with {cls.__name__}.local_by_name("{name}", "{project}") or remove it for rebuilding'
            )
        for sf in os.listdir(EMODELS_ITEMS_DIR):
            for f in os.listdir(os.path.join(EMODELS_ITEMS_DIR, sf)):
                df = DatasetFilename(os.path.join(EMODELS_ITEMS_DIR, sf, f))
                for sample in df:
                    sample["dataset_bucket"] = get_random_dataset(dataset_ratio)
                    result.append(sample)
        return result

    def to_hfdataset(self) -> HuggingFaceDatasetDict:
        """
        Convert to HuggingFace Dataset suitable for usage in transformers
        """

        def _generator(bucket: DatasetBucket):
            for sample in self:
                if sample["dataset_bucket"] != bucket:
                    continue
                for key, idx in sample["indexes"].items():
                    if idx is None:
                        continue
                    yield {
                        "markdown": sample["markdown"],
                        "attribute": key,
                        "start": idx[0],
                        "end": idx[1],
                    }

        train = HuggingFaceDataset.from_generator(partial(_generator, "train"))
        test = HuggingFaceDataset.from_generator(partial(_generator, "test"))

        ds = HuggingFaceDatasetDict({"train": train, "test": test})
        return ds
