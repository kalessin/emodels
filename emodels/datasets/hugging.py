"""
tools for huggingface compatibility
"""
from functools import partial

from datasets import Dataset as HuggingFaceDataset, DatasetDict as HuggingFaceDatasetDict

from emodels.datasets.utils import ExtractDatasetFilename, DatasetBucket


def to_hfdataset(target: ExtractDatasetFilename) -> HuggingFaceDatasetDict:
    """
    Convert to HuggingFace Dataset suitable for usage in transformers
    """

    def _generator(bucket: DatasetBucket):
        for sample in target:
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
