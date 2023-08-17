"""
tools for huggingface compatibility
"""
from functools import partial
from typing import Generator

from datasets import Dataset as HuggingFaceDataset, DatasetDict as HuggingFaceDatasetDict

from emodels.datasets.utils import ExtractDatasetFilename, DatasetBucket, ExtractSample


def to_hfdataset(target: ExtractDatasetFilename) -> HuggingFaceDatasetDict:
    """
    Convert to HuggingFace Dataset suitable for usage in transformers
    """

    def _generator(bucket: DatasetBucket) -> Generator[ExtractSample, None, None]:
        for sample in ExtractDatasetFilename(target):
            if sample["dataset_bucket"] != bucket:
                continue
            for key, idx in sample["indexes"].items():
                if idx is None:
                    continue
                yield ExtractSample(
                    {
                        "markdown": sample["markdown"],
                        "attribute": key,
                        "start": idx[0],
                        "end": idx[1],
                    }
                )

    train = HuggingFaceDataset.from_generator(partial(_generator, "train"))
    test = HuggingFaceDataset.from_generator(partial(_generator, "test"))

    ds = HuggingFaceDatasetDict({"train": train, "test": test})
    return ds


# def truncate_sample(sample: Sample, tokenizer: PreTrainedTokenizerBase) -> Sample:
#     max_length = tokenizer.model_max_length
#     prefix_len = max_length // 2
#     suffix_len = max_length - prefix_len
#     center = sample["start"]
#     return {
#         "markdown":
#     }
