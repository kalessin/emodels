"""
tools for huggingface compatibility
"""
from functools import partial
from typing import Generator, TypedDict, List

from datasets import Dataset as HuggingFaceDataset, DatasetDict as HuggingFaceDatasetDict
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

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


def truncate_sample(sample: ExtractSample, tokenizer: PreTrainedTokenizerBase) -> ExtractSample:
    max_length = tokenizer.model_max_length
    prefix_len = max_length // 2
    suffix_len = max_length - prefix_len
    center = (sample["start"] + sample["end"]) // 2
    mstart = max(0, center - prefix_len)
    mend = min(len(sample["markdown"]), center + suffix_len)
    return ExtractSample(
        {
            "markdown": sample["markdown"][mstart:mend],
            "attribute": sample["attribute"],
            "start": sample["start"] - mstart,
            "end": sample["end"] - mstart,
        }
    )


class TransformerTrainSample(TypedDict):
    input_ids: List[int]
    attention_mask: List[int]
    start_positions: int
    end_positions: int


def process_sample_for_train(sample: ExtractSample, tokenizer: PreTrainedTokenizerBase) -> TransformerTrainSample:
    truncated = truncate_sample(sample, tokenizer)
    question = f"Extract the {truncated['attribute']}."
    tokenized_data = tokenizer(truncated["markdown"], question, padding="max_length")

    start = tokenized_data.char_to_token(truncated["start"])
    correction = 1
    while start is None:
        start = tokenized_data.char_to_token(truncated["start"] - correction)
        correction += 1

    end = tokenized_data.char_to_token(truncated["end"])
    correction = 1
    while end is None:
        end = tokenized_data.char_to_token(truncated["end"] + correction)
        correction += 1

    return TransformerTrainSample(
        {
            "input_ids": tokenized_data["input_ids"],
            "attention_mask": tokenized_data["attention_mask"],
            "start_positions": start,
            "end_positions": end,
        }
    )


def prepare_datasetdict(
    hf: HuggingFaceDatasetDict, tokenizer: PreTrainedTokenizerBase, load_from_cache_file=True
) -> HuggingFaceDatasetDict:
    mapper = partial(process_sample_for_train, tokenizer=tokenizer)
    hff = hf.map(mapper, load_from_cache_file=load_from_cache_file)
    return hff
