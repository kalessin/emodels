"""
tools for huggingface compatibility
"""
from functools import partial
from typing import Generator, TypedDict, List, Tuple, Callable, Dict

from datasets import Dataset as HuggingFaceDataset, DatasetDict as HuggingFaceDatasetDict
from datasets.arrow_dataset import Dataset as ArrowDataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import AutoModelForQuestionAnswering, Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction
from sklearn.metrics import f1_score


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
    validation = HuggingFaceDataset.from_generator(partial(_generator, "validation"))
    test = HuggingFaceDataset.from_generator(partial(_generator, "test"))

    ds = HuggingFaceDatasetDict({"train": train, "test": test, "validation": validation})
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


def compute_f1_metrics(pred: EvalPrediction) -> Dict[str, float]:
    start_labels = pred.label_ids[0]
    start_preds = pred.predictions[0].argmax(-1)
    end_labels = pred.label_ids[1]
    end_preds = pred.predictions[1].argmax(-1)

    f1_start = f1_score(start_labels, start_preds, average="macro")
    f1_end = f1_score(end_labels, end_preds, average="macro")

    return {
        "f1_start": f1_start,
        "f1_end": f1_end,
    }


def get_qatransformer_trainer(
    hff: HuggingFaceDataset,
    hg_model_name: str,
    output_dir: str,
    eval_metrics: Callable[[EvalPrediction], Dict] = compute_f1_metrics,
) -> Tuple[AutoModelForQuestionAnswering, Trainer, ArrowDataset]:
    columns_to_return = ["input_ids", "attention_mask", "start_positions", "end_positions"]

    processed_train_data = hff["train"].flatten()
    processed_train_data.set_format(type="pt", columns=columns_to_return)

    processed_test_data = hff["test"].flatten()
    processed_test_data.set_format(type="pt", columns=columns_to_return)

    if "validation" in hff:
        processed_validation_data = hff["validation"].flatten()
        processed_validation_data.set_format(type="pt", columns=columns_to_return)
    else:
        processed_validation_data = processed_test_data

    training_args = TrainingArguments(
        output_dir=output_dir,  # output directory
        overwrite_output_dir=True,
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=8,  # batch size per device during training
        per_device_eval_batch_size=8,  # batch size for evaluation
        warmup_steps=20,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir=None,  # directory for storing logs
        logging_steps=50,
    )

    model = AutoModelForQuestionAnswering.from_pretrained(hg_model_name)

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=processed_train_data,  # training dataset
        eval_dataset=processed_validation_data,  # evaluation dataset
        compute_metrics=eval_metrics,
    )

    return model, trainer, processed_test_data
