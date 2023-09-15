"""
tools for huggingface compatibility
"""
import re
import sys
from random import random
from functools import partial
from collections import defaultdict
from typing import Generator, TypedDict, List, Tuple, Callable, Dict, Optional, Iterator

from datasets import Dataset as HuggingFaceDataset, DatasetDict as HuggingFaceDatasetDict
from datasets.arrow_dataset import Dataset as ArrowDataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import AutoModelForQuestionAnswering, Trainer, TrainingArguments, pipeline
from transformers.trainer_utils import EvalPrediction
from sklearn.metrics import f1_score


from emodels.datasets.utils import ExtractDatasetFilename
from emodels.datasets.stypes import DatasetBucket, ItemSample


class ExtractSample(TypedDict):
    markdown: str
    attribute: str
    start: int
    end: int


def to_hfdataset(target: ExtractDatasetFilename, **kwargs) -> HuggingFaceDatasetDict:
    """
    Convert to HuggingFace Dataset suitable for usage in transformers
    """

    def _generator(bucket: DatasetBucket) -> Generator[ExtractSample, None, None]:
        for sample in target.iter(**kwargs):
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
    question = f"Which is the {truncated['attribute']}?"
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
    **training_args_kw,
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

    trargs = dict(
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
    trargs.update(**training_args_kw)

    training_args = TrainingArguments(**trargs)

    model = AutoModelForQuestionAnswering.from_pretrained(hg_model_name)

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=processed_train_data,  # training dataset
        eval_dataset=processed_validation_data,  # evaluation dataset
        compute_metrics=eval_metrics,
    )

    return model, trainer, processed_test_data


def evaluate(
    eds: Iterator[ItemSample],
    model: str,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    print_each: int = 50,
    rate: float = 0.1,
) -> Dict[str, Dict[DatasetBucket, float]]:
    def _clean(txt):
        txt = re.sub(r"^\W+", "", txt)
        txt = re.sub(r"\W+$", "", txt)
        return txt

    def _to_dict(ddict):
        return_value = dict(ddict)
        for key in return_value.keys():
            return_value[key] = dict(return_value[key])
        return return_value

    score: Dict[str, Dict[DatasetBucket, float]] = defaultdict(lambda: defaultdict(float))
    totals: Dict[str, Dict[DatasetBucket, int]] = defaultdict(lambda: defaultdict(int))

    question_answerer = pipeline(task="question-answering", model=model, tokenizer=tokenizer)
    count = 0
    for sample in eds:
        source = sample["source"]
        bucket = sample["dataset_bucket"]
        for attr, idx in sample["indexes"].items():
            if random() > rate:
                continue
            count += 1
            model_answer = _clean(
                question_answerer(question=f"Which is the {attr}?", context=sample["markdown"])["answer"]
            )
            real_answer = sample["markdown"][slice(*idx)]
            model_answer = model_answer.replace(" ", "")
            real_answer = real_answer.replace(" ", "")
            totals[source][bucket] += 1
            if real_answer in model_answer:
                score[source][bucket] += len(real_answer) / len(model_answer)
            elif source not in score or bucket not in score[source]:
                score[source][bucket] = 0.0
            if count % print_each == 0:
                print("Score count: ", _to_dict(score), "Total count: ", _to_dict(totals), file=sys.stderr)
    for source in score.keys():
        for bucket in score[source].keys():
            score[source][bucket] /= totals[source][bucket]

    return _to_dict(score)
