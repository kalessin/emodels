"""
tools for huggingface compatibility
"""
import re
import sys
from random import random
from functools import partial
from collections import defaultdict
from typing import Generator, TypedDict, List, Tuple, Callable, Dict, Optional, Iterator

import torch
from datasets import Dataset as HuggingFaceDataset, DatasetDict as HuggingFaceDatasetDict
from datasets.arrow_dataset import Dataset as ArrowDataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import AutoModelForQuestionAnswering, Trainer, TrainingArguments, AutoTokenizer
from transformers.trainer_utils import EvalPrediction
from datasets.builder import DatasetGenerationError
from sklearn.metrics import f1_score


from emodels.datasets.utils import ExtractDatasetFilename
from emodels.datasets.stypes import DatasetBucket, ItemSample


class ExtractSample(TypedDict):
    markdown: str
    attribute: str
    start: int
    end: int


class TransformerTrainSample(TypedDict):
    input_ids: List[int]
    attention_mask: List[int]
    start_positions: int
    end_positions: int


def _adapt_attribute(attr: str) -> str:
    return attr.lower().replace("_", " ")


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
    try:
        validation = HuggingFaceDataset.from_generator(partial(_generator, "validation"))
    except DatasetGenerationError:
        validation = None
    test = HuggingFaceDataset.from_generator(partial(_generator, "test"))

    if validation is not None:
        ds = HuggingFaceDatasetDict({"train": train, "test": test, "validation": validation})
    else:
        ds = HuggingFaceDatasetDict({"train": train, "test": test})
    return ds


def process_sample_for_train(
    sample: ExtractSample, tokenizer: PreTrainedTokenizerBase, max_question_length: int = 12
) -> TransformerTrainSample:

    tokenized = tokenizer(sample["markdown"])
    input_ids = tokenized["input_ids"]

    tokens_start = tokenized.char_to_token(sample["start"])
    correction = 1
    while tokens_start is None:
        tokens_start = tokenized.char_to_token(sample["start"] - correction)
        correction += 1

    tokens_end = tokenized.char_to_token(sample["end"])
    correction = 1
    while tokens_end is None:
        tokens_end = tokenized.char_to_token(sample["end"] + correction)
        correction += 1

    max_length = tokenizer.model_max_length - max_question_length
    prefix_len = max_length // 2
    center = (tokens_start + tokens_end) // 2
    mstart = max(0, center - prefix_len)
    mend = mstart + max_length

    truncated_input_ids = input_ids[mstart:mend]
    if truncated_input_ids[0] != tokenizer.cls_token_id:
        truncated_input_ids = [tokenizer.cls_token_id] + truncated_input_ids[1:]
    if truncated_input_ids[-1] != tokenizer.sep_token_id:
        truncated_input_ids = truncated_input_ids[:-1] + [tokenizer.sep_token_id]

    answer_start = tokens_start - mstart
    answer_end = tokens_end - mstart

    question = f"Which is the {_adapt_attribute(sample['attribute'])}?"
    tokenized_question_ids = tokenizer(question)["input_ids"][1:]

    context_question_input_ids = truncated_input_ids + [tokenizer.sep_token_id] + tokenized_question_ids
    attention_mask = [1] * len(context_question_input_ids)

    pads = [tokenizer.pad_token_id] * (tokenizer.model_max_length - len(context_question_input_ids))
    context_question_input_ids += pads
    attention_mask += [0] * len(pads)

    return TransformerTrainSample(
        {
            "input_ids": context_question_input_ids,
            "attention_mask": attention_mask,
            "start_positions": answer_start,
            "end_positions": answer_end,
        }
    )


def prepare_datasetdict(
    hf: HuggingFaceDatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    load_from_cache_file=True,
    max_length: Optional[int] = None,
) -> HuggingFaceDatasetDict:
    mapper = partial(process_sample_for_train, tokenizer=tokenizer, max_length=max_length)
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


class QuestionAnswerer:
    def __init__(self, model_path: str):
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def answer(self, question: str, context: str) -> str:
        input_ids = self.tokenizer.encode(question, context)
        output = self.model(torch.tensor([input_ids]))
        answer_start = torch.argmax(output.start_logits)
        answer_end = torch.argmax(output.end_logits)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        return self.tokenizer.convert_tokens_to_string(tokens[answer_start:answer_end]).strip()


def evaluate(
    eds: Iterator[ItemSample],
    model: str,
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

    question_answerer = QuestionAnswerer(model)
    count = 0
    for sample in eds:
        source = sample["source"]
        bucket = sample["dataset_bucket"]
        for attr, idx in sample["indexes"].items():
            if random() > rate:
                continue
            count += 1
            attr_adapted = _adapt_attribute(attr)
            model_answer = _clean(question_answerer.answer(f"Which is the {attr_adapted}?", sample["markdown"]))
            real_answer = _clean(sample["markdown"][slice(*idx)])
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
