"""
tools for huggingface compatibility
"""
import re
import sys
from random import random
from functools import partial
from collections import defaultdict
from typing import Generator, TypedDict, List, Tuple, Callable, Dict, Iterator, Optional

import torch
from datasets import Dataset as HuggingFaceDataset, DatasetDict as HuggingFaceDatasetDict
from datasets.arrow_dataset import Dataset as ArrowDataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import AutoModelForQuestionAnswering, Trainer, TrainingArguments, AutoTokenizer, pipeline
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


def to_hfdataset(
    target: ExtractDatasetFilename, **kwargs
) -> HuggingFaceDatasetDict:
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
    max_question_length: int = 12,
) -> HuggingFaceDatasetDict:
    mapper = partial(process_sample_for_train, tokenizer=tokenizer, max_question_length=max_question_length)
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


def _clean(txt):
    txt = re.sub(r"^\W+", "", txt)
    txt = re.sub(r"\W+$", "", txt)
    return txt


def filter_empty_string(text: str) -> bool:
    return bool(text)


class QuestionAnswerer:
    def __init__(self, model_path: str, max_vectorized_overlaps: int = 7):
        """
        When context is big and does not fit the model size, it tries with several overlapped windows following
        this strategy:
        - generates many overlap sizes, starting from initial_window_overlap.
        - for each overlap size, it tests all the possible overlap windows until a result with a score equal or bigger
          than score_threshold is achieved.
        - if after all iterations, score_threshold is not achieved, it just return the best result.

        The strategy may slow down the prediction a lot as the context is bigger and the answer difficult to find
        (if any) but it achieves much better accuracy.

        max_vectorized_overlaps (class parameter) - the bigger, the faster, but uses more memory. A big number may
          unstabilize your system, so be careful. The default value is fair. You may want to increase it if you are
          running the model in a powerful machine with lots of memory available.
        """
        self._model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_vectorized_overlaps = max_vectorized_overlaps

    def __call__(
        self,
        question: str,
        context: str,
        initial_window_overlap: int = 2,
        score_threshold: float = -10.0,
        filters: Optional[List[Callable[[str], bool]]] = None,
    ) -> Tuple[str, float]:
        filters = filters or []
        filters.insert(0, filter_empty_string)

        best_result: str = ""
        best_score: float = -torch.inf

        while True:
            result, score = self.base_predict(question, context, initial_window_overlap, score_threshold)
            if score <= best_score:
                return best_result, best_score
            best_result = result
            best_score = score
            for fltr in filters:
                if not fltr(result):
                    # retry with a bigger score
                    score_threshold = score
                    break
            else:
                return result, score

    def base_predict(
        self,
        question: str,
        context: str,
        initial_window_overlap: int = 2,
        score_threshold: float = -10.0,
    ) -> Tuple[str, float]:
        context_input_ids = self.tokenizer.encode(context)[1:]
        question_input_ids = self.tokenizer.encode(question)

        max_context_len = min(len(context_input_ids), self.tokenizer.model_max_length - len(question_input_ids)) - 1

        best_best_answer: str = ""
        best_best_score = -torch.inf

        window_overlap_sizes = list(
            range(initial_window_overlap, max(max_context_len, initial_window_overlap + 1), initial_window_overlap)
        )
        while window_overlap_sizes:
            window_overlap = window_overlap_sizes.pop(0)
            window_step = max_context_len - window_overlap
            windows_starts = list(range(0, len(context_input_ids) - 1 - max_context_len + window_step, window_step))
            while windows_starts:
                apply_windows_starts, windows_starts = (
                    windows_starts[:self.max_vectorized_overlaps],
                    windows_starts[self.max_vectorized_overlaps:],
                )
                inputs = []
                attention_masks = []
                for start in apply_windows_starts:
                    context_ids = context_input_ids[start:start + max_context_len]
                    if context_ids[-1] != self.tokenizer.sep_token_id:
                        context_ids = context_ids[:-1] + [self.tokenizer.sep_token_id]
                    input_ids = question_input_ids + [self.tokenizer.sep_token_id] + context_ids
                    input_ids = input_ids + [self.tokenizer.pad_token_id] * (
                        self.tokenizer.model_max_length - len(input_ids)
                    )
                    inputs.append(input_ids)
                    attention_mask = [int(t != self.tokenizer.pad_token_id) for t in input_ids]
                    attention_masks.append(attention_mask)
                try:
                    outputs = self._model(torch.tensor(inputs), attention_mask=torch.tensor(attention_masks))
                except RuntimeError:
                    print(
                        f"Could not allocate memory for {len(inputs)} vectorized inputs."
                        "I will set max_vectorized_overlaps to {len(inputs) - 1}.",
                        file=sys.stderr,
                    )
                    self.max_vectorized_overlaps = len(inputs) - 1
                    # retry same overlap size with smaller tensors
                    window_overlap_sizes.insert(0, window_overlap)
                    break
                scores_start = torch.max(outputs.start_logits, dim=1)
                scores_end = torch.max(outputs.end_logits, dim=1)
                scores = (scores_start.values + scores_end.values) / 2
                best_idx = int(torch.argmax(scores))
                answer_start = scores_start.indices[best_idx]
                if answer_start < len(question_input_ids) + 1:
                    continue
                answer_end = scores_end.indices[best_idx]
                if answer_end <= answer_start:
                    continue
                tokens = self.tokenizer.convert_ids_to_tokens(inputs[best_idx])
                best_answer = self.tokenizer.convert_tokens_to_string(tokens[answer_start:answer_end]).strip()
                best_score = scores[best_idx]
                if best_score > score_threshold:
                    return best_answer, float(scores[best_idx])
                if best_score > best_best_score:
                    best_best_score = best_score
                    best_best_answer = best_answer
        return best_best_answer, float(best_best_score)


class HFQuestionAnswerer:
    def __init__(self, model_path: str):
        self.qa = pipeline(task="question-answering", model=model_path)

    def __call__(self, question: str, context: str) -> Tuple[str, float]:
        result = self.qa(question=question, context=context, align_to_words=False)
        return _clean(result["answer"]), result["score"]


def evaluate(
    eds: Iterator[ItemSample],
    model_path: str,
    print_each: int = 50,
    rate: float = 1.0,
    dataset_buckets: Tuple[DatasetBucket, ...] = (),
    sources: Tuple[str, ...] = (),
    attributes: Tuple[str, ...] = (),
    qaclass=QuestionAnswerer,
    **qa_kwargs,
) -> Dict[str, Dict[DatasetBucket, float]]:
    def _to_dict(ddict):
        return_value = dict(ddict)
        for key in return_value.keys():
            return_value[key] = dict(return_value[key])
        return return_value

    score: Dict[str, Dict[DatasetBucket, float]] = defaultdict(lambda: defaultdict(float))
    totals: Dict[str, Dict[DatasetBucket, int]] = defaultdict(lambda: defaultdict(int))

    question_answerer = qaclass(model_path)
    count = 0
    for sample in eds:
        source = sample["source"]
        if sources and source not in sources:
            continue
        bucket = sample["dataset_bucket"]
        if dataset_buckets and bucket not in dataset_buckets:
            continue
        for attr, idx in sample["indexes"].items():
            if attributes and attr not in attributes:
                continue
            if random() > rate:
                continue
            count += 1
            attr_adapted = _adapt_attribute(attr)
            model_answer = question_answerer(f"Which is the {attr_adapted}?", sample["markdown"], **qa_kwargs)[0]
            real_answer = sample["markdown"][slice(*idx)]
            totals[source][bucket] += 1
            if real_answer in model_answer:
                score[source][bucket] += len(real_answer) / len(model_answer)
            elif model_answer in real_answer:
                score[source][bucket] += len(model_answer) / len(real_answer)
            elif source not in score or bucket not in score[source]:
                score[source][bucket] = 0.0
            if count % print_each == 0:
                if print_each == 1:
                    print(
                        "Score count: ",
                        _to_dict(score),
                        "Total count: ",
                        _to_dict(totals),
                        "Model:",
                        repr(model_answer),
                        "Real:",
                        repr(real_answer),
                        file=sys.stderr,
                    )
                else:
                    print("Score count: ", _to_dict(score), "Total count: ", _to_dict(totals), file=sys.stderr)
    for source in score.keys():
        for bucket in score[source].keys():
            score[source][bucket] /= totals[source][bucket]

    return _to_dict(score)
