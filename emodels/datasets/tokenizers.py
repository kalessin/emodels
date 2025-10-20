"""
"""
import os
import shutil
from typing import Iterator

import sentencepiece as spm

from .utils import (
    ResponseConverter,
    DatasetFilename,
    WebsiteSampleData,
    build_response_from_sample_data,
    Filename,
)
from .stypes import ItemSample


class TokenizerFilename(Filename):
    pass


def extract_dataset_text_from_website_sampledata(
    dataset: Iterator[WebsiteSampleData],
    output_filename: Filename,
    response_converter: ResponseConverter,
):
    """
    Extracts text from a website sample dataset, suitable for usage in training tokenizer.
    The text is extracted using the specified ResponseConverter class, and saved into an output file
    for further tokenizer processing.
    """
    with output_filename.open("w") as output:
        for data in dataset:
            response = build_response_from_sample_data(data)
            text_pieces = response_converter.response_to_valid_text(response.text)
            print(" ".join(text_pieces), file=output)


def extract_dataset_text_from_item_sample(
    dataset_filename: DatasetFilename[ItemSample],
    output_filename: Filename,
):
    """
    Extracts text from an item sample dataset, suitable for usage in training tokenizer.
    """

    with output_filename.open("w") as output:
        for data in dataset_filename:
            print(data["markdown"], file=output)


def train_tokenizer(tokenizer_training_text: Filename, model_filename: TokenizerFilename):
    """
    Train a tokenizer using tokenizer_training_text file as input.
    Saves the model into the specified model_filename.
    """
    model_prefix = os.path.splitext(model_filename.basename)[0]
    spm.SentencePieceTrainer.train(f"--input={tokenizer_training_text} --model_prefix={model_prefix} --vocab_size=2000")
    shutil.move(f"{model_prefix}.model", model_filename)


def load_tokenizer_from_file(model_filename: TokenizerFilename) -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor()
    sp.load(model_filename)
    return sp
