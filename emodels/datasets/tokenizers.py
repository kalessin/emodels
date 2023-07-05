"""
Example of training:

> from lexisnexis.utils.website_checker import WebsiiteCheckerHelper

download samples:

> WebsiiteCheckerHelper.download_labelled_samples("samples.jl")

generate text for tokenizer training:

> from lexisnexis.utils.website_checker.tokenizers import extract_dataset_text
> extract_dataset_text("samples.jl", "tokenizer_training_text.txt")

"""
import os
import shutil

import sentencepiece as spm

from .utils import (
    DatasetFilename,
    ResponseConverter,
    build_response_from_sample_data,
    Filename,
)


class TokenizerFilename(Filename):
    pass


def extract_dataset_text(
    dataset_filename: DatasetFilename, output_filename: Filename, response_converter_cls: type[ResponseConverter]
):
    """
    Extracts text from a dataset, suitable for usage in training tokenizer.
    The text is extracted using the specified ResponseConverter class, and saved into an output file
    for further tokenizer processing.
    """
    converter = response_converter_cls()
    with output_filename.open("w") as output:
        for data in dataset_filename:
            response = build_response_from_sample_data(data)
            text_pieces = converter.response_to_valid_text(response.text)
            print(" ".join(text_pieces), file=output)


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
