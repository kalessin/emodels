"""
"""
import logging
from abc import abstractmethod
from typing import Generator, List, TypedDict, Any, Protocol, Tuple

import joblib
from scrapy.http import HtmlResponse
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sentencepiece import SentencePieceProcessor
from shub_workflow.deliver.futils import FSHelper
from shub_workflow.utils import get_project_settings

from emodels.datasets.utils import (
    build_response_from_sample_data,
    ResponseConverter,
    Filename,
    DatasetFilename,
)
from emodels.datasets.tokenizers import (
    extract_dataset_text,
    train_tokenizer,
    load_tokenizer_from_file,
    TokenizerFilename,
)
from emodels.datasets.utils import WebsiteSampleData


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class ModelFilename(Filename):
    pass


class VectorizerFilename(Filename):
    pass


class DatasetsDict(TypedDict):
    X_train: pd.DataFrame
    X_test: pd.Series
    Y_train: pd.DataFrame
    Y_test: pd.Series


class ModelWithDataset(Protocol):
    FSHELPER: FSHelper = None
    datasets: DatasetsDict | None = None

    dataset_repository: DatasetFilename
    features: Tuple[str, ...]
    target_label: str
    project: str

    def __init__(self):
        raise ValueError("Class must not be instantiated.")

    @classmethod
    @abstractmethod
    def build_fshelper(cls, settings) -> FSHelper:
        ...

    @classmethod
    def _fshelper(cls) -> FSHelper:
        if cls.FSHELPER is None:
            settings = get_project_settings()
            cls.FSHELPER = cls.build_fshelper(settings)
        return cls.FSHELPER

    @classmethod
    def get_dataset_from_filename(cls, filename: DatasetFilename) -> DatasetsDict:
        df = pd.read_json(filename, lines=True, compression="gzip")
        df = df[~df[cls.target_label].isnull()]

        df_train = df[df.dataset_bucket == "train"].drop("dataset_bucket", axis=1)
        df_test = df[df.dataset_bucket == "test"].drop("dataset_bucket", axis=1)

        df_X_train = df_train[list(cls.features)]
        df_Y_train = df_train[cls.target_label]

        df_X_test = df_test[list(cls.features)]
        df_Y_test = df_test[cls.target_label]

        return {
            "X_train": df_X_train,
            "X_test": df_X_test,
            "Y_train": df_Y_train,
            "Y_test": df_Y_test,
        }

    @classmethod
    def load_dataset(cls) -> DatasetsDict:
        dataset_local: DatasetFilename = DatasetFilename(cls.dataset_repository).local(cls.project)

        if cls._fshelper().exists(dataset_local):
            LOGGER.info(f"Found local copy of datasets {dataset_local}.")
        elif cls._fshelper().exists(cls.dataset_repository):
            LOGGER.info("Downloading datasets...")
            cls._fshelper().download_file(cls.dataset_repository, dataset_local)
        else:
            LOGGER.info("Generating datasets...")
            cls.download_labelled_samples(dataset_local)
            cls._fshelper().upload_file(dataset_local, cls.dataset_repository)
        return cls.get_dataset_from_filename(dataset_local)

    @classmethod
    def get_dataset(cls) -> DatasetsDict:
        if cls.datasets is None:
            cls.datasets = cls.load_dataset()
        return cls.datasets

    @classmethod
    def reset(cls):
        pass

    @classmethod
    def delete_model_files(cls, repository: Filename):
        for fname in (repository, repository.local(cls.project)):
            try:
                cls._fshelper().rm_file(fname)
                LOGGER.info(f"Deleted {fname}")
            except FileNotFoundError:
                pass

    @classmethod
    @abstractmethod
    def download_labelled_samples(cls, target: Filename):
        ...


class ModelWithTokenizer(ModelWithDataset, Protocol):
    tokenizer_repository: TokenizerFilename
    converter_class: type[ResponseConverter]

    tokenizer: SentencePieceProcessor | None = None

    @classmethod
    def load_tokenizer(cls) -> SentencePieceProcessor:
        """
        - If exists a local copy of the tokenizer in the local datasets folder, load it
        - If not, and exist a repository copy of the tokenizer, download to its local copy into the local
          datasets folder, and load it.
        - If not, train the tokenizer using the dataset text, store it into the local dataset folder,
          upload the trained model to the repository, and load it.
        """
        tokenizer_local: TokenizerFilename = TokenizerFilename(cls.tokenizer_repository).local(cls.project)
        if cls._fshelper().exists(tokenizer_local):
            LOGGER.info(f"Found local copy of tokenizer model {tokenizer_local}.")
        elif cls._fshelper().exists(cls.tokenizer_repository):
            LOGGER.info("Downloading tokenizer model...")
            cls._fshelper().download_file(cls.tokenizer_repository, tokenizer_local)
        else:
            LOGGER.info("Training tokenizer model...")
            training_text_filename = Filename("sptokenizer_training_text.txt").local(cls.project)
            cls.load_dataset()
            extract_dataset_text(
                DatasetFilename(cls.dataset_repository).local(cls.project), training_text_filename, cls.converter_class
            )
            train_tokenizer(training_text_filename, tokenizer_local)
            cls._fshelper().upload_file(tokenizer_local, cls.tokenizer_repository)
        return load_tokenizer_from_file(tokenizer_local)

    @classmethod
    def get_tokenizer(cls) -> SentencePieceProcessor:
        if cls.tokenizer is None:
            cls.tokenizer = cls.load_tokenizer()
        return cls.tokenizer

    @classmethod
    def reset(cls):
        cls.delete_model_files(cls.tokenizer_repository)
        cls.tokenizer = None
        super().reset()

    @classmethod
    def get_features_from_body(cls, converter: ResponseConverter, body: str) -> str:
        tokenizer = cls.get_tokenizer()
        text: str = " ".join(converter.response_to_valid_text(body))
        tokens: List[str] = tokenizer.encode_as_pieces(text)
        return " ".join(tokens)

    @classmethod
    def get_dataset_features(cls, dataset_bucket: pd.DataFrame) -> Generator[str, None, None]:
        converter = cls.converter_class()
        for _, row in dataset_bucket.iterrows():
            yield cls.get_features_from_body(converter, row["body"])


class ModelWithVectorizer(ModelWithTokenizer, ModelWithDataset, Protocol):
    vectorizer_repository: VectorizerFilename
    vectorizer: TfidfVectorizer | None = None

    @classmethod
    def load_vectorizer(cls) -> TfidfVectorizer:
        vectorizer_local: VectorizerFilename = VectorizerFilename(cls.vectorizer_repository).local(cls.project)

        if cls._fshelper().exists(vectorizer_local):
            LOGGER.info(f"Found local copy of vectorizer model {vectorizer_local}.")
        elif cls._fshelper().exists(cls.vectorizer_repository):
            LOGGER.info("Downloading vectorizer model...")
            cls._fshelper().download_file(cls.vectorizer_repository, vectorizer_local)
        else:
            LOGGER.info("Training vectorizer...")
            vectorizer = TfidfVectorizer(min_df=10, max_df=0.7, ngram_range=(1, 3))
            datasets = cls.load_dataset()
            X_train = datasets["X_train"]
            vectorizer.fit(cls.get_dataset_features(X_train))
            joblib.dump(vectorizer, vectorizer_local)
            cls._fshelper().upload_file(vectorizer_local, cls.vectorizer_repository)
        return joblib.load(vectorizer_local)

    @classmethod
    def get_vectorizer(cls) -> TfidfVectorizer:
        if cls.vectorizer is None:
            cls.vectorizer = cls.load_vectorizer()
        return cls.vectorizer

    @classmethod
    def reset(cls):
        cls.delete_model_files(cls.vectorizer_repository)
        cls.vectorizer = None
        super().reset()


class TrainableModel(ModelWithVectorizer, ModelWithTokenizer, ModelWithDataset, Protocol):
    model_repository: ModelFilename
    model: Any | None = None

    @classmethod
    def load_trained_model(cls) -> Any:
        model_local: ModelFilename = ModelFilename(cls.model_repository).local(cls.project)
        if cls._fshelper().exists(model_local):
            LOGGER.info(f"Found local copy of model {model_local}.")
        elif cls._fshelper().exists(cls.model_repository):
            LOGGER.info("Downloading model...")
            cls._fshelper().download_file(cls.model_repository, model_local)
        else:
            LOGGER.info("Training classifier...")
            model = cls.train()
            joblib.dump(model, model_local)
            cls._fshelper().upload_file(model_local, cls.model_repository)
        return joblib.load(model_local)

    @classmethod
    def get_trained_model(cls) -> Any:
        if cls.model is None:
            cls.model = cls.load_trained_model()
        return cls.model

    @classmethod
    @abstractmethod
    def train(cls):
        ...

    @classmethod
    def reset(cls):
        """Remove datasets and model files, so a retrain will be forced."""
        cls.delete_model_files(cls.model_repository)
        super().reset()


class ClassifierModel(TrainableModel):

    @classmethod
    @abstractmethod
    def classify_response(cls, response: HtmlResponse) -> bool:
        ...

    @classmethod
    def classify_from_row(cls, row: WebsiteSampleData) -> bool:
        response = build_response_from_sample_data(row)
        return cls.classify_response(response)

    @classmethod
    def predict(cls, df: pd.DataFrame) -> pd.Series:
        return df.apply(cls.classify_from_row, axis=1)

    @classmethod
    def evaluate(cls):
        datasets = cls.get_dataset()
        predicted = cls.predict(datasets["X_train"])
        y_train = datasets["Y_train"]

        def _stat(score_func, target, predicted):
            return str(round(score_func(target, predicted) * 100, 2)) + "%"

        def _print_confusion_matrix(target, predicted):
            cm = confusion_matrix(target, predicted)
            result = f"TN={cm[0, 0]} FP={cm[0, 1]}\n"
            result += f" FN={cm[1, 0]} TP={cm[1, 1]}"
            return result

        print("Train set scores")
        print("----------------")
        print("Recall:", _stat(recall_score, y_train, predicted))
        print("Precision:", _stat(precision_score, y_train, predicted))
        print("Accuracy:", _stat(accuracy_score, y_train, predicted))
        print("Roc Auc:", _stat(roc_auc_score, y_train, predicted))
        print("Confusion matrix:\n", _print_confusion_matrix(y_train, predicted))
        print()

        predicted = cls.predict(datasets["X_test"])
        y_test = datasets["Y_test"]

        print("Test set scores")
        print("---------------")
        print("Recall:", _stat(recall_score, y_test, predicted))
        print("Precision:", _stat(precision_score, y_test, predicted))
        print("Accuracy:", _stat(accuracy_score, y_test, predicted))
        print("Roc Auc:", _stat(roc_auc_score, y_test, predicted))
        print("Confusion matrix:\n", _print_confusion_matrix(y_test, predicted))


class SVMModel(ClassifierModel):
    gamma = 0.4
    C = 10

    @classmethod
    def train(cls) -> None:
        vectorizer = cls.get_vectorizer()

        LOGGER.info("Training SVM classifier...")
        model = SVC(kernel="rbf", C=cls.C, gamma=cls.gamma)
        datasets = cls.load_dataset()
        vfeatures = vectorizer.transform(cls.get_dataset_features(datasets["X_train"]))
        model.fit(vfeatures, datasets["Y_train"])
        return model

    @classmethod
    def classify_response(cls, response: HtmlResponse) -> bool:
        """ """

        vectorizer = cls.get_vectorizer()
        model = cls.get_trained_model()

        X_features = [cls.get_features_from_body(cls.converter_class(), response.text)]
        X_transformed = vectorizer.transform(X_features)
        return model.predict(X_transformed)[0]
