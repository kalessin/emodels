"""
"""
import os
import abc
import gzip
import json
import logging
from collections import defaultdict
from random import random, randrange, shuffle
from typing import (
    List,
    Tuple,
    Protocol,
    cast,
    Dict,
    IO,
    TypedDict,
    Optional,
    Generic,
    TypeVar,
    Union,
    Generator,
    Mapping,
    Any,
)

from typing_extensions import Self
from scrapy.http import TextResponse
import lxml.html

from emodels.config import EMODELS_REPOSITORY, EMODELS_ITEMS_DIR
from emodels.scrapyutils.response import ExtractTextResponse
from emodels.datasets.stypes import ItemSample, DatasetBucket, ExtractDict


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

NO_TEXT_TAGS = ["script", "style", "noscript"]

# first number represents probability of being assigned to train dataset bucket, second number to test dataset bucket.
# if they sum up below 1, the remaining will be assigned to validation dataset bucket.
DEFAULT_DATASET_RATIO = (0.70, 0.30)


class Filename(str):
    """
    A class that represents a filename.

    This class provides a number of methods for working with filenames,
    including getting the basename, creating a local standard path of the file,
    and opening the file.

    It also inherits all string methods.

    Example:

    >>> filename = Filename("s3://path/to/file.txt")
    >>> filename.basename
    'file.txt'
    >>> filename.local("myproject")
    '/home/myuser/.datasets/myproject/file.txt'
    >>> with filename.open() as f:
    ...     contents = f.read()
    """

    @property
    def basename(self):
        return self.__class__(os.path.basename(self))

    def local(self, project_name: str):
        """
        Creates a local standard path to find a copy of the source file.
        """
        basedir = os.path.join(EMODELS_REPOSITORY, project_name)
        os.makedirs(basedir, exist_ok=True)
        return self.__class__(os.path.join(basedir, self.basename))

    def open(self, mode="rt"):
        return open(self, mode)

    @classmethod
    def local_by_name(cls, localname: str) -> Self:
        """
        Returns a Filename object by project/name.
        """
        project, name = localname.split("/")
        return cls(os.path.join(EMODELS_REPOSITORY, project, f"{name}.jl.gz"))

    def delete_local(self, project_name: str):
        os.remove(self.local(project_name))


# Type of dataset samples
E = TypeVar("E", bound=Mapping[str, Any])


class DatasetFilename(Generic[E], Filename):
    """
    A class that represents a dataset filename. Datasets are gzipped
    and have json lines format, They are iterable and has a method
    append() in order to add new samples.
    """

    _file: Union[None, IO]

    def __new__(cls, text):
        obj = super().__new__(cls, text)
        obj._file = None
        return obj

    def open(self, mode="rt"):
        return gzip.open(self, mode)

    def __iter__(self):
        return self

    def __next__(self) -> E:
        if self._file is None:
            self._file = self.open()
        line = next(self._file)
        return cast(E, json.loads(line))

    def append(self, data: E):
        assert not self._file, "Already opened."
        folder = os.path.dirname(self)
        os.makedirs(folder, exist_ok=True)
        with self.open("at") as fz:
            print(json.dumps(data), file=fz)

    def iter(self, **kwargs) -> Generator[E, None, None]:
        df = self.__class__(self)
        for sample in df:
            for key, val in kwargs.items():
                if sample.get(key, None) != val:
                    break
            else:
                yield sample


class WebsiteSampleData(TypedDict):
    url: str
    body: str
    status: int


class WebsiteDatasetFilename(DatasetFilename[WebsiteSampleData]):
    """
    Website Datasets contain a collection of WebsiteSampleData
    """

    ...


class ExtractDatasetFilename(DatasetFilename[ItemSample]):
    @classmethod
    def build_from_items(
        cls,
        localname: str,
        classes: Optional[Tuple[str]] = None,
        dataset_ratio: Tuple[float, float] = DEFAULT_DATASET_RATIO,
        max_samples_per_source: Optional[int] = None,
    ) -> Self:
        """
        Build a dataset dict from extracted items in user dataset folder.
        - name is a name for the dataset. It will determine the storing filename.
        - project is the name of the project the dataset belongs to. It will determine the storing filename.
        - If classes is a tuple of strings, select only the specified
        item subfolders.
        - dataset_ratio is the same for get_random_dataset() and determines how samples are distributed
          among train, test and validation buckets.
        """
        result: Self = cls.local_by_name(localname)
        if os.path.exists(result):
            raise ValueError(
                "Output file already exists. "
                f'open with {cls.__name__}.local_by_name("{localname}") or remove it for rebuilding'
            )
        for source in os.listdir(EMODELS_ITEMS_DIR):
            randomizer = DatasetBucketRandomizer(dataset_ratio)
            files = os.listdir(os.path.join(EMODELS_ITEMS_DIR, source))
            shuffle(files)
            for f in files:
                df: DatasetFilename[ItemSample] = DatasetFilename(os.path.join(EMODELS_ITEMS_DIR, source, f))
                selected: List[ItemSample] = []
                count = 0
                for sample in df:
                    count += 1
                    if max_samples_per_source is None or len(selected) < max_samples_per_source:
                        selected.append(sample)
                    else:
                        idx = randrange(count)
                        if idx < max_samples_per_source:
                            selected = selected[:idx] + selected[idx + 1:]
                dataset_bucket = randomizer.get_random_dataset()
                LOGGER.info(f"Bucket {dataset_bucket} assigned to samples from source {source}/{f}.")
                for sample in selected:
                    sample["dataset_bucket"] = dataset_bucket
                    sample["source"] = source
                    randomizer.inc_assigned(dataset_bucket, len(sample["indexes"]))
                    result.append(sample)
        return result

    def count_samples_by_bucket(self) -> Dict[str, Dict[DatasetBucket, int]]:
        count: Dict[str, Dict[DatasetBucket, int]] = defaultdict(lambda: defaultdict(int))
        for sample in self.__class__(self):
            for _ in sample["indexes"].keys():
                count[sample["source"]][sample["dataset_bucket"]] += 1
        return dict(count)

    def count_samples_by_attribute(self) -> Dict[str, Dict[DatasetBucket, int]]:
        count: Dict[str, Dict[DatasetBucket, int]] = defaultdict(lambda: defaultdict(int))
        for sample in self.__class__(self):
            for key in sample["indexes"].keys():
                count[sample["source"]][key] += 1
        return dict(count)

    def convert_attributes(self, target_name: str, source_map_dict: Dict[str, Dict[str, str]]) -> Self:
        result: Self = self.__class__.local_by_name(target_name)
        for sample in self.__class__(self):
            if sample["source"] in source_map_dict:
                new_indexes: ExtractDict = ExtractDict({})
                for attr, val in sample["indexes"].items():
                    new_attr = source_map_dict[sample["source"]].get(attr, attr)
                    new_indexes[new_attr] = val
                sample["indexes"] = new_indexes
            result.append(sample)
        return result


class DatasetBucketRandomizer:
    def __init__(self, dataset_ratio: Tuple[float, float] = DEFAULT_DATASET_RATIO):
        assert len(dataset_ratio) == 2, "Invalid dataset_ratio len: must be 2."
        self.__ratios: Tuple[float, float, float] = dataset_ratio + (1 - sum(dataset_ratio),)
        self.__assigned: Tuple[int, int, int] = (0, 0, 0)
        self.__all_buckets: List[DatasetBucket] = ["train", "test", "validation"]

    def _get_current_ratios(self) -> Tuple[float, float, float]:
        total = sum(self.__assigned)
        if total == 0:
            return 0, 0, 0
        return cast(Tuple[float, float, float], tuple(v / total for v in self.__assigned))

    def inc_assigned(self, bucket: DatasetBucket, inc: int = 1):
        if bucket == "train":
            self.__assigned = (self.__assigned[0] + inc, self.__assigned[1], self.__assigned[2])
        elif bucket == "test":
            self.__assigned = (self.__assigned[0], self.__assigned[1] + inc, self.__assigned[2])
        else:
            self.__assigned = (self.__assigned[0], self.__assigned[1], self.__assigned[2] + inc)

    def _get_random_dataset(self) -> DatasetBucket:
        """
        - dataset_ratio: a 2-tuple of floats. The first element is the probability to yield "train",
          and the second element the probability to yield "test". If they sum below 1, the remaining
          is the probability to yield "validation".
        """
        r = random()
        if r < self.__ratios[0]:
            return "train"
        if r < sum(self.__ratios[:2]):
            return "test"
        return "validation"

    def get_random_dataset(self) -> DatasetBucket:
        below = [
            k[0]
            for k in zip(self.__all_buckets, [cr < t for cr, t in zip(self._get_current_ratios(), self.__ratios)])
            if k[1]
        ]

        zero = [
            k[0]
            for k in zip(self.__all_buckets, [a == 0 and t > 0 for a, t in zip(self.__assigned, self.__ratios)])
            if k[1]
        ]

        if zero:
            if len(zero) == 1:
                return zero[0]
            while True:
                bucket = self._get_random_dataset()
                if bucket in zero:
                    return bucket

        if below:
            if len(below) == 1:
                return below[0]
            while True:
                bucket = self._get_random_dataset()
                if bucket in below:
                    return bucket

        return self._get_random_dataset()


class ResponseConverter(Protocol):
    @abc.abstractmethod
    def response_to_valid_text(self, body: str) -> List[str]:
        """
        Converts html source into a list of text pieces.
        """
        ...


class lxmlResponseConverter(ResponseConverter):
    def __init__(self):
        self.htmlparser = lxml.html.HTMLParser()

    def response_to_valid_text(self, body: str) -> List[str]:
        """
        Returns the list of all text words extracted from an html body
        """
        texts: List[str] = []
        body = body.strip()
        if not body:
            return texts
        try:
            tree = lxml.html.document_fromstring(body.encode("utf8"), parser=self.htmlparser)
        except lxml.html.etree.ParserError:
            LOGGER.error(f"Error parsing {body[:100]}...")
            return texts
        except UnicodeEncodeError:
            LOGGER.error(f"Unicode error encoding {body[:100]}")
            return texts
        for _, element in lxml.html.etree.iterwalk(tree, events=("start",)):
            if not isinstance(element.tag, str):
                continue
            if element.tag in NO_TEXT_TAGS:
                continue
            if element.text is None:
                continue
            text = element.text.strip()
            if text:
                texts.append(text)
        return texts


def build_response_from_sample_data(sampledata: WebsiteSampleData) -> ExtractTextResponse:
    response = ExtractTextResponse(
        url=sampledata["url"],
        body=sampledata["body"].encode("utf8"),
        status=sampledata["status"],
    )
    return response


def build_sample_data_from_response(response: TextResponse) -> WebsiteSampleData:
    sampledata: WebsiteSampleData = {
        "url": response.url,
        "body": response.text,
        "status": response.status,
    }
    return sampledata


def save_sample_data_from_response(response: TextResponse, filename: Union[str, WebsiteDatasetFilename]):
    sampledata = build_sample_data_from_response(response)
    if not isinstance(filename, WebsiteDatasetFilename):
        filename = WebsiteDatasetFilename(filename)
    filename.append(sampledata)
