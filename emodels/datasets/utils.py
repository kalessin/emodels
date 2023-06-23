"""
"""
import os
import abc
import gzip
import json
import logging
from random import random
from typing import List, Literal, Tuple, Protocol, cast, Dict, Any, IO

from typing_extensions import TypedDict
from scrapy.http import TextResponse
import lxml.html

from emodels.config import EMODELS_DIR


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

NO_TEXT_TAGS = ["script", "style", "noscript"]

DatasetBucket = Literal["train", "validation", "test"]


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
        basedir = os.path.join(EMODELS_DIR, project_name)
        os.makedirs(basedir, exist_ok=True)
        return self.__class__(os.path.join(basedir, self.basename))

    def open(self, mode="rt"):
        return open(self, mode)


class DatasetFilename(Filename):
    """
    A class that represents a dataset filename. Datasets are gzipped
    and have json lines format
    """
    _file: None | IO

    def __new__(cls, text):
        obj = super().__new__(cls, text)
        obj._file = None
        return obj

    def open(self, mode="rt"):
        return gzip.open(self, mode)

    def __iter__(self):
        return self

    def __next__(self):
        if self._file is None:
            self._file = self.open()
        line = next(self._file)
        return json.loads(line)

    def append(self, data: Dict[str, Any]):
        assert not self._file, "Already opened."
        with self.open("at") as fz:
            print(json.dumps(data), file=fz)


class WebsiteSampleData(TypedDict):
    url: str
    body: str
    status: int


class WebsiteDatasetFilename(DatasetFilename):
    """
    Website Datasets contain a collection of WebsiteSampleData
    """
    def __next__(self) -> WebsiteSampleData:
        return cast(WebsiteSampleData, super().__next__())


def get_random_dataset(dataset_ratio: Tuple[float, ...] = (0.6, 0.4)) -> DatasetBucket:
    assert len(dataset_ratio) == 2, "Invalid dataset_ratio len: must be 2."
    r = random()
    if r < dataset_ratio[0]:
        return "train"
    if r < sum(dataset_ratio):
        return "test"
    return "validation"


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


def build_response_from_sample_data(sampledata: WebsiteSampleData) -> TextResponse:
    response = TextResponse(
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
