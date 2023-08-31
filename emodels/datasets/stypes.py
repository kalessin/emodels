from typing import NewType, Dict, Tuple, TypedDict, Literal

from typing_extensions import NotRequired

ExtractDict = NewType("ExtractDict", Dict[str, Tuple[int, int]])
DatasetBucket = Literal["train", "validation", "test"]


class ItemSample(TypedDict):
    indexes: ExtractDict
    markdown: str
    dataset_bucket: NotRequired[DatasetBucket]
    source: NotRequired[str]
