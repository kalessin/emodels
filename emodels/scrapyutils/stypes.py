from typing import NewType, Dict, Tuple, TypedDict

ExtractDict = NewType("ExtractDict", Dict[str, Tuple[int, int]])


class ItemSample(TypedDict):
    indexes: ExtractDict
    markdown: str
