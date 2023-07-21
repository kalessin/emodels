import os
import logging
from typing import NewType, Dict, Tuple, Optional

from scrapy.loader import ItemLoader
from scrapy.http import TextResponse
from scrapy import Item

from emodels.config import EMODELS_ITEMS_DIR, EMODELS_SAVE_EXTRACT_ITEMS
from emodels.datasets.utils import DatasetFilename
from emodels.scrapyutils.response import ExtractTextResponse, DEFAULT_SKIP_PREFIX


LOG = logging.getLogger(__name__)


ExtractDict = NewType("ExtractDict", Dict[str, Tuple[int, int]])


class ExtractItemLoader(ItemLoader):

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        if not hasattr(cls, "savefile"):
            folder = os.path.join(EMODELS_ITEMS_DIR, obj.default_item_class.__name__)
            os.makedirs(folder, exist_ok=True)
            findex = len(os.listdir(folder))
            cls.savefile = DatasetFilename(os.path.join(folder, f"{findex}.jl.gz"))
        return obj

    def _check_valid_response(self):
        return isinstance(self.context.get("response"), TextResponse)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._check_valid_response() and not isinstance(self.context["response"], ExtractTextResponse):
            self.context["response"] = self.context["response"].replace(cls=ExtractTextResponse)
        self.extract_indexes: ExtractDict = ExtractDict({})

    def add_text_re(
        self,
        attr: str,
        reg: str = "(.+?)",
        tid: Optional[str] = None,
        flags: int = 0,
        skip_prefix: str = DEFAULT_SKIP_PREFIX,
        strict_tid: bool = False,
        idx: int = 0,
        *processors,
        **kw,
    ):
        if not self._check_valid_response():
            raise ValueError("context response type is not a valid TextResponse.")
        extracted = self.context["response"].text_re(
            reg=reg, tid=tid, flags=flags, skip_prefix=skip_prefix, strict_tid=strict_tid, idx=idx, optimize=True
        )
        if extracted:
            t, s, e = extracted[0]
            if attr not in self.extract_indexes:
                self.extract_indexes[attr] = (s, e)
                self.add_value(attr, t, *processors, **kw)

    def load_item(self) -> Item:
        item = super().load_item()
        self._save_extract_sample()
        return item

    def _save_extract_sample(self):
        if EMODELS_SAVE_EXTRACT_ITEMS and self.extract_indexes:
            self.savefile.append({
                "indexes": self.extract_indexes,
                "markdown": self.context["response"].markdown,
            })
