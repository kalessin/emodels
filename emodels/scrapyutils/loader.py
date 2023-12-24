import os
import re
import logging
from typing import Optional

from scrapy.loader import ItemLoader
from scrapy.http import TextResponse
from scrapy import Item

from emodels.config import EMODELS_ITEMS_DIR, EMODELS_SAVE_EXTRACT_ITEMS
from emodels.datasets.utils import DatasetFilename
from emodels.scrapyutils.response import ExtractTextResponse, DEFAULT_SKIP_PREFIX
from emodels.datasets.stypes import ExtractDict, ItemSample


LOG = logging.getLogger(__name__)


class ExtractItemLoader(ItemLoader):
    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        if not hasattr(cls, "savefile"):
            folder = os.path.join(EMODELS_ITEMS_DIR, obj.default_item_class.__name__)
            os.makedirs(folder, exist_ok=True)
            findex = len(os.listdir(folder))
            cls.savefile: DatasetFilename[ItemSample] = DatasetFilename(os.path.join(folder, f"{findex}.jl.gz"))
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
        idx: int = 0,
        *processors,
        **kw,
    ):
        """
        attr - item attribute where selector extracted value will be assigned to.
        reg - Optional regular expression (it is optional because you can also use tid alone)
        tid - Optional css id or class specification (start with either # or .). When this parameter is present,
              regular expression match will be restricted to the text region with the specified id or class. Note:
              when the tid string starts with #, it is also able to match the itemprop attribute, not only the id.
        flags - Optional regular expression flag
        skip_prefix - This prefix is added to the provided regular expression in order to skip it from the result.
              The default one is any non alphanumeric character at begining of the line and in most cases
              you will use this value. Provided for convenience, in order to avoid to repeat it frequently
              in the regular expression parameter, making it more natural.
        idx - Regex selectors only return a single match, and by default it is the first one (idx=0). If you want
              instead to extract a different match, set the appropiate index with this parameter.
        *processors - Extraction processors passed to the method (same as in usual loaders)
        **kw - Additional extract parameters passed to the method (same as in usual loaders)
        """
        if not self._check_valid_response():
            raise ValueError("context response type is not a valid TextResponse.")
        extracted = self.context["response"].text_re(
            reg=reg, tid=tid, flags=flags, skip_prefix=skip_prefix, idx=idx, optimize=True
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
            self.savefile.append(
                {
                    "indexes": self.extract_indexes,
                    "markdown": self.context["response"].markdown,
                }
            )

    def _add_extraction_from_values(self, attr: str):
        for value in self.get_collected_values(attr):
            value = value.strip()
            value = re.sub(r"\s+", " ", value)
            value = re.sub(r"\s+,", ",", value)
            start = self.context["response"].markdown.find(value)
            if start > -1 and len(value):
                self.extract_indexes[attr] = (start, start + len(value))

    def add_xpath(self, field_name, xpath, *processors, **kw):
        super().add_xpath(field_name, xpath, *processors, **kw)
        self._add_extraction_from_values(field_name)

    def add_css(self, field_name, css, *processors, **kw):
        super().add_css(field_name, css, *processors, **kw)
        self._add_extraction_from_values(field_name)
