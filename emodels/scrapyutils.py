import re
import os
import json
import gzip
from typing import NewType, Dict, Tuple, List, Optional

from markdown import Markdown
from scrapy.loader import ItemLoader
from scrapy.http import TextResponse
from scrapy import Item

from emodels.config import EMODELS_ITEMS_DIR, EMODELS_SAVE_EXTRACT_ITEMS
from emodels.markdown.tables import TableExtension
from emodels import html2text


MARKDOWN_LINK_RE = re.compile(r"\[(.+?)\]\((.+?)\s*(\".+\")?\)")
LINK_RSTRIP_RE = re.compile("(%20)+$")
LINK_LSTRIP_RE = re.compile("^(%20)+")
COMMENT_RE = re.compile(r"\s<!--.+?-->")
DEFAULT_SKIP_PREFIX = "[^a-zA-Z0-9$]*"


class ExtractTextResponse(TextResponse):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._markdown = None
        self._markdown_ids = None
        self._markdown_classes = None

    @property
    def markdown(self):
        if self._markdown is None:
            h2t = html2text.HTML2Text(baseurl=self.url, bodywidth=0)
            self._markdown = self._clean_markdown(h2t.handle(self.text))
        return self._markdown

    @property
    def markdown_ids(self):
        if self._markdown_ids is None:
            h2t = html2text.HTML2Text(baseurl=self.url, bodywidth=0, ids=True)
            self._markdown_ids = self._clean_markdown(h2t.handle(self.text))
        return self._markdown_ids

    @property
    def markdown_classes(self):
        if self._markdown_classes is None:
            h2t = html2text.HTML2Text(baseurl=self.url, bodywidth=0, classes=True)
            self._markdown_classes = self._clean_markdown(h2t.handle(self.text))
        return self._markdown_classes

    def css_split(self, selector: str) -> List[TextResponse]:
        """Generate multiple responses from provided css selector"""
        result = []
        for html in self.css(selector).extract():
            new = self.replace(body=html.encode("utf-8"))
            result.append(new)
        return result

    def xpath_split(self, selector: str) -> List[TextResponse]:
        """Generate multiple responses from provided xpath selector"""
        result = []
        for html in self.xpath(selector).extract():
            new = self.replace(body=html.encode("utf-8"))
            result.append(new)
        return result

    def markdown_to_html(self, text: Optional[str] = None):
        text = text or self.markdown
        return Markdown(extensions=[TableExtension()]).convert(text).strip()

    @staticmethod
    def _clean_markdown(md: str):
        shrink = 0
        for m in MARKDOWN_LINK_RE.finditer(md):
            if m.groups()[1] is not None:
                start = m.start(2) - shrink
                end = m.end(2) - shrink
                link_orig = md[start:end]
                link = LINK_RSTRIP_RE.sub("", link_orig)
                link = LINK_LSTRIP_RE.sub("", link)
                md = md[:start] + link + md[end:]
                shrink += len(link_orig) - len(link)
        return md

    def text_re(
        self,
        reg: str = "(.+?)",
        tid: Optional[str] = None,
        flags: int = 0,
        skip_prefix: str = DEFAULT_SKIP_PREFIX,
        strict_tid: bool = False,
        optimize: bool = False,
    ):
        if tid and strict_tid:
            reg = f"(?:.*<!--.+-->)?{reg}"
        reg = f"{skip_prefix}{reg}"
        markdown = self.markdown
        if tid:
            if tid.startswith("#"):
                markdown = self.markdown_ids
            elif tid.startswith("."):
                tid = "\\" + tid
                markdown = self.markdown_classes
            reg += fr"\s+<!--{tid}-->"
        result = []
        for m in re.finditer(reg, markdown, flags):
            if m.groups():
                extracted = m.groups()[0]
                start = m.start(1)
                end = m.end(1)
            else:
                extracted = m.group()
                start = m.start()
                end = m.end()
            start += len(extracted) - len(extracted.lstrip())
            end -= len(extracted) - len(extracted.rstrip())
            extracted = extracted.strip()
            if extracted:
                if tid is not None:
                    new_extracted = COMMENT_RE.sub("", extracted).strip()
                    end -= len(extracted) - len(new_extracted)
                    extracted = new_extracted
                    accum = 0
                    for m in COMMENT_RE.finditer(markdown[:start]):
                        comment_len = m.end() - m.start()
                        accum += comment_len
                    start -= accum
                    end -= accum
                result.append((extracted, start, end))
                if optimize:
                    break
        return result


ExtractDict = NewType("ExtractDict", Dict[str, Tuple[int, int]])


class ExtractItemLoader(ItemLoader):

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        if not hasattr(cls, "savefile"):
            folder = os.path.join(EMODELS_ITEMS_DIR, obj.default_item_class.__name__)
            os.makedirs(folder, exist_ok=True)
            findex = len(os.listdir(folder))
            cls.savefile = os.path.join(folder, f"{findex}.jl.gz")
        return obj

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert "response" in self.context, '"response" is required.'
        if not isinstance(self.context["response"], ExtractTextResponse):
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
        *processors,
        **kw,
    ):
        extracted = self.context["response"].text_re(
            reg=reg, tid=tid, flags=flags, skip_prefix=skip_prefix, strict_tid=strict_tid, optimize=True
        )
        if extracted:
            t, s, e = extracted[0]
            if attr not in self.extract_indexes:
                self.extract_indexes[attr] = (s, e)
                self.add_value(attr, t, *processors, **kw)

    def add_text_re_as_html(self, attr: str, reg: str, flags: int = 0, *processors, **kw):
        extracted = self.context["response"].text_re(reg, flags=flags, optimize=True)
        if extracted:
            t, s, e = extracted[0]
            if attr not in self.extract_indexes:
                cleaned = self.context["response"].markdown_to_html(t)
                self.add_value(attr, cleaned, *processors, **kw)
                self.extract_indexes[attr] = (s, e)

    def load_item(self) -> Item:
        item = super().load_item()
        self._save_extract_sample()
        return item

    def _save_extract_sample(self):
        if EMODELS_SAVE_EXTRACT_ITEMS and self.extract_indexes:
            sample = {
                "indexes": self.extract_indexes,
                "markdown": self.context["response"].markdown,
            }
            with gzip.open(self.savefile, "at") as fz:
                print(json.dumps(sample), file=fz)
