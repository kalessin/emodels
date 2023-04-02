import re
import os
import json
import gzip
from typing import NewType, Dict, Tuple, List, Optional

from markdown2 import Markdown

from scrapy.loader import ItemLoader
from scrapy.http import TextResponse
from scrapy import Item

from emodels.config import EMODELS_ITEMS_DIR, EMODELS_SAVE_EXTRACT_ITEMS
from emodels import html2text


MARKDOWN_LINK_RE = re.compile(r"\[(.+?)\]\((.+?)\s*(\".+\")?\)")
LINK_RSTRIP_RE = re.compile("(%20)+$")
LINK_LSTRIP_RE = re.compile("^(%20)+")
COMMENT_RE = re.compile("<!--.+?-->")


class ExtractTextResponse(TextResponse):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._markdown = None

    @property
    def markdown(self):
        if self._markdown is None:
            h2t = html2text.HTML2Text(baseurl=self.url, bodywidth=0)
            self._markdown = self._clean_markdown(h2t.handle(self.text))
        return self._markdown

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

    def text_re(self, reg: str, flags: int = 0):
        result = []
        for m in re.finditer(reg, self.markdown, flags):
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
                new_extracted = COMMENT_RE.sub("", extracted)
                end -= len(extracted) - len(new_extracted)
                result.append((new_extracted, start, end))
        return result

    def text_id(self, tid: str, skip_prefix: Optional[str]=None):
        if skip_prefix is None:
            skip_prefix = "[^a-zA-Z0-9$]*"
        reg = f"{skip_prefix}(.+?)<!--{tid}-->"
        return self.text_re(reg)


ExtractDict = NewType("ExtractDict", Dict[str, Tuple[int, int]])


class ExtractItemLoader(ItemLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert "response" in self.context, '"response" is required.'
        if not isinstance(self.context["response"], ExtractTextResponse):
            self.context["response"] = self.context["response"].replace(cls=ExtractTextResponse)
        self.extract_indexes: ExtractDict = ExtractDict({})
        self._mconverter = Markdown()

    def add_text_re(self, attr: str, reg: str, flags: int = 0, *processors, **kw):
        extracted = self.context["response"].text_re(reg, flags)
        if extracted:
            t, s, e = extracted[0]
            if attr not in self.extract_indexes:
                self.extract_indexes[attr] = (s, e)
                self.add_value(attr, t, *processors, **kw)

    def add_text_id(self, attr: str, tid: str, *processors, **kw):
        skip_prefix = kw.pop("skip_prefix", None)
        extracted = self.context["response"].text_id(tid, skip_prefix)
        if extracted:
            t, s, e = extracted[0]
            if attr not in self.extract_indexes:
                self.extract_indexes[attr] = (s, e)
                self.add_value(attr, t, *processors, **kw)

    def add_text_re_as_html(self, attr: str, reg: str, flags: int = 0, *processors, **kw):
        extracted = self.context["response"].text_re(reg, flags)
        if extracted:
            t, s, e = extracted[0]
            if attr not in self.extract_indexes:
                cleaned = self._mconverter.convert(t).strip()
                self.add_value(attr, cleaned, *processors, **kw)
                self.extract_indexes[attr] = (s, e)

    def load_item(self) -> Item:
        item = super().load_item()
        self._save_extract_sample(item.__class__.__name__)
        return item

    def _save_extract_sample(self, clsname: str):
        if EMODELS_SAVE_EXTRACT_ITEMS and self.extract_indexes:
            markdown = self.context["response"].markdown
            new_indexes = ExtractDict({})
            sorted_indexes: List[Tuple[int, int, str]] = [(s, e, attr) for attr, (s, e) in sorted(self.extract_indexes.items(), key=lambda x: x[1][0])]
            accum = 0
            for m in COMMENT_RE.finditer(self.context["response"].markdown):
                comment_len = m.end() - m.start()
                markdown = markdown[:m.start() - accum] + markdown[m.end() - accum:]
                while sorted_indexes and m.start() > sorted_indexes[0][0]:
                    s, e, attr = sorted_indexes.pop(0)
                    new_indexes[attr] = (s - accum, e - accum)
                accum += comment_len

            while sorted_indexes:
                s, e, attr = sorted_indexes.pop(0)
                new_indexes[attr] = (s - accum, e - accum)

            sample = {
                "indexes": new_indexes,
                "markdown": markdown,
            }
            with gzip.open(os.path.join(EMODELS_ITEMS_DIR, f"{clsname}.jl.gz"), "at") as fz:
                print(json.dumps(sample), file=fz)
