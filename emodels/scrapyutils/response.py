import re
import logging
from typing import List, Optional

from scrapy.http import TextResponse

from emodels import html2text


MARKDOWN_LINK_RE = re.compile(r"\[(.+?)\]\((.+?)\s*(\".+\")?\)")
LINK_RSTRIP_RE = re.compile("(%20)+$")
LINK_LSTRIP_RE = re.compile("^(%20)+")
COMMENT_RE = re.compile(r"\s<!--.+?-->")
DEFAULT_SKIP_PREFIX = "[^a-zA-Z0-9$]*"
LOG = logging.getLogger(__name__)


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
