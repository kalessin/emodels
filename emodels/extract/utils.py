import re
from typing import Dict, Tuple

from emodels.scrapyutils.response import ExtractTextResponse


def apply_additional_regexes(
    additional_regexes: Dict[str, Tuple[str | Tuple[str | None, str], ...]] | None,
    result: Dict[str, str],
    response: ExtractTextResponse,
):
    for field, regexes in (additional_regexes or {}).items():
        assert isinstance(regexes, (list, tuple)), "additional_regexes values must be of type list."
        for regex_tid in regexes:
            tid = None
            if isinstance(regex_tid, (tuple, list)):
                regex, tid = regex_tid
            else:
                regex = regex_tid
            if regex is None:
                regex = "(.+?)"
            flags = re.M | re.I if regex.startswith("^") else re.I
            extracted = response.text_re(regex, tid=tid, flags=flags)
            if extracted:
                result[field] = extracted[0][0]
                break
    if "url" not in result:
        result["url"] = response.url
