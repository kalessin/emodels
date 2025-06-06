import re
from typing import Dict, Tuple, NewType, Literal

import dateparser

from emodels.scrapyutils.response import ExtractTextResponse


Constraints = NewType("Constraints", Dict[str, re.Pattern | Literal["date_type"]])


def apply_constraints(result: Dict[str, str], constraints: Constraints) -> bool:
    was_updated = False
    for k, pattern in constraints.items():
        if isinstance(pattern, str):
            if pattern == "date_type":
                if result.get(k) and dateparser.parse(result[k]) is None:
                    result.pop(k)
                    was_updated = True
        elif result.get(k) and pattern.search(result[k]) is None:
            result.pop(k)
            was_updated = True
    return was_updated


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
