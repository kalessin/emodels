import re
from collections import defaultdict
from typing import Dict, Tuple, NewType, Literal, List

import dateparser

from emodels.scrapyutils.response import ExtractTextResponse

Text = NewType("Text", str)
Keyword = NewType("Keyword", str)
# convenient way to represent a match with its position in the text
# Match is a tuple of (matched text, start position, end position, extracted value).
# It is build from a re.Match object, according to re_match_to_match function below.
Match = NewType("Match", Tuple[Text, int, int, Text])


def re_match_to_match(m: re.Match) -> Match:
    full_group = Text(m.group())
    value_group = Text(m.groups()[-1] if m.groups() else "")
    return Match((full_group, m.start(), m.end(), value_group))


Constraints = NewType("Constraints", Dict[Keyword, re.Pattern | Literal["date_type", "url_type"]])
Result = NewType("Result", Dict[Keyword, Text])
URL_RE = re.compile(r"^(https?://.+?)|(<https?://.+?>)|(\[.+\]\(https?://.+\))|(www\..+\..+)")


def apply_constraints(result: Dict[Keyword, Text], constraints: Constraints) -> bool:
    """Remove fields from result that don't match the given constraints.
    constraint is either a regex pattern or any of special keywords as defined in Constraints type:
    date_type: matches if dateparser accepts the field value.
    url_type: matches if the field value matches a url regex.
    """
    was_updated = False
    for k, pattern in constraints.items():
        if isinstance(pattern, str):
            if pattern == "date_type":
                if result.get(k) and dateparser.parse(result[k]) is None:
                    result.pop(k)
                    was_updated = True
                continue
            if pattern == "url_type":
                pattern = URL_RE
        if result.get(k) and pattern.search(result[k]) is None:
            result.pop(k)
            was_updated = True
    return was_updated


def parse_additional_regexes(
    additional_regexes: Dict[Keyword, Tuple[str | Tuple[str | None, str], ...]] | None,
    response: ExtractTextResponse,
    tiles_mode: bool = False,
) -> Dict[Keyword, List[Match]]:
    """
    Get additional regexes to the response and update the result dict with any matches found.
    - additional_regexes is a dict where keys are field names and values are tuples of regex patterns
      or tuples of (regex pattern, tid), according to the argument accepted by response.text_re method.
      If a regex pattern is None, it will only apply the tid-based extraction without any regex filtering.
    - response is the ExtractTextResponse object to apply the regexes on.
    - tiles_mode - If False, it will return the first match for each regex. If True, it will return all
      matches for each regex.
    """
    matches: Dict[Keyword, List[Match]] = defaultdict(list)
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
            all_extracted = response.text_re(regex, tid=tid, flags=flags)
            for extracted in all_extracted:
                matches[field].append(Match((Text(extracted[0]), extracted[1], extracted[2], Text(extracted[0]))))
                if not tiles_mode:
                    break
    return matches


def apply_additional_regexes(
    additional_regexes: Dict[Keyword, Tuple[str | Tuple[str | None, str], ...]] | None,
    result: Result,
    response: ExtractTextResponse,
) -> None:
    """
    Apply additional regexes to the response and update the result dict with any matches found.
    """
    for keyword, matches in parse_additional_regexes(additional_regexes, response).items():
        result[keyword] = matches[0][3]  # the extracted value is in the 4th position of the Match tuple

    if Keyword("url") not in result:
        result[Keyword("url")] = Text(response.url)
