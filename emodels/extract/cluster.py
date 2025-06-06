"""
Cluster extraction algorithm
"""

import re
from pprint import pformat
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

from sklearn.cluster import KMeans
import numpy as np

from emodels.extract.utils import Constraints, apply_constraints


def extract_by_keywords(  # noqa: C901
    markdown: str,
    keywords: Tuple[str, ...],
    required_fields: Tuple[str, ...] = (),
    value_filters: Optional[Dict[str, Tuple[str, ...]]] = None,
    value_presets: Optional[Dict[str, str]] = None,
    constraints: Optional[Constraints] = None,
    debug_mode=False,
    n_clusters: int = 0,  # this is a debug feature only
) -> Dict[str, str]:
    """
    Extracts fields from markdown text, based on keyword search and data clustering.

    - keywords (required). Search for specified keywords in order to locate the target data. Think in terms
      of markdown, so you can even use as field some common markups. So for example, if you use '^#' as field,
      titles are a match.

    Some additional filtering options can be provided in order to reduce noise, discard results and provide
    preset default values (for example, provided by another extraction approach):

    - required_fields (optional). If not empty, it will only accept the results that has the given fields.
    - value_filters (optional): filter out provided list of values per field
    - value_presets (optional): A map from field to a value. If field was not extracted, it is added to the final
      result. If it is extracted and has the given value, the group score is increased.
    - constraints (optional): a dict field: pattern to specify regex patterns that must be fulfilled by specific
      field in order to accept the extracted result. Pattern can also be special keywords:
        * datetype - value extracted must be a date
    - debug (optional, boolean): if True, provides additional debug information in order to understand what
      the algorithm is doing
    - n_clusters (optional, int): this is a debug feature only. It should not be used in practical situation.
      it can be used to understand how the algorithm behaves with different number of clusters
    """

    def _select_group(m: re.Match):
        if m.groups()[0].startswith("| |"):
            return ""
        text = m.groups()[-1].strip().strip("| \n")
        text = re.sub(r"^\*\*", "", text)
        text = re.sub(r"\*\*$", "", text)
        return text

    if keywords:
        required_fields += tuple(["title" if k.startswith("^#") else k for k in keywords])

    # generate matches
    matches: List[Tuple[str, re.Match]] = []
    max_groups = 0
    for keyword in keywords:
        mlist = list(re.finditer(rf"\|\s*{keyword}\s*\|((?s:.)+?)\|", markdown, flags=re.I))
        if not mlist:
            mlist = list(re.finditer(rf"{keyword}\s*([:|\s\n*]+)(.+)", markdown, flags=re.M | re.I))
        matches.extend(("title" if "#" in keyword else keyword, m) for m in mlist)
        max_groups = max(max_groups, len(mlist))

    # group with k-means by position in text
    groups: Dict[int, List[Tuple[str, re.Match]]] = defaultdict(list)
    groups[-1] = []
    if max_groups > 0:
        features = [m.span() for _, m in matches]
        kmeans = KMeans(n_clusters=n_clusters or max_groups, random_state=0, n_init="auto").fit(features)
        for grp, mch in zip(kmeans.labels_, matches):
            groups[grp].append(mch)

    if debug_mode:
        print(pformat(groups))
    # score groups
    max_score = -len(required_fields)
    max_score_group = {}
    max_score_group_idx = -1

    for idx, results in groups.items():
        results = [
            (k, m)
            for k, m in results
            if not any([re.search(vv, _select_group(m)) for vv in (value_filters or {}).get(k, [])])
        ]
        extracted_data = {f: _select_group(m) for f, m in results}
        if constraints is not None:
            apply_constraints(extracted_data, constraints)
        score = len(extracted_data)

        for k, v in (value_presets or {}).items():
            if extracted_data.get(k) == v:
                score += 1
            extracted_data.setdefault(k, v)

        for k, v in extracted_data.items():
            for j, w in extracted_data.items():
                if k != j and v and w and (w in v or v in w):
                    score -= 1
                    if debug_mode:
                        print(f"Reducing score by one: {k}:{v} {j}:{w}")

        missing_required_fields = set(required_fields).difference(set(extracted_data.keys()))
        score -= len(missing_required_fields)

        if score > max_score:
            max_score = score
            max_score_group = extracted_data
            max_score_group_idx = idx

    missing_required_fields = set(required_fields).difference(set(max_score_group.keys()))
    if debug_mode:
        print("Extracted fields on stage 1:", list(max_score_group.keys()))
        print("Missing required fields stage 1:", missing_required_fields)
    # try to add missing fields from secondary groups
    # return max_score_group
    if max_score_group and missing_required_fields:
        center = kmeans.cluster_centers_[max_score_group_idx]
        for field in missing_required_fields:
            if field.startswith("^#"):
                field = "title"
            better_extra_candidate = None
            better_extra_candidate_distance = float("inf")
            for idx, results in groups.items():
                if idx != max_score_group_idx:
                    for k, m in results:
                        if (
                            k == field
                            and (distance := np.linalg.norm(center - m.span())) < better_extra_candidate_distance
                        ):
                            better_extra_candidate_distance = float(distance)
                            better_extra_candidate = m
            if better_extra_candidate is not None:
                max_score_group[field] = _select_group(better_extra_candidate)
    return max_score_group
