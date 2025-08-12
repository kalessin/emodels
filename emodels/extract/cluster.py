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
from emodels.scrapyutils.response import ExtractTextResponse


def tile_extraction(
    response: ExtractTextResponse, keywords: Tuple[str, ...], debug_mode: bool = False, **extract_kwargs
):
    # return extract_by_keywords(response.markdown, keywords, debug_mode=debug_mode, **extract_kwargs)
    prev_groups = None
    current_response = response
    groups, kmeans = apply_kmeans_clustering(current_response.markdown, keywords, debug_mode=debug_mode)
    return groups.values()
    while True:
        groups, kmeans = apply_kmeans_clustering(current_response.markdown, keywords)
        if prev_groups is not None and len(groups) > len(prev_groups):
            break
        for grp in groups.values():
            pass
        prev_groups = groups

    return prev_groups.values()


def apply_kmeans_clustering(
    markdown: str, keywords: Tuple[str, ...], n_clusters: int = 0, debug_mode: bool = False
) -> Tuple[Dict[int, List[Tuple[str, re.Match]]], KMeans | None]:
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
    kmeans = None
    if max_groups > 0:
        features = [m.span() for _, m in matches]
        kmeans = KMeans(n_clusters=n_clusters or max_groups, random_state=0, n_init="auto").fit(features)
        for grp, mch in zip(kmeans.labels_, matches):
            groups[int(grp)].append(mch)

    if debug_mode:
        print(pformat(groups))

    return groups, kmeans


def extract_by_keywords(
    markdown: str,
    keywords: Tuple[str, ...],
    required_fields: Tuple[str, ...] = (),
    value_filters: Optional[Dict[str, Tuple[str, ...]]] = None,
    value_presets: Optional[Dict[str, str]] = None,
    constraints: Optional[Constraints] = None,
    debug_mode: bool = False,
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

    def _select_group(m: re.Match) -> Tuple[str, int]:
        if m.groups()[0].startswith("| |"):
            return "", 0
        score = 0
        text = m.groups()[-1]
        if (new_text := text.strip().strip("| \n")) != text:
            score += 1
            text = new_text
        if (new_text := re.sub(r"^\*\*", "", text)) != text:
            score += 1
            text = new_text
        if (text := re.sub(r"\*\*$", "", text)) != text:
            score += 1
            text = new_text
        return text, score

    def _best_values_dict(extracted_data: Dict[str, List[Tuple[str, int]]]) -> Dict[str, str]:
        result = {}
        for k, vv_scores in extracted_data.items():
            max_score = 0
            best_value = ""
            for vv, score in vv_scores:
                if score >= max_score:
                    max_score = score
                    best_value = vv
            result[k] = best_value
        return result

    groups, kmeans = apply_kmeans_clustering(markdown, keywords, n_clusters=n_clusters, debug_mode=debug_mode)

    if keywords:
        required_fields += tuple(["title" if k.startswith("^#") else k for k in keywords])

    # score groups
    max_score = -len(required_fields)
    max_score_group = {}
    max_score_group_idx = -1

    for idx, results in groups.items():
        results = [
            (k, m)
            for k, m in results
            if not any([re.search(vv, _select_group(m)[0]) for vv in (value_filters or {}).get(k, [])])
        ]
        extracted_data: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        for k, m in results:
            extracted_data[k].append(_select_group(m))
        if debug_mode:
            print("Candidate extraction:", pformat(dict(extracted_data)))
        extracted_dict: Dict[str, str] = _best_values_dict(extracted_data)
        if constraints is not None:
            apply_constraints(extracted_dict, constraints)
        score = len(extracted_dict)

        for k, v in (value_presets or {}).items():
            if extracted_dict.get(k) == v:
                score += 1
            extracted_dict.setdefault(k, v)

        for k, v in extracted_dict.items():
            for j, w in extracted_dict.items():
                if k != j and w and w in v:
                    score -= 1
                    if debug_mode:
                        print(f"Reducing score by one: {k}:{v} {j}:{w}")

        missing_required_fields = set(required_fields).difference(set(extracted_dict.keys()))
        score -= len(missing_required_fields)

        if score > max_score:
            max_score = score
            max_score_group = extracted_dict
            max_score_group_idx = idx

    missing_required_fields = set(required_fields).difference(set(max_score_group.keys()))
    if debug_mode:
        print("Max score group:", pformat(max_score_group))
        print("Extracted fields on stage 1:", list(max_score_group.keys()))
        print("Missing required fields stage 1:", missing_required_fields)
    # try to add missing fields from secondary groups
    # return max_score_group
    if max_score_group and missing_required_fields and kmeans is not None:
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
                max_score_group[field] = _select_group(better_extra_candidate)[0]

    for k, v in list(max_score_group.items()):
        for j, w in max_score_group.items():
            if k != j and w and w in v:
                vv = re.sub(w, "", v, flags=re.I)
                if v == vv:
                    continue
                vvv = re.sub(j, "", vv, flags=re.I)
                if vv == vvv:
                    continue
                vvvv = vvv.strip("*| :")
                if vvvv != vvv and vvvv in v:
                    max_score_group[k] = vvvv

    return max_score_group
