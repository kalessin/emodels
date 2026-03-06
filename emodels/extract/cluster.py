"""
Cluster extraction algorithm
"""

import re
from pprint import pformat
from collections import defaultdict, OrderedDict
from typing import List, Dict, Tuple, Optional

from sklearn.cluster import KMeans
import numpy as np

from emodels.extract.utils import Constraints, apply_constraints, Result, Text, Keyword, Match, re_match_to_match
from emodels.scrapyutils.response import ExtractTextResponse


def tiles_kmeans(
    matches: List[Tuple[Keyword, Match]], keywords: Tuple[Keyword, ...], debug_mode: bool = False
) -> Dict[int, List[Tuple[Keyword, Match]]]:
    keywords_matches = sorted([(k, m) for k, m in matches], key=lambda x: x[1][1])
    groups: List[Dict[Keyword, Match]] = []
    for k, m in keywords_matches:
        if not groups:
            groups.append(OrderedDict({k: m}))
            if debug_mode:
                print("Added", {k: m[0]}, "to new group. Index:", m[1:])
        elif k in groups[-1].keys():
            current_group = list(groups[-1].items())
            present_keywords = list(k for k, _ in current_group)
            filled_keywords = list(k for k in keywords if k in present_keywords)
            if len(current_group) > 1 and current_group[0][0] == k:
                diff1 = current_group[1][1][1] - current_group[0][1][2]
                diff2 = m[1] - current_group[-1][1][2]
                if debug_mode:
                    print("diff1:", diff1, "diff2:", diff2)
                if diff2 < diff1 and present_keywords != filled_keywords:
                    if debug_mode:
                        print("  Present keywords:", present_keywords, "Filled keywords:", filled_keywords)
                    groups[-1].pop(k)  # remove previous match to preserver ordering of keywords in group
                    groups[-1][k] = m
                    if debug_mode:
                        print("Updated", k, "to", m[0], "in current group. Index:", m[1:])
                else:
                    groups.append(OrderedDict({k: m}))
                    if debug_mode:
                        print("Added", {k: m[0]}, "to new group. Index:", m[1:])
            else:
                groups.append(OrderedDict({k: m}))
                if debug_mode:
                    print("Added", {k: m[0]}, "to new group. Index:", m[1:])
        else:
            groups[-1][k] = m
            if debug_mode:
                print("Added", {k: m[0]}, "to current group. Index:", m[1:])
    groups = [g for g in groups if len(g) > 2]
    groups.insert(0, {})
    return {i - 1: list(g.items()) for i, g in enumerate(groups)}


def clean_group(
    m: Match, group_results: Optional[List[Tuple[Keyword, Match]]] = None, keywords: Tuple[Keyword, ...] = ()
) -> Tuple[Text, int]:
    return clean_text(m[3], group_results=group_results, keywords=keywords)


def clean_text(
    text: Text, group_results: Optional[List[Tuple[Keyword, Match]]] = None, keywords: Tuple[Keyword, ...] = ()
) -> Tuple[Text, int]:
    if text.startswith("| |"):
        return Text(""), 0
    score = 0
    for _, mm in group_results or []:
        subtext = mm[0]
        if subtext in text and subtext != text:
            text = Text(text.replace(subtext, ""))
    changed = True
    while changed:
        changed = False
        if (new_text := re.sub(r"^\*\*", "", text)) != text:
            score += 1
            text = Text(new_text)
            changed = True
        if (new_text := re.sub(r"\*\*$", "", text)) != text:
            score += 1
            text = Text(new_text)
            changed = True
        if (new_text := text.strip().strip("| \n")) != text:
            score += 1
            text = Text(new_text)
            changed = True
        for kw in keywords:
            if (new_text := re.sub(rf"\s*{re.escape(kw)}\s*", "", text, flags=re.I)) != text:
                score += 1
                text = Text(new_text)
                changed = True
    return text, score


def apply_kmeans_clustering(
    markdown: str,
    keywords: Tuple[Keyword, ...],
    n_clusters: int = 0,
    tiles_mode: bool = False,
    debug_mode: bool = False,
) -> Tuple[Dict[int, List[Tuple[Keyword, Match]]], KMeans | None]:
    # generate matches
    matches: List[Tuple[Keyword, Match]] = []
    max_groups = 0
    for keyword in keywords:
        mlist: List[Match] = [
            re_match_to_match(m) for m in re.finditer(rf"\|\s*{keyword}\s*\|((?s:.)+?)\|", markdown, flags=re.I)
        ]
        if not all([clean_group(m, keywords=keywords)[0] for m in mlist]):
            mlist = [
                re_match_to_match(m)
                for m in re.finditer(rf"\|[ \t]*{keyword}[ \t]*\|[ \t]*(?:\|[ \t]*)?((?s:.)+?)\|", markdown, flags=re.I)
            ]
        if not mlist:
            mlist = [
                re_match_to_match(m) for m in re.finditer(rf"{keyword}\s*([:|\s\n*]+)(.+)", markdown, flags=re.M | re.I)
            ]
        matches.extend((Keyword("title") if "#" in keyword else keyword, m) for m in mlist)
        max_groups = max(max_groups, len(mlist))

    # group with k-means by position in text
    groups: Dict[int, List[Tuple[Keyword, Match]]] = defaultdict(list)
    groups[-1] = []
    kmeans = None
    if max_groups > 0:
        if tiles_mode:
            groups = tiles_kmeans(matches, keywords, debug_mode=debug_mode)
        else:
            features: List[Tuple[int, int]] = [(m[1], m[2]) for _, m in matches]
            kmeans = KMeans(
                n_clusters=n_clusters or max_groups,
                random_state=0,
                n_init="auto",
            ).fit(features)
            for grp, mch in zip(kmeans.labels_, matches):
                groups[int(grp)].append(mch)
    if debug_mode:
        print(pformat(groups))
    return groups, kmeans


def extract_by_keywords(
    markdown: str,
    keywords: Tuple[Keyword, ...],
    required_fields: Tuple[Keyword, ...] = (),
    value_filters: Optional[Dict[Keyword, Tuple[Text, ...]]] = None,
    value_presets: Optional[Dict[Keyword, Text]] = None,
    constraints: Optional[Constraints] = None,
    tiles_mode: bool = False,
    debug_mode: bool = False,
    response: Optional[ExtractTextResponse] = None,
    n_clusters: int = 0,  # this is a debug feature only
) -> List[Result]:
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
      field in order to accept it. If field value does not match the constraint, the field is popped out from the
      result. Pattern can also be special keywords. See apply_constraints docstring for special keywords available.
    - tiles_mode (optional, boolean): if True, do tiles-based extraction. In this mode, it will return all groups
      that match the same pattern.
    - response (optional, ExtractTextResponse): if provided, it will be used to apply additional regexes in order
      to extract missing fields from the best group or the tiles groups (if tiles_mode is True).
    - debug (optional, boolean): if True, provides additional debug information in order to understand what
      the algorithm is doing
    - n_clusters (optional, int): this is a debug feature only. It should not be used in practical situation.
      it can be used to understand how the algorithm behaves with different number of clusters
    """

    def _best_values_dict(extracted_data: Dict[Keyword, List[Tuple[Text, int]]]) -> Dict[Keyword, Text]:
        result = {}
        for k, vv_scores in extracted_data.items():
            max_score = 0
            best_value = Text("")
            for vv, score in vv_scores:
                if score >= max_score:
                    max_score = score
                    best_value = vv
            result[k] = best_value
        return result

    groups, kmeans = apply_kmeans_clustering(
        markdown, keywords, n_clusters=n_clusters, tiles_mode=tiles_mode, debug_mode=debug_mode
    )

    if keywords:
        required_fields += tuple([Keyword("title") if k.startswith("^#") else k for k in keywords])

    # score groups
    max_score = -len(required_fields)
    max_score_group: Result = Result({})
    max_score_group_idx = -1
    tiles_groups: List[Result] = []

    for idx, results in groups.items():
        results = [
            (k, m)
            for k, m in results
            if not any([re.search(vv, clean_group(m, results)[0]) for vv in (value_filters or {}).get(k, [])])
        ]
        extracted_data: Dict[Keyword, List[Tuple[Text, int]]] = defaultdict(list)
        for k, m in results:
            extracted_data[k].append(clean_group(m, results))
        extracted_dict: Dict[Keyword, Text] = _best_values_dict(extracted_data)
        if constraints is not None:
            apply_constraints(extracted_dict, constraints)
        score = len(extracted_dict)
        if debug_mode:
            print("Candidate extraction:", pformat(dict(extracted_data)), "score:", score)

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
            max_score_group = Result(extracted_dict)
            max_score_group_idx = idx
            tiles_groups = []
        elif tiles_mode and (score == max_score or (score > 0 and not missing_required_fields)):
            tiles_groups.append(Result(extracted_dict))

    missing_required_fields = set(required_fields).difference(set(max_score_group.keys()))
    if debug_mode:
        print("Max score group:", pformat(max_score_group))
        print("Extracted fields on stage 1:", list(max_score_group.keys()))
        print("Missing required fields stage 1:", missing_required_fields)
    # try to add missing fields from secondary groups
    # return max_score_group
    if not tiles_mode and max_score_group and missing_required_fields and kmeans is not None:
        center = kmeans.cluster_centers_[max_score_group_idx]
        for field in missing_required_fields:
            if field.startswith("^#"):
                field = Keyword("title")
            better_extra_candidate = None
            better_extra_candidate_distance = float("inf")
            for idx, results in groups.items():
                if idx != max_score_group_idx:
                    for k, m in results:
                        if (
                            k == field
                            and (distance := np.linalg.norm(center - (m[1], m[2]))) < better_extra_candidate_distance
                        ):
                            better_extra_candidate_distance = float(distance)
                            better_extra_candidate = m
            if better_extra_candidate is not None:
                max_score_group[field] = clean_group(better_extra_candidate)[0]

    max_score_groups = [max_score_group]
    if tiles_mode:
        max_score_groups += tiles_groups
    # clean results
    for max_score_group in max_score_groups:
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
                        max_score_group[k] = Text(vvvv)

    return max_score_groups
