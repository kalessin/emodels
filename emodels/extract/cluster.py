"""
Cluster extraction algorithm
"""

import re
from pprint import pformat
from collections import defaultdict, OrderedDict
from typing import List, Dict, Tuple, Optional
from operator import itemgetter

from sklearn.cluster import KMeans
import numpy as np

from emodels.extract.utils import (
    Constraints,
    apply_constraints,
    Result,
    Text,
    Keyword,
    Match,
    parse_additional_regexes,
    re_match_to_match,
)
from emodels.scrapyutils.response import ExtractTextResponse


def tiles_kmeans(
    matches: List[Tuple[Keyword, Match]],
    keywords: Tuple[Keyword, ...],
    additional_keywords: Tuple[Keyword, ...] = (),
    debug_mode: bool = False,
) -> Dict[int, List[Tuple[Keyword, Match]]]:
    keywords = tuple([Keyword("title") if k.startswith("^#") else k for k in keywords])
    keywords_matches = sorted([(k, m) for k, m in matches], key=lambda x: x[1][1])
    groups: List[Dict[Keyword, Match]] = []
    for k, m in keywords_matches:
        if not groups:
            groups.append(OrderedDict({k: m}))
            if debug_mode:
                print("Added", {k: m[0]}, "to new group. Index:", m[1:])
        else:
            current_group = list(groups[-1].items())
            present_keywords = list(k for k, _ in current_group)
            filled_keywords = list(k for k in keywords if k in present_keywords)
            if k in groups[-1].keys():
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
                    # for now the 5 is just a fixed heuristic. This can be improved further.
                    diffm = m[1] - current_group[-1][1][2]
                    if current_group[-1][0] == k and 0 < diffm < 5:
                        _, m0 = current_group[-1]
                        new_match = Match((Text(m0[0] + " " * diffm + m[0]), m0[1], m[2], Text(m0[3] + "| " + m[3])))
                        groups[-1][k] = new_match
                        if debug_mode:
                            print("Merged", k, "to", new_match[0], "in current group. Index:", new_match[1:])
                    else:
                        groups.append(OrderedDict({k: m}))
                        if debug_mode:
                            print("Added", {k: m[0]}, "to new group. Index:", m[1:])
            else:
                sorted_candidate_keywords = [kk for kk in (present_keywords + [k]) if kk not in additional_keywords]
                sorted_keywords = [kk for kk in keywords if kk in (filled_keywords + [k])]
                if sorted_candidate_keywords == sorted_keywords:
                    # order is preserved, we can add to current group.
                    if debug_mode:
                        print("Present keywords:", present_keywords, "Filled keywords:", filled_keywords)
                    groups[-1][k] = m
                    if debug_mode:
                        print("Added", {k: m[0]}, "to current group. Index:", m[1:])
                else:
                    groups.append(OrderedDict({k: m}))
                    if debug_mode:
                        print("Added", {k: m[0]}, "to new group. Index:", m[1:])
    groups = [g for g in groups if len(g) > 2 - len(additional_keywords)]
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
    response: ExtractTextResponse,
    keywords: Tuple[Keyword, ...],
    n_clusters: int = 0,
    tiles_mode: bool = False,
    additional_regexes: Optional[Dict[Keyword, Tuple[str | Tuple[str | None, str], ...]]] = None,
    fill_fields: Tuple[Keyword, ...] = (),
    debug_mode: bool = False,
) -> Tuple[Dict[int, List[Tuple[Keyword, Match]]], KMeans | None]:
    # generate matches
    additional_regexes = additional_regexes or {}
    matches: List[Tuple[Keyword, Match]] = []
    max_groups = 0
    markdown = response.markdown
    for keyword in keywords:
        mlist: List[Match] = [
            re_match_to_match(m) for m in re.finditer(rf"\|\s*{keyword}\s*\|((?s:.)+?)\|", markdown, flags=re.I)
        ]
        if mlist and debug_mode:
            print(f"Matches I for keyword '{keyword}':", len(mlist))
        if not all([clean_group(m, keywords=keywords)[0] for m in mlist]):
            mlist = [
                re_match_to_match(m)
                for m in re.finditer(rf"\|[ \t]*{keyword}[ \t]*\|[ \t]*(?:\|[ \t]*)?((?s:.)+?)\|", markdown, flags=re.I)
            ]
            if mlist and debug_mode:
                print(f"Matches II for keyword '{keyword}':", len(mlist))
        if not mlist or tiles_mode:
            nmlist = [
                re_match_to_match(m) for m in re.finditer(rf"{keyword}\s*([:|\s\n*]+)(.+)", markdown, flags=re.M | re.I)
            ]
            if nmlist and debug_mode:
                print(f"Matches III for keyword '{keyword}':", len(nmlist))
            if not tiles_mode or len(nmlist) > len(mlist):
                mlist = nmlist
        matches.extend((Keyword("title") if "#" in keyword else keyword, m) for m in mlist)
        max_groups = max(max_groups, len(mlist))

    for k, mm in parse_additional_regexes(additional_regexes, response, tiles_mode=tiles_mode).items():
        for m in mm:
            matches.append((k, m))
            if not tiles_mode:
                break

    # group with k-means by position in text
    groups: Dict[int, List[Tuple[Keyword, Match]]] = defaultdict(list)
    groups[-1] = []
    kmeans = None
    if max_groups > 0:
        if tiles_mode:
            additional_keywords = tuple(additional_regexes.keys()) + fill_fields
            groups = tiles_kmeans(matches, keywords, additional_keywords=additional_keywords, debug_mode=debug_mode)
        else:
            features: List[Tuple[int, int]] = [(m[1], m[2]) for _, m in matches]
            kmeans = KMeans(
                n_clusters=n_clusters or max_groups,
                random_state=0,
                n_init="auto",
            ).fit(features)
            for grp, mch in zip(kmeans.labels_, matches):
                groups[int(grp)].append(mch)

    for fk in fill_fields:
        for group in groups.values():
            start = end = 0
            last_k: Keyword | None = None
            for idx, k in enumerate([Keyword("title") if k.startswith("^#") else k for k in keywords]):
                if k == fk and last_k is not None:
                    start = [m[2] for kk, m in group if kk == last_k][0]
                elif k in [kk for kk, _ in group]:
                    if start > end:
                        end = [m[1] for kk, m in group if kk == k][0]
                        text = Text(response.markdown[start:end])
                        cleaned_text = clean_text(text, group_results=group, keywords=keywords)[0]
                        group.insert(idx - 1, (fk, Match((text, start, end, cleaned_text))))
                        if debug_mode:
                            print(f"Filled field '{fk}' with text between '{last_k}' and '{k}':", cleaned_text)
                    last_k = k

    if debug_mode:
        print(pformat(groups))
    return groups, kmeans


def extract_by_keywords(
    response: ExtractTextResponse,
    keywords: Tuple[Keyword, ...],
    required_fields: Tuple[Keyword, ...] = (),
    value_filters: Optional[Dict[Keyword, Tuple[Text, ...]]] = None,
    value_presets: Optional[Dict[Keyword, Text]] = None,
    constraints: Optional[Constraints] = None,
    tiles_mode: bool = False,
    tiles_mode_tolerance: int | float = 0.45,
    additional_regexes: Optional[Dict[Keyword, Tuple[str | Tuple[str | None, str], ...]]] = None,
    fill_fields: Tuple[Keyword, ...] = (),
    debug_mode: bool = False,
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
    - tiles_mode_tolerance (optional, int | float): tolerance of missing required fields in tiles mode.
      If integer and >= 1, define tolerance in terms of number of missing fields. If float, define tolerance as
      a fraction of the total required fields. Default: 1.
    - additional_regexes (optional): a dict field: list of regexes to specify additional regexes to be applied for
      specific fields.
    - fill_fields (optional): a tuple of fields without explicit keyword in target markdown, but yet can be foun
      between text with explicit keywords. This is used to put all text between keyword extracted fields. This must be
      a keyword present in the keywords tuple, despite is not strictly a keyword, but yet is necesary to understand
      the order between the keywords extracted texts.
    - debug_mode (optional, boolean): if True, provides additional debug information in order to understand what
      the algorithm is doing
    - n_clusters (optional, int): this is a debug feature only. It should not be used in practical situation.
      it can be used to understand how the algorithm behaves with different number of clusters
    """

    assert keywords, "At least one keyword should be provided. The more keywords, the better the algorithm works."

    def _best_values_dict(extracted_data: Dict[Keyword, List[Tuple[Text, int]]]) -> Dict[Keyword, Text]:
        result = {}
        for k, vv_scores in extracted_data.items():
            max_score = -1
            best_value = Text("")
            for vv, score in vv_scores:
                if score > max_score:
                    max_score = score
                    best_value = vv
            result[k] = best_value
        return result

    groups, kmeans = apply_kmeans_clustering(
        response,
        keywords,
        n_clusters=n_clusters,
        tiles_mode=tiles_mode,
        additional_regexes=additional_regexes,
        fill_fields=fill_fields,
        debug_mode=debug_mode,
    )

    if not required_fields:
        all_keywords = set(keywords).union(set(additional_regexes.keys() if additional_regexes else set()))
        required_fields = tuple([Keyword("title") if k.startswith("^#") else k for k in all_keywords])
    assert tiles_mode_tolerance >= 0, "tiles_mode_tolerance should be non-negative"
    tiles_mode_tolerance = (
        tiles_mode_tolerance if tiles_mode_tolerance < 1 else tiles_mode_tolerance / len(required_fields)
    )

    # score groups
    max_score = -len(required_fields)
    max_score_group: Result = Result({})
    max_score_group_idx = -1
    # list of tuple index, result, score
    tiles_groups: List[Tuple[int, Result, int]] = []

    for idx, results in groups.items():
        results = [
            (k, m)
            for k, m in results
            if not any(
                [re.search(vv, clean_group(m, results)[0], flags=re.I) for vv in (value_filters or {}).get(k, [])]
            )
        ]
        extracted_data: Dict[Keyword, List[Tuple[Text, int]]] = defaultdict(list)
        for k, m in results:
            extracted_data[k].append(
                clean_group(m, [(k, m) for k, m in results if k not in (additional_regexes or {})])
            )
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
            if tiles_mode:
                tiles_groups.append((idx, Result(extracted_dict), score))
                rejected_groups = [t for t in tiles_groups if t[2] < max_score * (1 - tiles_mode_tolerance)]
                tiles_groups = [t for t in tiles_groups if t[2] >= max_score * (1 - tiles_mode_tolerance)]
                if debug_mode and rejected_groups:
                    print("Results rejected:", rejected_groups, ". Max score:", max_score)
        elif tiles_mode:
            if score >= max_score * (1 - tiles_mode_tolerance):
                tiles_groups.append((idx, Result(extracted_dict), score))
            elif debug_mode:
                print("Result rejected:", extracted_dict, ". Max score:", max_score, ", Actual score:", score)

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

    if tiles_mode:
        max_score_groups = [t[1] for t in sorted(tiles_groups, key=itemgetter(0))]
    else:
        max_score_groups = [max_score_group]
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
        if not tiles_mode and Keyword("url") not in max_score_group:
            max_score_group[Keyword("url")] = Text(response.url)

    return max_score_groups
