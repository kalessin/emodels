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
    safe_keyword_escape,
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
                        new_match = Match(
                            (Text(m0[0] + " " * diffm + m[0]), m0[1], m[2], Text(m0[3].strip() + " | " + m[3]))
                        )
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


def clean_group(m: Match) -> Text:
    return clean_text(m[3])


def clean_text(text: Text) -> Text:
    if text.startswith("| |"):
        return Text("")
    changed = True
    while changed:
        changed = False
        if (new_text := re.sub(r"^\*\*", "", text)) != text:
            text = Text(new_text)
            changed = True
        if (new_text := re.sub(r"\*\*$", "", text)) != text:
            text = Text(new_text)
            changed = True
        if (new_text := text.strip().strip("| \n")) != text:
            text = Text(new_text)
            changed = True
    return text


def apply_kmeans_clustering(
    response: ExtractTextResponse,
    keywords: Tuple[Keyword, ...],
    n_clusters: int = 0,
    tiles_mode: bool = False,
    additional_regexes: Optional[Dict[Keyword, Tuple[str | Tuple[str | None, str], ...]]] = None,
    fill_fields: Tuple[Keyword, ...] = (),
    multiline_fields: Optional[Dict[Keyword, int]] = None,
    debug_mode: bool = False,
) -> Tuple[Dict[int, List[Tuple[Keyword, Match]]], KMeans | None]:
    # generate matches
    additional_regexes = additional_regexes or {}
    matches: List[Tuple[Keyword, Match]] = []
    max_groups = 0
    markdown = response.markdown
    multiline_fields = multiline_fields or {}
    for keyword in keywords:
        keywordp = safe_keyword_escape(keyword)
        mlist: List[Match] = [
            re_match_to_match(m) for m in re.finditer(rf"\|\s*{keywordp}\s*\|((?s:.)+?)\|", markdown, flags=re.I)
        ]
        if mlist and debug_mode:
            print(f"Matches I for keyword '{keyword}':", len(mlist))
        if not all([clean_group(m) for m in mlist]):
            mlist = [
                re_match_to_match(m)
                for m in re.finditer(
                    rf"\|[ \t]*{keywordp}[ \t]*\|[ \t]*(?:\|[ \t]*)?((?s:.)+?)\|", markdown, flags=re.I
                )
            ]
            if mlist and debug_mode:
                print(f"Matches II for keyword '{keyword}':", len(mlist))
        if not mlist or tiles_mode:
            reps = multiline_fields.get(keyword, 1)
            nmlist = [
                re_match_to_match(m)
                for m in re.finditer(
                    rf"{keywordp}\s*([:|\s\n*]+)((?:\n*.+\n?){{1,{reps}}})", markdown, flags=re.M | re.I
                )
            ]
            if nmlist and debug_mode:
                print(f"Matches III for keyword '{keyword}':", len(nmlist))
            if not tiles_mode or len(nmlist) > len(mlist):
                mlist = nmlist
        # if no success, try to match using a more flexible keyword regex.
        if not mlist and (sub_keyword := re.sub(r"\s+", r"[:|\\s\\n*]+", keyword)) != keyword:
            mlist = [
                re_match_to_match(m) for m in re.finditer(rf"\|\s*{sub_keyword}\s*\|((?s:.)+?)\|", markdown, flags=re.I)
            ]
            if mlist and debug_mode:
                print(f"Matches IV for keyword '{keyword}':", len(mlist))
            if not mlist:
                mlist = [
                    re_match_to_match(m)
                    for m in re.finditer(
                        rf"\|[ \t]*{sub_keyword}[ \t]*\|[ \t]*(?:\|[ \t]*)?((?s:.)+?)\|", markdown, flags=re.I
                    )
                ]
                if mlist and debug_mode:
                    print(f"Matches V for keyword '{keyword}':", len(mlist))
            if not mlist:
                mlist = [
                    re_match_to_match(m)
                    for m in re.finditer(rf"{sub_keyword}\s*([:|\s\n*]+)(.+)", markdown, flags=re.M | re.I)
                ]
                if mlist and debug_mode:
                    print(f"Matches VI for keyword '{keyword}':", len(mlist))

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
                init="k-means++",
                n_init=10,
                max_iter=300,
                tol=1e-4,
                random_state=0,
                algorithm="lloyd",
            ).fit(features)
            temp_groups: Dict[int, List[Tuple[Keyword, Match]]] = defaultdict(list)
            for grp, mch in zip(kmeans.labels_, matches):
                temp_groups[int(grp)].append(mch)
            # reorder groups by position of first match in group
            for matches in sorted(temp_groups.values(), key=lambda x: x[0][1][1]):
                if groups[-1] and groups[-1][-1][1][2] == matches[0][1][1]:
                    groups[-1].extend(matches)
                else:
                    idx = len(groups) - 1
                    groups[idx] = matches

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
                        cleaned_text = clean_text(text)
                        group.insert(idx - 1, (fk, Match((text, start, end, cleaned_text))))
                        if debug_mode:
                            print(f"Filled field '{fk}' with text between '{last_k}' and '{k}':", cleaned_text)
                    last_k = k

    groups = {k: sorted(v, key=lambda x: x[1][1]) for k, v in groups.items()}
    if debug_mode:
        print(pformat(groups))
    return groups, kmeans


def _best_values_selection(group_matches: List[Tuple[Keyword, Match]], debug_mode: bool) -> List[Tuple[Keyword, Match]]:
    data = defaultdict(list)
    for k, m in group_matches:
        data[k].append(m)
    result = {}
    for k, matches in data.items():
        max_score = -100
        for m in matches:
            score = -(m[2] - m[1] - len(m[3]) - len(k))  # prefer more compact matches that are closer to the keyword
            if score > max_score:
                max_score = score
                best_value = (k, m)
                if debug_mode:
                    print(f"New best value for keyword '{k}':", repr(m[0]), "with score:", score)
        result[k] = [best_value]
    return [(k, v[0][1]) for k, v in result.items()]


def _overlap_clean(k: Keyword, m: Match, kk: Keyword, mm: Match, debug_mode: bool) -> None | Match:
    klongest, kshortest = (k, kk) if len(k) >= len(kk) else (kk, k)
    if m[1] < mm[1] < m[2] and (m[2] != mm[2] or not klongest.endswith(kshortest)):
        shift = mm[1] - m[1]
        newm = Match((Text(m[0][:shift]), m[1], mm[1], Text(m[3][: shift - m[0].find(m[3])])))
        if debug_mode:
            print(
                f"Adjusted match for keyword '{k}' to avoid overlap with '{kk}':\n",
                m[0],
                "->",
                newm[0],
                "\n",
                m[3],
                "->",
                newm[3],
            )
        return newm
    return None


def extract_by_keywords(
    response: ExtractTextResponse,
    keywords: Tuple[Keyword, ...],
    required_fields: Tuple[Keyword, ...] = (),
    value_filters: Optional[Dict[Keyword, Tuple[str, ...]]] = None,
    value_presets: Optional[Dict[Keyword, Text]] = None,
    constraints: Optional[Constraints] = None,
    tiles_mode: bool = False,
    tiles_mode_tolerance: int | float = 0.45,
    additional_regexes: Optional[Dict[Keyword, Tuple[str | Tuple[str | None, str], ...]]] = None,
    fill_fields: Tuple[Keyword, ...] = (),
    multiline_fields: Optional[Dict[Keyword, int]] = None,
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

    - required_fields (optional). If not empty, it will reduce score of results that don't have any of the required
      fields. By default, all fields provided in keywords argument are required.
    - value_filters (optional): filter out provided list of value regexes per field.
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
    - additional_regexes (optional): A mapping (field -> regexes) for additional extraction capabilities.
      regexes is a list. Each element in regexes is used in the response.text_re() function provided by
      ExtractTextResponse. For each field, you can provide a list of alternatives. The first one extracting something
      will be the value used. Remaining ones will be discarded.
      Each element in regexes can be either a regex string, or a tuple where first element is a regex or None (for
      default regex) and the second regex the value of parameter tid (see emodels ExtractResponse.text_re docstring)
    - fill_fields (optional): a tuple of fields without explicit keyword in target markdown, but yet can be foun
      between text with explicit keywords. This is used to put all text between keyword extracted fields. This must be
      a keyword present in the keywords tuple, despite is not strictly a keyword, but yet is necesary to understand
      the order between the keywords extracted texts.
    - multiline_fields. By default, and in order to avoid to extract noise, the algorithm only extracts text in the
      same line as the keyword. This is ok for 99% of cases. With this option, you can specify a max number of lines
      to extract for specific fields. For example, if you specify {"address": 4}, the algorithm will extract up to 4
      lines of text for the "address" field.
    - debug_mode (optional, boolean): if True, provides additional debug information in order to understand what
      the algorithm is doing
    - n_clusters (optional, int): this is a debug feature only. It should not be used in practical situation.
      it can be used to understand how the algorithm behaves with different number of clusters
    """

    assert keywords, "At least one keyword should be provided. The more keywords, the better the algorithm works."

    if not required_fields:
        all_keywords = set(keywords).union(set(additional_regexes.keys() if additional_regexes else set()))
        required_fields = tuple([Keyword("title") if k.startswith("^#") else k for k in all_keywords])
    assert tiles_mode_tolerance >= 0, "tiles_mode_tolerance should be non-negative"
    tiles_mode_tolerance = (
        tiles_mode_tolerance if tiles_mode_tolerance < 1 else tiles_mode_tolerance / len(required_fields)
    )

    groups, kmeans = apply_kmeans_clustering(
        response,
        keywords,
        n_clusters=n_clusters,
        tiles_mode=tiles_mode,
        additional_regexes=additional_regexes,
        fill_fields=fill_fields,
        multiline_fields=multiline_fields,
        debug_mode=debug_mode,
    )

    # score groups
    max_score = -len(required_fields)
    max_score_group: Result = Result({})
    max_score_matches: List[Tuple[Keyword, Match]] = []
    max_score_group_idx = -1
    # a dict from group index to score
    groups_scores: Dict[int, int] = {}
    # list of tuple index, result, score
    tiles_groups: List[Tuple[int, Result, int]] = []
    group_matches: List[Tuple[Keyword, Match]]
    for idx, group_matches in groups.items():
        for iidx, (k, m) in enumerate(group_matches):
            for kk, mm in list(group_matches):
                if k == kk:
                    continue
                if (newm := _overlap_clean(k, m, kk, mm, debug_mode)) is not None:
                    group_matches[iidx] = (k, newm)
                    break

        group_matches = _best_values_selection(group_matches, debug_mode)

        if debug_mode:
            print(f"Group {idx} matches after best values selection:", pformat(group_matches))

        group_matches = [
            (k, m)
            for k, m in group_matches
            if not any([re.search(vv, clean_group(m), flags=re.I) for vv in (value_filters or {}).get(k, [])])
        ]

        extracted_data: Dict[Keyword, List[Tuple[Text, Match]]] = defaultdict(list)
        for k, m in group_matches:
            extracted_data[k].append((clean_group(m), m))

        extracted_dict: Dict[Keyword, Text] = {k: v[0][0] for k, v in extracted_data.items() if v[0][0]}
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

        if debug_mode:
            print("Candidate extraction:", pformat(extracted_dict), "score:", score)

        groups_scores[idx] = score
        if score > max_score:
            max_score = score
            max_score_group = Result(extracted_dict)
            max_score_matches = group_matches
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
            better_extra_candidate: Match | None = None
            better_extra_candidate_distance = float("inf")
            better_extra_candidate_score = -float("inf")
            for idx, group_matches in sorted(groups.items(), key=lambda x: groups_scores[x[0]], reverse=True):
                if idx != max_score_group_idx:
                    group_matches = [
                        (k, m)
                        for k, m in group_matches
                        if not any(
                            [re.search(vv, clean_group(m), flags=re.I) for vv in (value_filters or {}).get(k, [])]
                        )
                    ]

                    for k, m in group_matches:
                        if (
                            k == field
                            and (distance := np.linalg.norm(center - (m[1], m[2]))) < better_extra_candidate_distance
                            and groups_scores[idx] >= better_extra_candidate_score
                        ):
                            better_extra_candidate_distance = float(distance)
                            better_extra_candidate = m
                            better_extra_candidate_score = groups_scores[idx]
                            if debug_mode:
                                print(
                                    f"Found better candidate for missing field '{field}' in group {idx}:",
                                    better_extra_candidate[3],
                                )
            if better_extra_candidate is not None:
                for k, m in max_score_matches:
                    if (newm := _overlap_clean(field, better_extra_candidate, k, m, debug_mode)) is not None:
                        better_extra_candidate = newm
                        max_score_group[field] = clean_group(better_extra_candidate)
                        break
                for k, m in list(max_score_matches):
                    if (newm := _overlap_clean(k, m, field, better_extra_candidate, debug_mode)) is not None:
                        max_score_matches = [(kk, newm) if kk == k else (kk, mm) for kk, mm in max_score_matches]
                        max_score_group[k] = clean_group(newm)
                max_score_group[field] = clean_group(better_extra_candidate)
                better_extra_candidate = Match(
                    (
                        better_extra_candidate[0],
                        better_extra_candidate[1],
                        better_extra_candidate[2],
                        max_score_group[field],
                    )
                )
                max_score_matches.append((field, better_extra_candidate))
                max_score_matches = sorted(max_score_matches, key=lambda x: x[1][1])
        if constraints is not None:
            apply_constraints(max_score_group, constraints)

    if tiles_mode:
        max_score_groups = [t[1] for t in sorted(tiles_groups, key=itemgetter(0))]
    else:
        max_score_groups = [max_score_group]
    # some final cleanup
    for max_score_group in max_score_groups:
        for k, v in list(max_score_group.items()):
            if not v:
                max_score_group.pop(k)
        if not tiles_mode and Keyword("url") not in max_score_group:
            max_score_group[Keyword("url")] = Text(response.url)

    return max_score_groups
