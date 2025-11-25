import re
from operator import itemgetter
from urllib.parse import urlparse, ParseResult, urljoin
from typing import List, Dict, Set, Tuple, Callable, NewType, Optional, Iterable

from scrapy.http import TextResponse
from scrapy import Selector

from emodels.extract.utils import Constraints, apply_constraints, Result

NUMBER_RE = re.compile(r"\d+$")
MAX_HEADER_COLUMNS = 20

Columns = NewType("Columns", Tuple[str, ...])
Uid = NewType("Uid", Tuple[Tuple[str, str], ...])


def extract_row_text(row: Selector) -> List[str]:
    text = []
    for td in row.xpath(".//th") or row.xpath(".//td"):
        text.append(" ".join(td.xpath(".//text()").extract()).strip())
    return text


def iterate_rows(table: Selector, parsed: ParseResult | None = None) -> Iterable[Tuple[List[str], List[str]]]:
    for row in table.xpath(".//tr"):
        urls = []
        for _url in row.xpath(".//a/@href").extract():
            if not _url.startswith("mailto:"):
                url = _url
                if parsed is not None:
                    url = urljoin(parsed.geturl(), url)
                urls.append(url)
        yield extract_row_text(row), urls


def find_table_headers(table: Selector, candidate_fields: Tuple[str, ...]) -> List[List[str]]:
    score_rows: List[Tuple[List[str], int]] = []
    for rowtexts, _ in iterate_rows(table):
        row_score = 0
        lower_rowtexts = [t.lower() for t in rowtexts]
        for kw in candidate_fields:
            if kw in lower_rowtexts:
                row_score += 1
        if len(list(filter(None, rowtexts))) <= MAX_HEADER_COLUMNS:
            score_rows.append((rowtexts, row_score))
    return [i[0] for i in sorted(score_rows, key=itemgetter(1), reverse=True)]


def find_tables(
    tables: List[Selector], candidate_fields: Tuple[str, ...]
) -> List[Tuple[Selector, List[str]]]:
    # list of tuples (table selector, header, score1 score2)
    scored_tables: List[Tuple[Selector, List[str], int, int]] = []
    for table in tables[::-1]:
        headers = find_table_headers(table, candidate_fields)[0]
        score1 = len(
            list(
                filter(
                    None,
                    set(candidate_fields).intersection([f.lower() for f in headers]),
                )
            )
        )
        score2 = len(list(filter(None, headers)))
        if score1 <= MAX_HEADER_COLUMNS:
            scored_tables.append((table, headers, score1, score2))
    return [(c[0], c[1]) for c in sorted(scored_tables, key=itemgetter(2, 3), reverse=True)]


def extract_urls(urls: List[str], parsed: ParseResult) -> Dict[str, str]:
    url_data = {}
    for url in urls:
        if parsed.netloc in url:
            url_data["url"] = url
        else:
            url_data["website"] = url
    return url_data


def parse_table(table: Selector, headers: List[str], parsed: ParseResult):
    header_find_status = False
    headers_lower = [h.lower() for h in headers]
    for row, urls in iterate_rows(table, parsed):
        if not header_find_status and row != headers:
            continue
        header_find_status = True
        if row == headers:
            continue
        if len(row) != len(headers):
            continue
        data = dict(zip(headers_lower, row))
        data.update(extract_urls(urls, parsed))
        yield data


def parse_table_ii(table: Selector, headers: List[str], parsed: ParseResult):
    headers = list(filter(None, headers))
    header_find_status = False
    headers_lower = [h.lower() for h in headers]
    for row, urls in iterate_rows(table, parsed):
        row = list(filter(None, row))
        if not header_find_status and row != headers:
            continue
        header_find_status = True
        if row == headers:
            continue
        data = dict(zip(headers_lower, row))
        data.update(extract_urls(urls, parsed))
        yield data


def default_validate_result(result: Dict[str, str], columns: Columns) -> bool:
    return True


def score_results(results: List[Result]) -> int:
    score = 0
    for result in results:
        score += len([i for i in result.keys() if i])
    return score


def unique_id(result: Dict[str, str], dedupe_keywords: Columns) -> Tuple[Uid, Uid]:
    uid = []
    full_uid = []
    for key, value in result.items():
        if value:
            full_uid.append((key, value))
            if key in dedupe_keywords:
                uid.append((key, value))
    return Uid(tuple(uid)), Uid(tuple(full_uid))


def remove_all_empty_fields(results: List[Result]):
    fields: Set[str] = set()
    for result in results:
        fields.update(result.keys())
    all_empty_fields = []
    for field in fields:
        if all(not r[field] for r in results):
            all_empty_fields.append(field)
    for field in all_empty_fields:
        for result in results:
            result.pop(field)


def parse_tables_from_response(
    response: TextResponse,
    columns: Columns,
    validate_result: Callable[[Result, Columns], bool] = default_validate_result,
    dedupe_keywords: Columns = Columns(()),
    constraints: Optional[Constraints] = None,
    max_tables: int = 1,
) -> List[Result]:
    """
    Identifies and extracts data from an html table, based on the column names provided.
    response - The target response where to search for the table
    columns - the name of the columns to extract
    validate_result - a callable which validates and eventually filters out each result generated
                      by the algorithm
    dedupe_keywords - which columns use to deduplicate results (results with all same values in the same fields are
                      mutual dupes)
    """
    all_tables = response.xpath("//table")
    all_results: List[Result] = []
    seen: Set[Uid] = set()
    table_count = 0
    if all_tables:
        parsed = urlparse(response.url)
        for _, headers in find_tables(all_tables, columns):
            all_table_results: List[Result] = []
            fields: Set[str] = set()
            for parse_method in parse_table, parse_table_ii:
                all_table_results_method = []
                for table in all_tables:
                    for result in parse_method(table, headers, parsed):
                        uid, fuid = unique_id(result, dedupe_keywords)
                        if validate_result(result, columns) and uid not in seen and fuid not in seen:
                            if uid:
                                seen.add(uid)
                            if fuid:
                                seen.add(fuid)
                            if constraints is not None and apply_constraints(result, constraints):
                                continue
                            all_table_results_method.append(result)
                            fields.update(result.keys())
                if score_results(all_table_results_method) > score_results(all_table_results):
                    all_table_results = all_table_results_method
            for result in all_table_results:
                for field in fields:
                    result.setdefault(field, "")
            remove_all_empty_fields(all_table_results)
            if all_table_results:
                all_results.extend(all_table_results)
                table_count += 1
                if table_count == max_tables:
                    break
    return all_results
