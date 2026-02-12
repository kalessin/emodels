from typing import Optional, Callable, List, Dict, Tuple, Set

from emodels.scrapyutils.response import ExtractTextResponse
from emodels.extract.table import parse_tables_from_response, default_validate_result, Columns
from emodels.extract.cluster import extract_by_keywords
from emodels.extract.utils import Constraints, Result


def parse_combined_from_response(
    response: ExtractTextResponse,
    columns: Columns,
    validate_result: Callable[[Result, Columns], bool] = default_validate_result,
    dedupe_keywords: Columns = Columns(()),
    constraints: Optional[Constraints] = None,
    value_filters: Optional[Dict[str, Tuple[str, ...]]] = None,
    value_presets: Optional[Dict[str, str]] = None,
    debug_mode: bool = False,
) -> Result:
    results_to_combine: List[Result] = []
    seen_columns: Set[str] = set()
    for result in parse_tables_from_response(
        response, columns, validate_result, dedupe_keywords, constraints, max_tables=2
    ):
        results_to_combine.append(result)
        seen_columns.update(result.keys())
    required_fields = tuple(c for c in columns if c not in seen_columns)
    if required_fields:
        result = extract_by_keywords(
            response.markdown, columns, required_fields, value_filters, value_presets, constraints, debug_mode
        )
        if result:
            results_to_combine.append(result)
    if results_to_combine:
        results_to_combine, final_result = results_to_combine[:-1], results_to_combine[-1]
        for result in results_to_combine[::-1]:
            final_result.update(result)
        return final_result
    return Result({})
