import re
import json
from typing import Dict, Set, Tuple, Optional, Literal, Iterable, cast
from abc import abstractmethod

from scrapy.http import TextResponse
from scrapy import Request, Spider

from emodels.scrapyutils.response import ExtractTextResponse
from emodels.extract.cluster import extract_by_keywords
from emodels.extract.utils import apply_additional_regexes, Constraints, Result, Keyword, Text
from emodels.extract.table import parse_tables_from_response, unique_id, Uid
from emodels.extract.combined import parse_combined_from_response


MULTIS_RE = re.compile(r"\s+")
MARKDOWN_URL_RE = re.compile(r"\[.*\]\((.+)\)")


class ExtractionSpider(Spider):
    name: str

    # target fields to extract
    fields: Tuple[Keyword, ...] = ()

    # a list of fields to drop from the final result. Applies on item post processing. that is, from final items
    # that are valid. The difference with value_filters is that value_filters affects extraction process itself,
    # by reducing the score of candidates with values matching the patterns, while drop_fields just removes the
    # field from the final result.
    drop_fields: Tuple[Keyword, ...] = ()

    # a dict of field -> tuple of regexes to define patterns that makes any matching item to be completely dropped out.
    drop_items: Dict[Keyword, Tuple[str]] = {}

    # - in listing mode, it will try to extract data from a listing table from each visited page.
    # - in item mode, it will try to extract item-like data from each visited page (that is, a single item per page)
    # - in any mode, it will first try to assume each page is a listing, and if nothing is extracted, it will try to
    #   extract using item mode.
    # - in hybrid mode, first listing extraction is applied. If the items has an item url, or item_url_template
    #   attribute is set, it is followed and the data is completed with the data extracted from the item page.
    # - In combined mode, extraction is tried both with listing and item mode, but in this case from the same page
    #   (not additional request) and then combined. This mode is intended to combine extraction from more than one
    #   table into the same result (by default, only the best scored listing table is used to extract the results,
    #   see max_tables parameter)
    # - in tiles mode, the target page consists of items with repeated fields, organized in tiles.
    extract_mode: Literal["listing", "item", "any", "hybrid", "combined", "tiles"] = "listing"

    # in hybrid mode, build item url by using a python format template that will
    # receive each result as parameters dict.
    item_url_template: str = ""

    # If not empty, it will reduce score of results that don't have any of the required
    # fields. By default, all fields provided in keywords argument are required.
    required_fields: Tuple[Keyword, ...] = ()

    # which fields are use to deduplicate results (results with all same values in the same fields are
    # considered the same
    dedupe_keywords: Tuple[Keyword, ...] = ()

    # A dict (field -> list of value regexes)
    # filter out given fields with value regexes with any of the given list of values.
    # Applies during items pre processing, so it affects extraction process by reducing the score of candidates
    # with values matching the values.
    value_filters: Dict[Keyword, Tuple[str, ...]] | None = None

    # A mapping (field -> regexes) for additional extraction capabilities in item extraction mode.
    # regexes is a list. Each element in regexes is used in the response.text_re() function provided by
    # ExtractTextResponse. For each field, you can provide a list of alternatives. The first one extracting something
    # will be the value used. Remaining ones will be discarded.
    # Each element in regexes can be either a regex string, or a tuple where first element is a regex or None (for
    # default regex) and the second regex the value of parameter tid (see emodels ExtractResponse.text_re docstring)
    additional_regexes: Dict[Keyword, Tuple[str | Tuple[str | None, str], ...]] | None = None

    # in tiles/item mode, a tuple of fields without explicit keyword in target markdown, but yet can be found between
    # text with explicit keywords. This is used to put all text between keyword extracted fields. This must be
    # a keyword present in the keywords tuple, despite is not strictly a keyword, but yet is necesary to understand
    # the order between the keywords extracted texts.
    fill_fields: Tuple[Keyword, ...] = ()

    # In tiles/item mode, by default, and in order to avoid to extract noise, the algorithm only extracts text in the
    # same line as the keyword. This is ok for 99% of cases. With this option, you can specify a max number of lines
    # to extract for specific fields. For example, if you specify {"address": 4}, the algorithm will extract up to 4
    # lines of text for the "address" field.
    multiline_fields: Optional[Dict[Keyword, int]] = None

    # A map field->pattern that field value must fit. Otherwise the field is removed.
    # pattern is either a regex or a type keyword. Actually supported keywords: date_type, url_type
    # Don't change constraints directly unless you know what you are doing. Just use constraints_overrides
    #       for adding new constraints.
    constraints: Constraints = Constraints(
        {
            Keyword("isin"): re.compile(r"^[a-z0-9]{12}$", re.I),
            Keyword("website"): "url_type",
        }
    )
    constraints_overrides: Optional[Dict[Keyword, str]] = None
    # for listing table extraction, increase if target data are on more than one table with different headers in the
    # same page. The limit avoids to generate noisy data, so only the table with biggest score is considered, but in
    # some cases more than one table requires to be consireded for getting all the relevant data, more typically
    # in "combined" extract mode
    max_tables: int = 1

    # if True, it will generate a markdown file for each visited page with the markdown text extracted by
    # ExtractTextResponse. This is specially useful for debugging extraction results and writing additional
    # regexes. It will also activate debug mode on extract_by_keywords, in order to debug the item extraction process.
    debug_mode: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen_results: Set[Uid] = set()
        self.visited_pages: Set = set()
        for attr in (
            "fields",
            "required_fields",
            "value_filters",
            "additional_regexes",
            "fill_fields",
            "drop_fields",
            "drop_items",
            "constraints_overrides",
        ):
            if isinstance(getattr(self, attr), str):
                setattr(self, attr, json.loads(getattr(self, attr)))
        if isinstance(self.max_tables, str):
            self.max_tables = int(self.max_tables)
        if isinstance(self.fill_fields, list):
            self.fill_fields = tuple(self.fill_fields)
        self.constraints = self.override_constraints(self.constraints, self.constraints_overrides)
        self.dedupe_keywords = self.dedupe_keywords or self.fields
        self.markdown_count = 0

    def make_request(self, url, callback, cb_kwargs=None, **kwargs) -> Request:
        return Request(url, callback=callback, cb_kwargs=cb_kwargs, **kwargs)

    @staticmethod
    def override_constraints(constraints, overrides) -> Constraints:
        constraints = Constraints(constraints.copy())
        for k, v in (overrides or {}).items():
            if v in ("date_type", "url_type"):
                constraints[k] = v
            else:
                constraints[k] = re.compile(v)
        return constraints

    @abstractmethod
    def validate_result(self, result: Result, columns: Tuple[Keyword, ...]) -> bool:
        """
        Validates a result. If False, the result is not accepted.
        Override conveniently.
        """

    def extract_items_from_page(self, response: TextResponse) -> Iterable[Result | Request]:
        """
        Extract items from a page by selecting the chosen algorithm according to extract_mode parameter.
        """
        response = response.replace(cls=ExtractTextResponse)
        if self.debug_mode:
            open(f"markdown{self.markdown_count}.md", "w").write(response.markdown)
            self.markdown_count += 1
        has_results = False
        if self.extract_mode in ("any", "listing"):
            for result in self.extract_listing(response):
                yield result
                has_results = True
        if self.extract_mode in ("any", "item", "tiles") and not has_results:
            for result in self.parse_items_from_response(response):
                yield result
        if self.extract_mode == "hybrid":
            for result in self.extract_listing(response):
                if self.item_url_template:
                    try:
                        url = self.item_url_template.format(**{str(k): v for k, v in result.items()})
                        if url:
                            result[Keyword("url")] = Text(url)
                    except Exception as e:
                        self.logger.warning(f"Cannot build item url: {e!r} from '{result}'")
                if "url" in result:
                    yield self.make_request(
                        url=result[Keyword("url")], callback=self.parse_items_from_response, cb_kwargs=result
                    )
        if self.extract_mode == "combined":
            result = parse_combined_from_response(
                response=response,
                columns=self.fields,
                validate_result=self.validate_result,
                dedupe_keywords=self.dedupe_keywords,
                constraints=self.constraints,
                value_filters=self.value_filters,
                debug_mode=self.debug_mode,
            )
            if result:
                uid = unique_id(result, self.dedupe_keywords)[0]
                if uid in self.seen_results:
                    self.logger.warning(f"Duplicate result found with uid {uid}, skipping: {result}")
                    return
                self.seen_results.add(uid)
                yield result

    @abstractmethod
    def adapt_result(self, result: Result, response: TextResponse):
        """
        Perform any adaptation on result data, like changing field names, post processing values, dropping
        fields according to some specific logic, etc.
        """

    def _adapt_result(self, result: Result, response: TextResponse):
        self.adapt_result(result, response)
        for k in list(result.keys()):
            if m := MARKDOWN_URL_RE.match(result[k]):
                result[k] = Text(m.groups()[0])
            if result[k] == "":
                result.pop(k)
            if k in result:
                result[Keyword(k.strip("*| \\"))] = Text(MULTIS_RE.sub(" ", result.pop(k)))
        if "url" in result:
            result[Keyword("url")] = Text(response.urljoin(result[Keyword("url")]))
        for field in self.drop_fields:
            result.pop(field, None)

    def extract_listing(self, response: TextResponse, **kwargs) -> Iterable[Result]:
        response = response.replace(cls=ExtractTextResponse)
        for result in parse_tables_from_response(
            response,
            columns=self.fields,
            validate_result=self.validate_result,
            dedupe_keywords=self.dedupe_keywords,
            constraints=self.constraints,
            required_fields=self.required_fields,
            max_tables=self.max_tables
        ):
            result.update(cast(Dict[Keyword, Text], kwargs))
            apply_additional_regexes(self.additional_regexes, result, response)
            self._adapt_result(result, response)
            if self.extract_mode != "hybrid":
                uid = unique_id(result, self.dedupe_keywords)[0]
                if uid in self.seen_results:
                    self.logger.warning(f"Duplicate result found with uid {uid}, skipping: {result}")
                    continue
                self.seen_results.add(uid)
            yield result

    def parse_items_from_response(self, response: TextResponse, **kwargs) -> Iterable[Result]:
        response = response.replace(cls=ExtractTextResponse)

        results = extract_by_keywords(
            response,
            keywords=self.fields,
            required_fields=self.required_fields,
            value_filters=self.value_filters,
            value_presets=cast(Dict[Keyword, Text], kwargs),
            constraints=self.constraints,
            tiles_mode=self.extract_mode == "tiles",
            additional_regexes=self.additional_regexes,
            fill_fields=self.fill_fields,
            multiline_fields=self.multiline_fields,
            debug_mode=self.debug_mode,
        )
        for result in results:
            self._adapt_result(result, response)
            for field, regexes in self.drop_items.items():
                for regex in regexes:
                    if re.search(regex, result.get(field, "")):
                        continue
            uid = unique_id(result, self.dedupe_keywords)[0]
            if uid in self.seen_results:
                self.logger.warning(f"Duplicate result found with uid {uid}, skipping: {result}")
                continue
            self.seen_results.add(uid)
            yield result
