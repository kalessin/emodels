import os
import re
from unittest import TestCase
from typing import Tuple

from emodels.scrapyutils.response import ExtractTextResponse
from emodels.extract.combined import parse_combined_from_response
from emodels.extract.utils import Constraints, Keyword, Result


NUMBER_RE = re.compile(r"\d+$")
DEDUPE_KEYWORDS = (
    Keyword("code"),
    Keyword("company"),
    Keyword("company name"),
    Keyword("isin"),
    Keyword("isin code"),
    Keyword("issuer code"),
    Keyword("issuer name"),
    Keyword("lei"),
    Keyword("name"),
    Keyword("name isin"),
    Keyword("phone"),
    Keyword("share code"),
    Keyword("symbol"),
    Keyword("ticker"),
)


def validate_result(result: Result, candidate_fields: Tuple[Keyword, ...]) -> bool:
    score = 0
    for field in candidate_fields:
        if field in result:
            score += 1
    if score < 2:
        return False
    result.pop(Keyword(""), None)
    if issuer := result.get(Keyword("issuer")):
        m = NUMBER_RE.match(issuer)
        if m:
            return False
    if Keyword("code") in result and len(result[Keyword("code")]) > 20:
        return False
    return True


class CombinedExtractionTests(TestCase):
    maxDiff = None

    def open_resource(self, name):
        rname = os.path.join(os.path.dirname(__file__), "extract_resources", name)
        return open(rname, "rb")

    def test_combined_i(self):
        with self.open_resource("test24.html") as f:
            response = ExtractTextResponse(
                url="https://www.cse.com.cy/en-GB/regulated-market/listing/listed-companies/", status=200, body=f.read()
            )
            columns = (
                Keyword("industry"),
                Keyword("sector"),
                Keyword("super-sector"),
                Keyword("sub-sector"),
                Keyword("code"),
                Keyword("isin"),
                Keyword("listing date"),
                Keyword("security name"),
                Keyword("address"),
                Keyword("web-site"),
            )
            result = parse_combined_from_response(
                response,
                columns=columns,
                validate_result=validate_result,
                dedupe_keywords=DEDUPE_KEYWORDS,
                constraints=Constraints(
                    {Keyword("listing date"): "date_type", Keyword("isin"): re.compile(r"^[a-z0-9]{12}$", re.I)}
                ),
            )
            self.assertEqual(
                result,
                {
                    "address": "Nissi Avenue 38, 5341 Ayia Napa",
                    "code": "TSH",
                    "face value": "0.340(EUR)",
                    "isin": "CY0006091512",
                    "listing date": "1/11/2000",
                    "no. of shares": "251,200,000",
                    "security name": "A. TSOKKOS HOTELS PUBLIC LTD",
                    "trading currency": "EUR",
                    "industry": "Consumer Discretionary",
                    "sector": "Travel and Leisure",
                    "sub-sector": "Hotels and Motels",
                    "super-sector": "Travel and Leisure",
                    "url": "https://www.cse.com.cy/en-GB/regulated-market/listing/listed-companies/",
                    "web-site": "www.tsokkos.com",
                },
            )

    def test_combined_ii(self):
        with self.open_resource("test38.html") as f:
            response = ExtractTextResponse(
                url="https://www2.jpx.co.jp/tseHpFront/JJK020010Action.do", status=200, body=f.read()
            )
            columns = (
                Keyword("^###"),
                Keyword("code"),
                Keyword("market segment"),
                Keyword("industry"),
                Keyword("isin code"),
                Keyword("date of incorporation"),
                Keyword("date of listing"),
                Keyword("address of main office"),
                Keyword("listed exchange"),
                Keyword("name of representative"),
            )
            result = parse_combined_from_response(
                response,
                columns=columns,
                validate_result=validate_result,
                dedupe_keywords=DEDUPE_KEYWORDS,
                constraints=Constraints({Keyword("isin code"): re.compile(r"^[a-z0-9]{12}$", re.I)}),
                max_tables=3,
                debug_mode=True,
            )
            self.assertEqual(
                result,
                {
                    "address of main office": "Tokyo",
                    "code": "13010",
                    "date of general shareholders meeting (scheduled)": "-",
                    "date of incorporation": "1937/09/03",
                    "date of listing": "1949/05/16",
                    "fiscal year-end": "March,31",
                    "industry": "Fishery, Agriculture & Forestry",
                    "investment unit as of the end of last month": "544,000",
                    "isin code": "JP3257200000",
                    "listed exchange": "Tokyo",
                    "market segment": "Prime",
                    "name of representative": "井上\u3000誠",
                    "title": "KYOKUYO CO.,LTD.",
                    "title of representative": "代表取締役社長",
                    "trading unit": "100",
                    "url": "https://www2.jpx.co.jp/tseHpFront/JJK020010Action.do",
                },
            )
