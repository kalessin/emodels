import os
import re
from unittest import TestCase
from typing import Dict

from scrapy.http import TextResponse

from emodels.extract.table import parse_tables_from_response, Columns
from emodels.extract.utils import Constraints


NUMBER_RE = re.compile(r"\d+$")
TEST_TABLE_COLUMNS = Columns(
    (
        "address",
        "code",
        "company",
        "company name",
        "fullname",
        "industry group",
        "instrument",
        "isin",
        "isin code",
        "issuer",
        "issuer code",
        "issuer name",
        "lei",
        "market",
        "name",
        "name isin",
        "open price",
        "open",
        "sector",
        "share code",
        "stock symbol",
        "symbol",
        "ticker",
    )
)
DEDUPE_KEYWORDS = Columns(
    (
        "code",
        "company",
        "company name",
        "isin",
        "isin code",
        "issuer code",
        "issuer name",
        "lei",
        "name",
        "name isin",
        "phone",
        "share code",
        "symbol",
        "ticker",
    )
)


def validate_result(result: Dict[str, str], candidate_fields: Columns = TEST_TABLE_COLUMNS) -> bool:
    score = 0
    for field in candidate_fields:
        if field in result:
            score += 1
    if score < 2:
        return False
    result.pop("", None)
    if issuer := result.get("issuer"):
        m = NUMBER_RE.match(issuer)
        if m:
            return False
    return True


class TableSpiderTests(TestCase):
    def open_resource(self, name):
        rname = os.path.join(os.path.dirname(__file__), "extract_resources", name)
        return open(rname, "rb")

    def test_table_i(self):
        with self.open_resource("test1.html") as f:
            response = TextResponse(url="http://example.com", status=200, body=f.read())
            results = parse_tables_from_response(response, columns=TEST_TABLE_COLUMNS, dedupe_keywords=DEDUPE_KEYWORDS)
            self.assertEqual(len(results), 115)
            for result in results:
                self.assertEqual(
                    list(result.keys()),
                    ["symbol", "isin", "name", "eng market group", "dual listed", "short sell", "trading"],
                )
            self.assertEqual(
                results[23],
                {
                    "dual listed": "No",
                    "eng market group": "Financials",
                    "isin": "AEN000801011",
                    "name": "National Bank of Fujairah",
                    "short sell": "Not Allowed",
                    "symbol": "NBF",
                    "trading": "Active",
                },
            )

    def test_table_ii(self):
        with self.open_resource("test2.html") as f:
            response = TextResponse(url="http://example.com", status=200, body=f.read())
            results = parse_tables_from_response(response, columns=TEST_TABLE_COLUMNS)
            self.assertEqual(len(results), 162)
            for result in results:
                self.assertEqual(
                    list(result.keys()),
                    [
                        "company's name",
                        "company's short name",
                        "symbol",
                        "code",
                        "market",
                        "listed shares",
                        "notes",
                        "url",
                    ],
                )
            self.assertEqual(
                results[13],
                {
                    "code": "113023",
                    "company's name": "ARAB BANK",
                    "company's short name": "ARAB BANK",
                    "listed shares": "640,800,000",
                    "market": "1",
                    "symbol": "ARBK",
                    "notes": "",
                    "url": "/en/company_historical/ARBK",
                },
            )
            self.assertEqual(results[13]["url"], "/en/company_historical/ARBK")
            self.assertEqual(
                results[96],
                {
                    "code": "131221",
                    "company's name": "PETRA EDUCATION COMPANY",
                    "company's short name": "PETRA EDUCATION",
                    "listed shares": "20,000,000",
                    "market": "1",
                    "symbol": "PEDC",
                    "notes": "",
                    "url": "/en/company_historical/PEDC",
                },
            )
            self.assertEqual(
                results[63],
                {
                    "company's name": "UNION LAND DEVELOPMENT CORPORATION",
                    "company's short name": "UNION LAND DEV",
                    "symbol": "ULDC",
                    "code": "131073",
                    "market": "2",
                    "listed shares": "42,065,129",
                    "notes": "OTC",
                    "url": "/en/company_historical/ULDC",
                },
            )

    def test_table_iii(self):
        with self.open_resource("test3.html") as f:
            response = TextResponse(url="http://example.com", status=200, body=f.read())
            results = parse_tables_from_response(response, columns=TEST_TABLE_COLUMNS)
            self.assertEqual(len(list(results)), 236)

    def test_table_iv(self):
        with self.open_resource("test4.html") as f:
            response = TextResponse(url="http://example.com", status=200, body=f.read())
            results = parse_tables_from_response(response, columns=TEST_TABLE_COLUMNS)
            # this should be really 38, but the detection of the missing result is challenging
            # as extracted values are not aligned with columns in this case
            self.assertEqual(len(list(results)), 37)

    def test_table_v(self):
        with self.open_resource("test5.html") as f:
            response = TextResponse(url="http://example.com", status=200, body=f.read())
            results = parse_tables_from_response(response, columns=TEST_TABLE_COLUMNS, dedupe_keywords=DEDUPE_KEYWORDS)
            self.assertEqual(len(list(results)), 12)
            self.assertEqual(
                results[1],
                {
                    "№": "2",
                    "name": 'S.V.M. "IUVENTUS-DS" S.A',
                    "address": (
                        "MD-2062, mun. Chisinau, bd. Dacia, 47/2 adresa poștală: MD2001, "
                        "Chisinau, bd. Stefan cel Mare, 65, of. 710"
                    ),
                    "phone": "(022)270035  (022)271337",
                    "email": "iuventus@iuventus.md",
                },
            )

    def test_table_vi(self):
        with self.open_resource("test6.html") as f:
            response = TextResponse(url="http://example.com", status=200, body=f.read())
            results = parse_tables_from_response(response, columns=TEST_TABLE_COLUMNS)
            self.assertEqual(len(list(results)), 70)
            self.assertEqual(
                results[23],
                {
                    "%": "-0.42",
                    "+/-": "-0.002",
                    "ask €": "0.478",
                    "bid €": "0.478",
                    "close €": "0.48",
                    "company": "Panevėžio statybos trestas",
                    "home  market": "VLN",
                    "ind.": "IND",
                    "last €": "0.478",
                    "sect.": "CON",
                    "ticker": "PTR1L",
                    "trades": "11",
                    "turnover €": "1,159",
                    "url": "/statistics/en/instrument/LT0000101446/trading",
                    "volume": "2,431",
                },
            )

    def test_table_vii(self):
        with self.open_resource("test7.html") as f:
            response = TextResponse(url="http://example.com", status=200, body=f.read())
            results = parse_tables_from_response(response, columns=TEST_TABLE_COLUMNS)
            self.assertEqual(len(list(results)), 50)

    def test_table_viii(self):
        with self.open_resource("test8.html") as f:
            response = TextResponse(url="http://example.com", status=200, body=f.read())
            results = parse_tables_from_response(response, columns=TEST_TABLE_COLUMNS, validate_result=validate_result)
            self.assertEqual(len(list(results)), 20)

    def test_table_ix(self):
        self.maxDiff = None
        with self.open_resource("test24.html") as f:
            response = TextResponse(url="http://example.com", status=200, body=f.read())
            columns = Columns(
                ("industry", "sector", "super-sector", "sub-sector", "code", "isin", "listing date", "security name")
            )
            results = parse_tables_from_response(
                response,
                columns=columns,
                validate_result=validate_result,
                dedupe_keywords=DEDUPE_KEYWORDS,
                constraints=Constraints({"listing date": "date_type", "isin": re.compile(r"^[a-z0-9]{12}$", re.I)}),
                max_tables=2,
            )
            self.assertEqual(
                results,
                [
                    {
                        "code": "TSH",
                        "face value": "0.340(EUR)",
                        "isin": "CY0006091512",
                        "listing date": "1/11/2000",
                        "no. of shares": "251,200,000",
                        "security name": "A. TSOKKOS HOTELS PUBLIC LTD",
                        "trading currency": "EUR",
                    },
                    {
                        "industry": "Consumer Discretionary",
                        "sector": "Travel and Leisure",
                        "sub-sector": "Hotels and Motels",
                        "super-sector": "Travel and Leisure",
                    },
                ],
            )
