import os
import re
from unittest import TestCase
from typing import Tuple

from scrapy.http import TextResponse

from emodels.extract.table import parse_tables_from_response
from emodels.extract.utils import Constraints, Keyword, Result


NUMBER_RE = re.compile(r"\d+$")
TEST_TABLE_COLUMNS = (
        Keyword("address"),
        Keyword("code"),
        Keyword("company"),
        Keyword("company name"),
        Keyword("fullname"),
        Keyword("industry group"),
        Keyword("instrument"),
        Keyword("isin"),
        Keyword("isin code"),
        Keyword("issuer"),
        Keyword("issuer code"),
        Keyword("issuer name"),
        Keyword("lei"),
        Keyword("market"),
        Keyword("name"),
        Keyword("name isin"),
        Keyword("open price"),
        Keyword("open"),
        Keyword("sector"),
        Keyword("share code"),
        Keyword("stock symbol"),
        Keyword("symbol"),
        Keyword("ticker"),
)
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


class TableExtractTests(TestCase):
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
                    "url": "http://example.com/en/company_historical/ARBK",
                },
            )
            self.assertEqual(results[13][Keyword("url")], "http://example.com/en/company_historical/ARBK")
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
                    "url": "http://example.com/en/company_historical/PEDC",
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
                    "url": "http://example.com/en/company_historical/ULDC",
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
                    "url": "http://example.com/statistics/en/instrument/LT0000101446/trading",
                    "volume": "2,431",
                    "website": (
                        "http://lt.morningstar.com/gj8uge2g9k/stockprofile/default.aspx?"
                        "externalid=LT0000101446&externalidexchange=EX$$$$XLIT&externalidtype"
                        "=ISIN&LanguageId=en-GB&CurrencyId=EUR"
                    ),
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
        with self.open_resource("test24.html") as f:
            response = TextResponse(
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
            )
            results = parse_tables_from_response(
                response,
                columns=columns,
                validate_result=validate_result,
                dedupe_keywords=DEDUPE_KEYWORDS,
                constraints=Constraints(
                    {Keyword("listing date"): "date_type", Keyword("isin"): re.compile(r"^[a-z0-9]{12}$", re.I)}
                ),
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

    def test_table_x(self):
        with self.open_resource("test28.html") as f:
            response = TextResponse(url="https://www.mse.mk/en/issuers/shares-listing", status=200, body=f.read())
            columns = (
                    Keyword("name"),
                    Keyword("business"),
                    Keyword("address"),
                    Keyword("city"),
                    Keyword("state"),
                    Keyword("phone"),
                    Keyword("site"),
            )
            results = parse_tables_from_response(response, columns=columns, validate_result=validate_result)
            self.assertEqual(len(results), 90)
            self.assertEqual(
                results[34],
                {
                    "address": "ul. 808 br. 8",
                    "business": "Industry",
                    "city": "Skopje",
                    "name": "Evropa AD Skopje",
                    "phone": "+389 2 3114 066",
                    "site": "Link to site",
                    "url": "https://www.mse.mk/en/issuer/evropa-ad-skopje/",
                    "website": "http://www.evropa.com.mk",
                },
            )

    def test_table_xi(self):
        with self.open_resource("test29.html") as f:
            response = TextResponse(url="https://www.bse-sofia.bg/en/listed-instruments", status=200, body=f.read())
            columns = (Keyword("code"), Keyword("lei"), Keyword("name"))
            results = parse_tables_from_response(
                response,
                columns=columns,
                constraints=Constraints(
                    {
                        Keyword("website"): re.compile(
                            r"^(https?://.+?)|(<https?://.+?>)|(\[.+\]\(https?://.+\))|(www\..+\..+)"
                        )
                    }
                ),
            )
            self.assertEqual(len(results), 420)
            self.assertEqual(
                results[23],
                {
                    "code": "EAC",
                    "currency": "EUR",
                    "lei": "213800A3AEHGOGT4KM74",
                    "name": "Elana Agrocredit AD",
                    "nominal": "0.5100",
                    "total volume of the issue": "46 692 133",
                },
            )

    def test_table_xii(self):
        with self.open_resource("test31.html") as f:
            response = TextResponse(url="https://www.egx.com.eg/en/listedstocks.aspx", status=200, body=f.read())
            columns = (Keyword("company name"), Keyword("isin code"), Keyword("sector"))
            results = parse_tables_from_response(
                response,
                columns=columns,
                constraints=Constraints({Keyword("isin code"): re.compile(r"^[a-z0-9]{12}$", re.I)}),
                required_fields=(Keyword("company name"), Keyword("isin code")),
            )
            self.assertEqual(len(results), 240)
            self.assertEqual(
                results[55],
                {
                    "company name": "Delta Insurance",
                    "isin code": "EGS63031C016",
                    "sector": "Non-bank financial services",
                    "url": "https://www.egx.com.eg/en/CompanyDetails.aspx?ISIN=EGS63031C016",
                },
            )
