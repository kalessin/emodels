import os
import re
from unittest import TestCase
from typing import Dict

from scrapy.http import TextResponse

from emodels.scrapyutils.response import ExtractTextResponse
from emodels.extract.table import Columns
from emodels.extract.combined import parse_combined_from_response
from emodels.extract.utils import Constraints


NUMBER_RE = re.compile(r"\d+$")
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


def validate_result(result: Dict[str, str], candidate_fields: Columns) -> bool:
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
    if "code" in result and len(result["code"]) > 20:
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
                url="https://www.cse.com.cy/en-GB/regulated-market/listing/listed-companies/",
                status=200, body=f.read()
            )
            columns = Columns((
                "industry",
                "sector",
                "super-sector",
                "sub-sector",
                "code",
                "isin",
                "listing date",
                "security name",
                "address",
                "web-site"
            ))
            result = parse_combined_from_response(
                response,
                columns=columns,
                validate_result=validate_result,
                dedupe_keywords=DEDUPE_KEYWORDS,
                constraints=Constraints({"listing date": "date_type", "isin": re.compile(r"^[a-z0-9]{12}$", re.I)}),
            )
            self.assertEqual(
                result,
                    {
                        'address': 'Nissi Avenue 38, 5341 Ayia Napa',
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
                        'web-site': 'www.tsokkos.com',
                    },
            )
