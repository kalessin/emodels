import os
from unittest import TestCase

from emodels.scrapyutils.response import ExtractTextResponse
from emodels.extract.cluster import extract_by_keywords
from emodels.extract.utils import apply_additional_regexes


class ClusterExtractTests(TestCase):

    maxDiff = None

    def open_resource(self, name):
        rname = os.path.join(os.path.dirname(__file__), "extract_resources", name)
        return open(rname, "rb")

    def test_stock_quote_i(self):
        with self.open_resource("test9.html") as f:
            response = ExtractTextResponse(url="http://example.com", status=200, body=f.read())
            result = extract_by_keywords(response.markdown, keywords=("^#", "industry", "sector", "stock"))
            self.assertEqual(
                result,
                {
                    "industry": "Restaurants & Bars",
                    "sector": "Consumer Discretionary",
                    "stock": "AW",
                    "title": "A & W Food Services of Canada Inc.",
                },
            )

    def test_stock_quote_ii(self):
        with self.open_resource("test10.html") as f:
            response = ExtractTextResponse(url="http://example.com", status=200, body=f.read())
            result = extract_by_keywords(
                response.markdown, keywords=("^#",), value_presets={"stock": "HDGE"}
            )
            self.assertEqual(
                result,
                {
                    "stock": "HDGE",
                    "title": "Accelerate Absolute Return Fund",
                },
            )

    def test_stock_quote_iii(self):
        with self.open_resource("test11.html") as f:
            response = ExtractTextResponse(url="http://example.com", status=200, body=f.read())
            result = extract_by_keywords(
                response.markdown,
                keywords=("^#",),
                value_filters={"name": ("ETF 101", "https://", "Key Data"), "stock": ("https://",)},
                value_presets={"stock": "ATSX"},
            )
            self.assertEqual(
                result,
                {
                    "stock": "ATSX",
                    "title": "Accelerate Canadian Long Short Equity Fund",
                },
            )

    def test_stock_quote_iv(self):
        with self.open_resource("test12.html") as f:
            response = ExtractTextResponse(url="http://example.com", status=200, body=f.read())
            result = extract_by_keywords(
                response.markdown,
                keywords=("address", "isin", "listing date", "website", "^#"),
            )
            self.assertEqual(
                result,
                {
                    "address": "Nakawa Business Park, Block A, 4th Floor, P.O.Box 23552, Kampala",
                    "isin": "UG0000000071",
                    "listing date": "10/28/2003 - 17:15",
                    "title": "USE All Share Index (100@31.12.2001)",
                    "website": "<http://www.use.or.ug>",
                },
            )

    def test_stock_quote_v(self):
        with self.open_resource("test13.html") as f:
            response = ExtractTextResponse(url="http://example.com", status=200, body=f.read())
            result = extract_by_keywords(
                response.markdown,
                keywords=("address", "listing date", "^#"),
                value_filters={"address": ("P.O. Box 6771",)},
            )
            self.assertEqual(
                result,
                {
                    "address": "Kampala",
                    "listing date": "11/07/2023 - 10:52",
                    "title": "Airtel Uganda",
                },
            )

    def test_stock_quote_vi(self):
        with self.open_resource("test14.html") as f:
            response = ExtractTextResponse(url="http://www.ux.ua/en/issue.aspx?code=CEEN", status=200, body=f.read())
            result = extract_by_keywords(
                response.markdown,
                keywords=("name", "isin", "short name", "ticker", "trading as of", "type", "website"),
            )
            self.assertEqual(
                result,
                {
                    "name": "Centerenergo, PJSC, Common",
                    "isin": "UA4000079081",
                    "short name": "Centerenergo, PJSC",
                    "ticker": "CEEN",
                    "trading as of": "16.03.2009",
                    "type": "common stock",
                    "website": "[www.centrenergo.com](http://www.centrenergo.com)",
                },
            )

    def test_stock_quote_vii(self):
        with self.open_resource("test15.html") as f:
            response = ExtractTextResponse(url="http://example.com", status=200, body=f.read())
            result = extract_by_keywords(
                response.markdown,
                keywords=(
                    "name",
                    "abbreviation",
                    "address",
                    "date of first listing",
                    "sector",
                    "www",
                    "full name",
                    "company address",
                ),
            )
            self.assertEqual(
                result,
                {
                    "name": "POWSZECHNY ZAKŁAD UBEZPIECZEŃ SPÓŁKA AKCYJNA",
                    "abbreviation": "PZU",
                    "address": "RONDO IGNACEGO DASZYŃSKIEGO 4 00-843 WARSZAWA",
                    "company address": "RONDO IGNACEGO DASZYŃSKIEGO 4 00-843 WARSZAWA",
                    "date of first listing": "05.2010",
                    "full name": "POWSZECHNY ZAKŁAD UBEZPIECZEŃ SPÓŁKA AKCYJNA",
                    "sector": "insurance offices",
                    "www": "[www.pzu.pl](http://www.pzu.pl)",
                },
            )

    def test_stock_quote_viii(self):
        with self.open_resource("test16.html") as f:
            response = ExtractTextResponse(url="https://money.tmx.co/", status=200, body=f.read())
            result = extract_by_keywords(
                response.markdown,
                keywords=(
                    "activity",
                    "address",
                    "site url",
                    "listing in athex",
                    "sector / subsector",
                    "reference symbols",
                ),
            )
            self.assertEqual(
                result,
                {
                    "activity": (
                        "Production of, and trade in, combed cotton yarns and fabrics for shirt "
                        "manufacture and related activities."
                    ),
                    "address": "AG. GEORGIOU STR. 40-44 \nPostal Code: \nPEFKI",
                    "site url": "http://www.nafpaktos-yarns.gr",
                    "listing in athex": "Jul 8, 1996",
                    "reference symbols": '[NAYP](https://money.tmx.co/stock-snapshot/-/select-stock/122 "NAYP")',
                    "sector / subsector": "Basic Resources / Textile Products (Jul 1, 2019)",
                },
            )

    def test_stock_quote_ix(self):
        with self.open_resource("test17.html") as f:
            response = ExtractTextResponse(url="https://money.tmx.co/", status=200, body=f.read())
            result = extract_by_keywords(
                response.markdown,
                keywords=("^##", "listing type", "listing status", "\\*\\*listed", "available to"),
            )
            self.assertEqual(
                result,
                {
                    "\\*\\*listed": "21 Jun 2022",
                    "available to": "Qualified Investors",
                    "listing status": "Listed",
                    "listing type": "International Debt",
                    "title": "Acamar Films Limited - Secured Loan Note - Issue 035, 6.00% Notes " "Due April 14, 2027",
                },
            )

    def test_stock_quote_x(self):
        with self.open_resource("test18.html") as f:
            response = ExtractTextResponse(
                url="https://www.boerse-duesseldorf.de/aktien/DE000A2P4HL9/123fahrschule-se-inhaber-aktien-o-n/",
                status=200,
                body=f.read(),
            )
            result = extract_by_keywords(
                response.markdown,
                keywords=("^##", "wkn", "marktsegment", "erstnotierung", "wertpapiertyp"),
            )
            apply_additional_regexes({"isin": ("isin: \\*\\*(.+?)\\*\\*",)}, result, response)
            self.assertEqual(
                result,
                {
                    "erstnotierung": "13.10.2020",
                    "isin": "DE000A2P4HL9",
                    "marktsegment": "Primärmarkt",
                    "title": "123fahrschule SE Inhaber-Aktien o.N.",
                    "url": "https://www.boerse-duesseldorf.de/aktien/DE000A2P4HL9/123fahrschule-se-inhaber-aktien-o-n/",
                    "wertpapiertyp": "Stammaktien",
                    "wkn": "A2P4HL",
                },
            )

    def test_stock_quote_xi(self):
        with self.open_resource("test19.html") as f:
            response = ExtractTextResponse(
                url="https://www.bse.hu/pages/company_profile/$issuer/3439",
                status=200,
                body=f.read(),
            )
            result = extract_by_keywords(
                response.markdown,
                keywords=("full name", "short name", "sector", "Business activity", "contact"),
            )
            self.assertEqual(
                result,
                {
                    "Business activity": "",
                    "contact": "HU-1095 Budapest, Máriássy utca 7.\n"
                    "\n"
                    "Phone: +36-1-451-4760\n"
                    "\n"
                    "Fax: +36-1-451-4289\n"
                    "\n"
                    "Web: [www.wing.hu](http://www.wing.hu)",
                    "full name": "WINGHOLDING Ingatlanfejlesztő és Beruházó Zártkörűen Működő " "Részvénytársaság",
                    "sector": "",
                    "short name": "WINGHOLDING Zrt.",
                },
            )

    def test_stock_quote_xii(self):
        with self.open_resource("test20.html") as f:
            response = ExtractTextResponse(
                url="https://www.casablanca-bourse.com/en/live-market/emetteurs/AFI050112",
                status=200,
                body=f.read(),
            )
            result = extract_by_keywords(
                response.markdown,
                keywords=(
                    "company name",
                    "corporate address",
                    "external auditors",
                    "date of creation",
                    "date of ipo",
                    "length of fiscal year",
                    "social object",
                ),
            )
            self.assertEqual(
                result,
                {
                    "company name": "AFRIC INDUSTRIES SA",
                    "corporate address": "Zone Industrielle, Route de Tétouan, Lot 107, BP 368",
                    "date of creation": "17/12/1980",
                    "date of ipo": "05/01/2012",
                    "external auditors": "A & T Auditeurs Consultants / A.Saaidi & Associés",
                    "length of fiscal year": "12",
                    "social object": "* The development, production and marketing of abrasive "
                    "products of all shapes and contents.\n"
                    " * The manufacturing and sale of tapes and adhesive and "
                    "self-adhesive tapes.\n"
                    " * The manufacturing, assembly, glazing, installation and "
                    "marketing of all types of joinery and finished aluminum "
                    "products and other materials.\n"
                    " * The purchase, sale, import, export, manufacturing, "
                    "processing, assembly, installation laying of all "
                    "equipments, materials, tools, accessories, raw materials "
                    "and spare parts;",
                },
            )

    def test_stock_quote_xiii(self):
        with self.open_resource("test21.html") as f:
            response = ExtractTextResponse(
                url="https://www.csx.ky/companies/equity.asp?SecId=01510001",
                status=200,
                body=f.read(),
            )
            result = extract_by_keywords(
                response.markdown,
                keywords=(
                    "ticker",
                    "isin",
                    "listing type",
                    "company website",
                    "^###",
                ),
            )
            self.assertEqual(
                result,
                {
                    "company website": "<http://www.caymannational.com/>",
                    "isin": "KYG198141056",
                    "listing type": "Primary Listing on CSX",
                    "ticker": "CNC KY",
                    "title": "Cayman National Corporation Ltd.",
                },
            )

    def test_stock_quote_xiv(self):
        with self.open_resource("test22.html") as f:
            response = ExtractTextResponse(
                url="https://www.csx.ky/companies/equity.asp?SecId=01510001",
                status=200,
                body=f.read(),
            )
            result = extract_by_keywords(
                response.markdown,
                keywords=("company type", "listing date", "company overview", "^####"),
            )
            apply_additional_regexes({"ticker": ("^## ([A-Z]+)",)}, result, response)
            self.assertEqual(
                result,
                {
                    "company overview": "Alpha Dhabi Holding (ADX: ALPHADHABI) is one of the MENA "
                    "region's largest and fastest-growing listed investment "
                    "platforms, with a portfolio of more than 250 companies "
                    "and 95,000 employees, it connects investors to the "
                    "exceptional returns of a vibrant economy. ADH has a "
                    "portfolio of the leading Abu Dhabi-based companies that "
                    "are, or have the potential to become, regional and "
                    "global champions. Whether market leaders or the next "
                    "generation of home-grown companies, ADH builds scale, "
                    "creates synergies, and enables innovation, moving "
                    "quickly to add value to its portfolio. ADH offers "
                    "investors access to a diverse portfolio of premium "
                    "assets across eight primary pillars and geographies: "
                    "climate capital, real estate, healthcare, industries, "
                    "construction, hospitality, energy, and investments. ADH "
                    "has a global mindset and continuously looks to invest in "
                    "countries with a compelling vision for the future of "
                    "their economies and leverages its scale and agility to "
                    "capitalise on markets and investment opportunities to "
                    "drive value across the platform, expand its portfolio "
                    "and generate future alpha. ADH and its companies are "
                    "helping to drive forward the vision of Abu Dhabi and the "
                    "UAE. From capital market expansion to developing "
                    "national talent and advancing towards net-zero, ADH "
                    "proudly create value for the UAE.",
                    "company type": "Public",
                    "listing date": "26 Jun 2021",
                    "title": "Alpha Dhabi Holding PJSC",
                    "ticker": "ALPHADHABI",
                    "url": "https://www.csx.ky/companies/equity.asp?SecId=01510001",
                },
            )

    def test_stock_quote_xv(self):
        with self.open_resource("test23.html") as f:
            response = ExtractTextResponse(
                url="https://www.cse.com.bd/company/companydetails/AIL",
                status=200,
                body=f.read(),
            )
            result = extract_by_keywords(
                response.markdown,
                keywords=(
                    "trading code",
                    "scrip code",
                    "listing year",
                    "debut trade date",
                    "type of instrument",
                    "sector",
                    "market category",
                ),
            )
            apply_additional_regexes({"name": ((None, ".com_title"),)}, result, response)
            self.assertEqual(
                result,
                {
                    "debut trade date": "25 January, 2018",
                    "listing year": "2018",
                    "market category": "A",
                    "name": "ALIF INDUSTRIES LIMITED",
                    "scrip code": "12012",
                    "sector": "TEXTILES & CLOTHING",
                    "trading code": "AIL",
                    "type of instrument": "",
                    "url": "https://www.cse.com.bd/company/companydetails/AIL",
                },
            )
