import os
import re
from unittest import TestCase

from emodels.scrapyutils.response import ExtractTextResponse
from emodels.extract.cluster import extract_by_keywords
from emodels.extract.utils import Constraints, Keyword, Text


class ClusterExtractTests(TestCase):
    maxDiff = None

    def open_resource(self, name):
        rname = os.path.join(os.path.dirname(__file__), "extract_resources", name)
        return open(rname, "rb")

    def test_cluster_i(self):
        with self.open_resource("test9.html") as f:
            response = ExtractTextResponse(url="http://example.com", status=200, body=f.read())
            result = extract_by_keywords(
                response, keywords=(Keyword("^#"), Keyword("industry"), Keyword("sector"), Keyword("stock"))
            )
            self.assertEqual(
                result[0],
                {
                    "industry": "Restaurants & Bars",
                    "sector": "Consumer Discretionary",
                    "stock": "AW",
                    "title": "A & W Food Services of Canada Inc.",
                    "url": "http://example.com",
                },
            )

    def test_cluster_ii(self):
        with self.open_resource("test10.html") as f:
            response = ExtractTextResponse(url="http://example.com", status=200, body=f.read())
            result = extract_by_keywords(
                response, keywords=(Keyword("^#"),), value_presets={Keyword("stock"): Text("HDGE")}
            )
            self.assertEqual(
                result[0],
                {
                    "stock": "HDGE",
                    "title": "Accelerate Absolute Return Fund",
                    "url": "http://example.com",
                },
            )

    def test_cluster_iii(self):
        with self.open_resource("test11.html") as f:
            response = ExtractTextResponse(url="http://example.com", status=200, body=f.read())
            result = extract_by_keywords(
                response,
                keywords=(Keyword("^#"),),
                value_filters={
                    Keyword("name"): (Text("ETF 101"), Text("https://"), Text("Key Data")),
                    Keyword("stock"): (Text("https://"),),
                },
                value_presets={Keyword("stock"): Text("ATSX")},
            )
            self.assertEqual(
                result[0],
                {
                    "stock": "ATSX",
                    "title": "Accelerate Canadian Long Short Equity Fund",
                    "url": "http://example.com",
                },
            )

    def test_cluster_iv(self):
        with self.open_resource("test12.html") as f:
            response = ExtractTextResponse(url="http://example.com", status=200, body=f.read())
            result = extract_by_keywords(
                response,
                keywords=(
                    Keyword("address"),
                    Keyword("isin"),
                    Keyword("listing date"),
                    Keyword("website"),
                    Keyword("^#"),
                ),
                constraints=Constraints({Keyword("website"): "url_type"}),
            )
            self.assertEqual(
                result[0],
                {
                    "address": "Nakawa Business Park, Block A, 4th Floor, P.O.Box 23552, Kampala",
                    "isin": "UG0000000071",
                    "listing date": "10/28/2003 - 17:15",
                    "title": "USE All Share Index (100@31.12.2001)",
                    "url": "http://example.com",
                    "website": "<http://www.use.or.ug>",
                },
            )
            result = extract_by_keywords(
                response,
                keywords=(
                    Keyword("address"),
                    Keyword("isin"),
                    Keyword("listing date"),
                    Keyword("website"),
                    Keyword("^#"),
                ),
                constraints=Constraints({Keyword("website"): re.compile("aaa")}),
            )
            self.assertEqual(
                result[0],
                {
                    "address": "Nakawa Business Park, Block A, 4th Floor, P.O.Box 23552, Kampala",
                    "isin": "UG0000000071",
                    "listing date": "10/28/2003 - 17:15",
                    "title": "USE All Share Index (100@31.12.2001)",
                    "url": "http://example.com",
                },
            )

    def test_cluster_v(self):
        with self.open_resource("test13.html") as f:
            response = ExtractTextResponse(url="http://example.com", status=200, body=f.read())
            result = extract_by_keywords(
                response,
                keywords=(Keyword("address"), Keyword("listing date"), Keyword("^#")),
                value_filters={Keyword("address"): (Text("P.O. Box 6771"),)},
            )
            self.assertEqual(
                result[0],
                {
                    "address": "Kampala",
                    "listing date": "11/07/2023 - 10:52",
                    "title": "Airtel Uganda",
                    "url": "http://example.com",
                },
            )

    def test_cluster_vi(self):
        with self.open_resource("test14.html") as f:
            response = ExtractTextResponse(url="http://www.ux.ua/en/issue.aspx?code=CEEN", status=200, body=f.read())
            result = extract_by_keywords(
                response,
                keywords=(
                    Keyword("name"),
                    Keyword("isin"),
                    Keyword("short name"),
                    Keyword("ticker"),
                    Keyword("trading as of"),
                    Keyword("type"),
                    Keyword("website"),
                ),
            )
            self.assertEqual(
                result[0],
                {
                    "name": "Centerenergo, PJSC, Common",
                    "isin": "UA4000079081",
                    "short name": "Centerenergo, PJSC",
                    "ticker": "CEEN",
                    "trading as of": "16.03.2009",
                    "type": "common stock",
                    "url": "http://www.ux.ua/en/issue.aspx?code=CEEN",
                    "website": "[www.centrenergo.com](http://www.centrenergo.com)",
                },
            )

    def test_cluster_vii(self):
        with self.open_resource("test15.html") as f:
            response = ExtractTextResponse(url="http://example.com", status=200, body=f.read())
            result = extract_by_keywords(
                response,
                keywords=(
                    Keyword("name"),
                    Keyword("abbreviation"),
                    Keyword("date of first listing"),
                    Keyword("sector"),
                    Keyword("www"),
                    Keyword("full name"),
                    Keyword("company address"),
                ),
                debug_mode=True,
            )
            self.assertEqual(
                result[0],
                {
                    "name": "PZU",
                    "abbreviation": "PZU",
                    "company address": "RONDO IGNACEGO DASZYŃSKIEGO 4 00-843 WARSZAWA",
                    "date of first listing": "05.2010",
                    "full name": "POWSZECHNY ZAKŁAD UBEZPIECZEŃ SPÓŁKA AKCYJNA",
                    "sector": "insurance offices",
                    "url": "http://example.com",
                    "www": "[www.pzu.pl](http://www.pzu.pl)",
                },
            )

    def test_cluster_viii(self):
        with self.open_resource("test16.html") as f:
            response = ExtractTextResponse(url="https://money.tmx.co/", status=200, body=f.read())
            result = extract_by_keywords(
                response,
                keywords=(
                    Keyword("activity"),
                    Keyword("address"),
                    Keyword("site url"),
                    Keyword("listing in athex"),
                    Keyword("sector / subsector"),
                    Keyword("reference symbols"),
                ),
            )
            self.assertEqual(
                result[0],
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
                    "url": "https://money.tmx.co/",
                },
            )

    def test_cluster_ix(self):
        with self.open_resource("test17.html") as f:
            response = ExtractTextResponse(url="https://money.tmx.co/", status=200, body=f.read())
            result = extract_by_keywords(
                response,
                keywords=(
                    Keyword("^##"),
                    Keyword("listing type"),
                    Keyword("listing status"),
                    Keyword("\\*\\*listed"),
                    Keyword("available to"),
                ),
            )
            self.assertEqual(
                result[0],
                {
                    "\\*\\*listed": "21 Jun 2022",
                    "available to": "Qualified Investors",
                    "listing status": "Listed",
                    "listing type": "International Debt",
                    "title": "Acamar Films Limited - Secured Loan Note - Issue 035, 6.00% Notes " "Due April 14, 2027",
                    "url": "https://money.tmx.co/",
                },
            )

    def test_cluster_x(self):
        with self.open_resource("test18.html") as f:
            response = ExtractTextResponse(
                url="https://www.boerse-duesseldorf.de/aktien/DE000A2P4HL9/123fahrschule-se-inhaber-aktien-o-n/",
                status=200,
                body=f.read(),
            )
            result = extract_by_keywords(
                response,
                keywords=(
                    Keyword("^##"),
                    Keyword("wkn"),
                    Keyword("marktsegment"),
                    Keyword("erstnotierung"),
                    Keyword("wertpapiertyp"),
                ),
                additional_regexes={Keyword("isin"): ("isin: \\*\\*(.+?)\\*\\*",)},
            )
            self.assertEqual(
                result[0],
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

    def test_cluster_xi(self):
        with self.open_resource("test19.html") as f:
            response = ExtractTextResponse(
                url="https://www.bse.hu/pages/company_profile/$issuer/3439",
                status=200,
                body=f.read(),
            )
            result = extract_by_keywords(
                response,
                keywords=(
                    Keyword("full name"),
                    Keyword("short name"),
                    Keyword("sector"),
                    Keyword("Business activity"),
                    Keyword("contact"),
                ),
            )
            self.assertEqual(
                result[0],
                {
                    "contact": "HU-1095 Budapest, Máriássy utca 7.\n"
                    "\n"
                    "Phone: +36-1-451-4760\n"
                    "\n"
                    "Fax: +36-1-451-4289\n"
                    "\n"
                    "Web: [www.wing.hu](http://www.wing.hu)",
                    "full name": "WINGHOLDING Ingatlanfejlesztő és Beruházó Zártkörűen Működő " "Részvénytársaság",
                    "short name": "WINGHOLDING Zrt.",
                    "url": "https://www.bse.hu/pages/company_profile/$issuer/3439",
                },
            )

    def test_cluster_xii(self):
        with self.open_resource("test20.html") as f:
            response = ExtractTextResponse(
                url="https://www.casablanca-bourse.com/en/live-market/emetteurs/AFI050112",
                status=200,
                body=f.read(),
            )
            result = extract_by_keywords(
                response,
                keywords=(
                    Keyword("company name"),
                    Keyword("corporate address"),
                    Keyword("external auditors"),
                    Keyword("date of creation"),
                    Keyword("date of ipo"),
                    Keyword("length of fiscal year"),
                    Keyword("social object"),
                ),
            )
            self.assertEqual(
                result[0],
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
                    "url": "https://www.casablanca-bourse.com/en/live-market/emetteurs/AFI050112",
                },
            )

    def test_cluster_xiii(self):
        with self.open_resource("test21.html") as f:
            response = ExtractTextResponse(
                url="https://www.csx.ky/companies/equity.asp?SecId=01510001",
                status=200,
                body=f.read(),
            )
            result = extract_by_keywords(
                response,
                keywords=(
                    Keyword("ticker"),
                    Keyword("isin"),
                    Keyword("listing type"),
                    Keyword("company website"),
                    Keyword("^###"),
                ),
            )
            self.assertEqual(
                result[0],
                {
                    "company website": "<http://www.caymannational.com/>",
                    "isin": "KYG198141056",
                    "listing type": "Primary Listing on CSX",
                    "ticker": "CNC KY",
                    "title": "Cayman National Corporation Ltd.",
                    "url": "https://www.csx.ky/companies/equity.asp?SecId=01510001",
                },
            )

    def test_cluster_xiv(self):
        with self.open_resource("test22.html") as f:
            response = ExtractTextResponse(
                url="https://www.csx.ky/companies/equity.asp?SecId=01510001",
                status=200,
                body=f.read(),
            )
            result = extract_by_keywords(
                response,
                keywords=(
                    Keyword("company type"),
                    Keyword("listing date"),
                    Keyword("company overview"),
                    Keyword("^####"),
                ),
                additional_regexes={Keyword("ticker"): ("^## ([A-Z]+)",)},
            )
            self.assertEqual(
                result[0],
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

    def test_cluster_xv(self):
        with self.open_resource("test23.html") as f:
            response = ExtractTextResponse(
                url="https://www.cse.com.bd/company/companydetails/AIL",
                status=200,
                body=f.read(),
            )
            result = extract_by_keywords(
                response,
                keywords=(
                    Keyword("trading code"),
                    Keyword("scrip code"),
                    Keyword("listing year"),
                    Keyword("debut trade date"),
                    Keyword("type of instrument"),
                    Keyword("sector"),
                    Keyword("market category"),
                ),
                additional_regexes={Keyword("name"): ((None, ".com_title"),)},
            )
            self.assertEqual(
                result[0],
                {
                    "debut trade date": "25 January, 2018",
                    "listing year": "2018",
                    "market category": "A",
                    "name": "ALIF INDUSTRIES LIMITED",
                    "scrip code": "12012",
                    "sector": "TEXTILES & CLOTHING",
                    "trading code": "AIL",
                    "url": "https://www.cse.com.bd/company/companydetails/AIL",
                },
            )

    def test_cluster_xvi(self):
        with self.open_resource("test25.html") as f:
            response = ExtractTextResponse(
                url="https://www.dsebd.org/displayCompany.php?name=AAMRANET",
                status=200,
                body=f.read(),
            )
            result = extract_by_keywords(
                response,
                keywords=(
                    Keyword("type of instrument"),
                    Keyword("debut trading date"),
                    Keyword("trading code"),
                    Keyword("scrip code"),
                    Keyword("web address"),
                    Keyword("sector"),
                ),
                debug_mode=True,
            )
            self.assertEqual(
                result[0],
                {
                    "debut trading date": "02 Oct, 2017",
                    "scrip code": "22649",
                    "sector": "IT Sector",
                    "trading code": "AAMRANET",
                    "type of instrument": "Equity",
                    "url": "https://www.dsebd.org/displayCompany.php?name=AAMRANET",
                    "web address": "[ http://www.aamra.com.bd](http://www.aamra.com.bd)",
                },
            )

    def test_cluster_xvii(self):
        with self.open_resource("test26.html") as f:
            response = ExtractTextResponse(
                url="https://www.ecseonline.com/profiles/GPCL/?type=equities",
                status=200,
                body=f.read(),
            )
            value_presets = {
                Keyword("company name"): Text("Grenreal Property Corporation Ltd."),
                Keyword("isin"): Text("GD3456401067"),
                Keyword("ticker"): Text("GPCL"),
            }
            result = extract_by_keywords(
                response,
                keywords=(Keyword("symbol"), Keyword("company name"), Keyword("isin"), Keyword("website")),
                value_presets=value_presets,
            )
            # nothing new was extracted
            expected = value_presets.copy()
            expected[Keyword("url")] = Text("https://www.ecseonline.com/profiles/GPCL/?type=equities")
            self.assertEqual(result[0], expected)

    def test_cluster_xviii(self):
        with self.open_resource("test34.html") as f:
            response = ExtractTextResponse(
                url="https://www.msx.om/snapshot.aspx?s=CMII",
                status=200,
                body=f.read(),
            )
            result = extract_by_keywords(
                response,
                keywords=(
                    Keyword("^#####"),
                    Keyword("activity"),
                    Keyword("commercial id"),
                    Keyword("isin"),
                    Keyword("established in"),
                    Keyword("listed date"),
                    Keyword("subsector"),
                    Keyword("representative"),
                    Keyword("address"),
                    Keyword("website"),
                ),
            )
            self.assertEqual(
                result[0],
                {
                    "activity": "MINING OF LIMESTONE AND MANUFACTURING LIMESTONE",
                    "address": "P O BOX 36 POSTAL CODE 327 SOHAR SULTANATE OF OMAN",
                    "commercial id": "1/05292/6",
                    "established in": "Jun 15, 1977",
                    "isin": "OM0000001368",
                    "listed date": "Jan 21, 2002",
                    "representative": "Talal Naser Oqlah",
                    "subsector": "Constructions Materials Support",
                    "title": (
                        "CONSTRUCTION MATERIAL INDUSTRIES (CMII) "
                        "![](https://www.msx.om/MSMDocs/Images/Companies/Logo-136-18102023.JPG)"
                    ),
                    "url": "https://www.msx.om/snapshot.aspx?s=CMII",
                    "website": "<http://cmioman.com>",
                },
            )

    def test_cluster_xix(self):
        with self.open_resource("test35.html") as f:
            response = ExtractTextResponse(
                url="https://www.msx.om/snapshot.aspx?s=OGWF",
                status=200,
                body=f.read(),
            )
            result = extract_by_keywords(
                response,
                keywords=(
                    Keyword("^#####"),
                    Keyword("activity"),
                    Keyword("commercial id"),
                    Keyword("isin"),
                    Keyword("established in"),
                    Keyword("listed date"),
                    Keyword("subsector"),
                    Keyword("representative"),
                    Keyword("address"),
                    Keyword("website"),
                ),
                value_filters={Keyword("address"): ("telephone", "registration profile")},
                constraints=Constraints(
                    {
                        Keyword("website"): re.compile(
                            r"^(https?://.+?)|(<https?://.+?>)|(\\[.+\\]\\(https?://.+\\))|(www\\..+\\..+)"
                        )
                    }
                ),
            )
            self.assertEqual(
                result[0],
                {
                    "established in": "Mar 02, 2026",
                    "isin": "OM0000010823",
                    "listed date": "-",
                    "title": "OMAN GATEWAY FUND (OGWF)",
                    "url": "https://www.msx.om/snapshot.aspx?s=OGWF",
                },
            )

    def test_cluster_xx(self):
        with self.open_resource("test36.html") as f:
            response = ExtractTextResponse(
                url="https://www.msx.om/snapshot.aspx?s=AMII",
                status=200,
                body=f.read(),
            )
            result = extract_by_keywords(
                response,
                keywords=(
                    Keyword("^#####"),
                    Keyword("activity"),
                    Keyword("commercial id"),
                    Keyword("isin"),
                    Keyword("established in"),
                    Keyword("listed date"),
                    Keyword("subsector"),
                    Keyword("representative"),
                    Keyword("company contact address"),
                    Keyword("website"),
                ),
                constraints=Constraints(
                    {
                        Keyword("website"): re.compile(
                            r"^(https?://.+?)|(<https?://.+?>)|(\\[.+\\]\\(https?://.+\\))|(www\\..+\\..+)"
                        )
                    }
                ),
                debug_mode=True,
            )
            self.assertEqual(
                result[0],
                {
                    "activity": "activities for holding companies",
                    "commercial id": "1010000",
                    "company contact address": "Al Madina Investment Holding Co. (SAOG), Tilal "
                    "Offices, Block 6,4th floor, Muscat Grand Mall P.O "
                    "B",
                    "established in": "Mar 10, 1996",
                    "isin": "OM0000001962",
                    "listed date": "Nov 12, 2002",
                    "representative": "Mansoor Nasser Al Hatmi",
                    "subsector": "Investment",
                    "title": "AL MADINA INVESTMENT HOLDING (AMII) "
                    "![](https://www.msx.om/MSMDocs/Images/Companies/Logo-196-08112022.JPG)",
                    "url": "https://www.msx.om/snapshot.aspx?s=AMII",
                },
            )

    def test_cluster_xxi(self):
        with self.open_resource("test37.html") as f:
            response = ExtractTextResponse(
                url="https://www.nsx.com.au/marketdata/company-directory/details/218/",
                status=200,
                body=f.read(),
            )
            result = extract_by_keywords(
                response,
                keywords=(
                    Keyword("^#"),
                    Keyword("acn/arbn"),
                    Keyword("abn"),
                    Keyword("nsx code"),
                    Keyword("nsx listed securities"),
                    Keyword("listing date"),
                    Keyword("principal activities"),
                    Keyword("industry class"),
                    Keyword("street address"),
                    Keyword("company base"),
                    Keyword("web"),
                    Keyword("isin"),
                    Keyword("figi"),
                    Keyword("bloomberg ticker"),
                    Keyword("security description"),
                    Keyword("security type"),
                    Keyword("certificated"),
                    Keyword("settlement"),
                    Keyword("security status"),
                ),
                multiline_fields={Keyword("street address"): 4},
            )
            self.assertEqual(
                result[0],
                {
                    "abn": "42 635 120 517",
                    "acn/arbn": "635 120 517",
                    "bloomberg ticker": "218 AO",
                    "certificated": "0 - Uncertificated",
                    "company base": "Australia",
                    "figi": "BBG00WZ8Y0P8",
                    "industry class": "Consumer Discretionary",
                    "isin": "AU0000101248",
                    "listing date": "Monday, August 31, 2020",
                    "nsx code": "218",
                    "nsx listed securities": (
                        "[218](https://www.nsx.com.au/marketdata/company-directory/218/>) "
                        "\\- Rofina Group Limited - FPO"
                    ),
                    "principal activities": "Travel Goods and Apparel",
                    "security description": "Rofina Group Limited - FPO",
                    "security status": "Active",
                    "security type": "01 - Ordinary",
                    "settlement": "Chess T+2",
                    "street address": (
                        "C/- Boardroom Pty Limited Level 8 \n" "210 George Street \n" "Sydney - NSW Australia 2000"
                    ),
                    "title": "Rofina Group Limited - FPO",
                    "url": "https://www.nsx.com.au/marketdata/company-directory/details/218/",
                    "web": "<https://www.rofinagroup.com/>",
                },
            )

    def test_cluster_tile_i(self):
        with self.open_resource("test27.html") as f:
            response = ExtractTextResponse(
                url="https://www.ese.co.sz/issuers/securities/",
                status=200,
                body=f.read(),
            )
            result_list = extract_by_keywords(
                response,
                keywords=(Keyword("isin"), Keyword("ticker"), Keyword("founded"), Keyword("listed")),
                tiles_mode=True,
            )
            self.assertEqual(
                result_list,
                [
                    {"founded": "1838", "isin": "SZE000331064", "listed": "5th December, 2023", "ticker": "FNBE"},
                    {"founded": "2009", "isin": "SZE000331023", "listed": "1st November, 2010", "ticker": "GRYS"},
                    {"founded": "2017", "isin": "SZE000331049", "listed": "1st January, 2019", "ticker": "INALA"},
                    {"founded": "1882", "isin": "SZ0005797904", "listed": "1st January, 1990", "ticker": "NED"},
                    {"founded": "2007", "isin": "SZE000331056", "listed": "9th November, 2023", "ticker": "NPC"},
                    {"founded": "1973", "isin": "SZ0005797920", "listed": "1st January, 1992", "ticker": "RSC"},
                    {"founded": "2011", "isin": "SZE000331031", "listed": "10th February, 2014", "ticker": "SBC"},
                    {"founded": "1998", "isin": "SZE000331015", "listed": "1st June, 2004", "ticker": "SEL"},
                    {"founded": "1996", "isin": "SZ0005797946", "listed": "1st January, 1999", "ticker": "SWP"},
                ],
            )

    def test_cluster_tile_ii(self):
        with self.open_resource("test30.html") as f:
            response = ExtractTextResponse(
                url="https://bse.co.bw/companies/",
                status=200,
                body=f.read(),
            )
            results = extract_by_keywords(
                response,
                keywords=(
                    Keyword("counter"),
                    Keyword("physical and postal address"),
                    Keyword("sector"),
                    Keyword("board"),
                ),
                tiles_mode=True,
                additional_regexes={Keyword("name"): ((None, ".lvca-panel-title"),)},
            )
            self.assertEqual(
                results,
                [
                    {
                        "counter": "ACCESS",
                        "name": "Access Bank Botswana Limited",
                        "sector": "Banking",
                        "board": "Domestic Main Board",
                        "physical and postal address": "Access House, Plot 62433, Fairgrounds Gaborone, Botswana",
                    },
                    {
                        "counter": "BOTALA",
                        "name": "Botala Energy Limited",
                        "physical and postal address": "24 Hasler Road, Osborne Park WA 6017, Australia",
                        "sector": "Mining",
                        "board": "Foreign Main Board",
                    },
                    {
                        "counter": "ABSA",
                        "name": "ABSA Bank of Botswana Limited",
                        "physical and postal address": (
                            "5th Floor, Building 4 Plaza Plot 74358 Gaborone, "
                            "Central Business District, \nP O Box 478 Gaborone Botswana,"
                        ),
                        "sector": "Banking",
                        "board": "Domestic Main Board",
                    },
                    {
                        "counter": "BIHL",
                        "name": "Botswana Insurance Holdings Limited",
                        "physical and postal address": (
                            "Plot 66458, Fairgrounds Office Park \n" "3rd Floor, Block A, \nP O Box 336 Gaborone,"
                        ),
                        "sector": "Financial Services",
                        "board": "Domestic Main Board",
                    },
                    {
                        "counter": "BTCL",
                        "name": "Botswana Telecommunications Corporation Limited",
                        "physical and postal address": (
                            "Plot 50350, Megaleng House \nKhama Crescent \n"
                            "Gaborone, Botswana \n(P.O. Box 700, Gaborone, Botswana),"
                        ),
                        "sector": "ICT",
                        "board": "Domestic Main Board",
                    },
                    {
                        "counter": "CHOBE",
                        "name": "Chobe Holdings Limited",
                        "physical and postal address": "Private Bag 198, \nMaun,Botswana",
                        "sector": "Tourism",
                        "board": "Domestic Main Board",
                    },
                    {
                        "counter": "CHOPPIES",
                        "name": "Choppies Enterprises Limited",
                        "physical and postal address": (
                            "Plot 46, \nGaborone International Commerce Park, \n"
                            "Gaborone, Botswana, \nPrivate Bag 00278, \nGaborone."
                        ),
                        "sector": "Retail & Wholesale",
                        "board": "Domestic Main Board",
                    },
                    {
                        "counter": "CRESTA",
                        "name": "Cresta Marakanelo Limited",
                        "physical and postal address": (
                            "2nd Floor, Marula House Prime Plaza Plot 74538 " "New CBD Gaborone"
                        ),
                        "sector": "Tourism",
                        "board": "Domestic Main Board",
                    },
                    {
                        "counter": "ENGEN",
                        "name": "Engen Botswana Limited",
                        "physical and postal address": "Plot 54026, \nWestern Bypass, \nGaborone.",
                        "sector": "Energy",
                        "board": "Domestic Main Board",
                    },
                    {
                        "counter": "FNBB",
                        "name": "First National Bank Botswana Limited",
                        "physical and postal address": "4th Floor First Place, \nPlot 54362 Gaborone CBD, \nBotswana",
                        "sector": "Banking",
                        "board": "Domestic Main Board",
                    },
                    {
                        "counter": "FPC",
                        "name": "The Far Property Company Limited",
                        "physical and postal address": (
                            "Plot 46, Gaborone International Commerce Park, " "\nGaborone, \nBotswana"
                        ),
                        "sector": "Property",
                        "board": "Domestic Main Board",
                    },
                    {
                        "counter": "G4S",
                        "name": "G4S Botswana Limited",
                        "physical and postal address": "Western Bypass, \nGaborone, \nBotswana",
                        "sector": "Security Services",
                        "board": "Domestic Main Board",
                    },
                    {
                        "counter": "LETLOLE",
                        "name": "Letlole La Rona Limited",
                        "physical and postal address": (
                            "1st Floor, Unit 2B, Peelo Place, Plot 54366, CBD \n" "P O Box 700ABG, Gaborone, Botswana"
                        ),
                        "sector": "Property",
                        "board": "Domestic Main Board",
                    },
                    {
                        "counter": "LETSHEGO",
                        "name": "Letshego Holdings Limited",
                        "physical and postal address": (
                            "Tower C, Zambezi Towers, Plot 54352, Central Business " "District, Gaborone Botswana"
                        ),
                        "sector": "Financial Services",
                        "board": "Domestic Main Board",
                    },
                    {
                        "counter": "MINERGY",
                        "name": "Minergy Limited",
                        "physical and postal address": (
                            "Unit B3 & B4, \n1st Floor Plot 43175, " "\nPhakalane, \nGaborone"
                        ),
                        "sector": "Mining",
                        "board": "Domestic Main Board",
                    },
                    {
                        "counter": "NAP",
                        "name": "New African Properties Limited",
                        "physical and postal address": "Plot 20573/4 Block 3 Gaborone",
                        "sector": "Property",
                        "board": "Domestic Main Board",
                    },
                    {
                        "counter": "TURNSTAR",
                        "name": "Turnstar Holdings Limited",
                        "physical and postal address": "Game City Retail Mall,Kgale, Gaborone",
                        "sector": "Property",
                    },
                    {
                        "counter": "STANCHART",
                        "name": "Standard Chartered Botswana Limited",
                        "physical and postal address": (
                            "5th Floor, \nStandard Chartered House, "
                            "\nPlot 1124-30, Queens Road, \nGaborone, \nBotswana"
                        ),
                        "sector": "Banking",
                        "board": "Domestic Main Board",
                    },
                    {
                        "counter": "SEFALANA",
                        "name": "Sefalana Holding Company Limited",
                        "physical and postal address": (
                            "Plot 10247/50 Corner of Noko and Lejara Roads, " "\nBroadjurst Industrial, \nGaborone"
                        ),
                        "sector": "Retail & Wholesale",
                        "board": "Domestic Main Board",
                    },
                    {
                        "counter": "SEEDCO",
                        "name": "Seedco International Limited",
                        "physical and postal address": (
                            "PO BOX 47143,GABORONE, \nBOTSWANA, " "\nUnit 1 Plot 43178, Phakalane, \nGaborone"
                        ),
                        "sector": "Agriculture",
                        "board": "Domestic Main Board",
                    },
                    {
                        "counter": "SECHABA",
                        "name": "Sechaba Brewery Holdings Limited",
                        "physical and postal address": (
                            "Central Business District \nPlot 54367,2nd floor,Mogobe Plaza "
                            "\nP O Box 1956 AAD, Gaborone ,Botswana"
                        ),
                        "sector": "Retail & Wholesale",
                        "board": "Domestic Main Board",
                    },
                    {
                        "counter": "RDCP",
                        "name": "RDC Properties Limited",
                        "physical and postal address": (
                            "Lejara Road, \nPlot 5624, " "\nBroadhurst Ind. – P O Box 495, \nGaborone, \nBotswana"
                        ),
                        "sector": "Property",
                        "board": "Domestic Main Board",
                    },
                    {
                        "counter": "PRIMETIME",
                        "name": "PrimeTime Property Holdings Limited",
                        "physical and postal address": (
                            "Plot 79961 \nOffice 1 Setlhoa Corner " "\nGaborone, Botswana \n(PO Box 1395, Gaborone)"
                        ),
                        "sector": "Property",
                        "board": "Domestic Main Board",
                    },
                    {
                        "counter": "OLYMPIA",
                        "name": "Olympia Capital Corporation (Botswana) Limited",
                        "physical and postal address": "Plot 50371 Fairgrounds, \nGaborone, \nBotswana",
                        "sector": "Retail & Wholesale",
                        "board": "Domestic Main Board",
                    },
                    {
                        "counter": "BMIN-EQO",
                        "name": "Botswana Minerals plc",
                        "physical and postal address": (
                            "Principal Office, \n162 Clontarf Road, \nDublin 3, \nIreland, "
                            "\nRegistered Office, \n20-22 Bedford Row, \nLondon, \nWC1R 4ES"
                        ),
                        "sector": "Mining",
                        "board": "Foreign Venture Capital Board",
                    },
                    {
                        "counter": "TLOU",
                        "name": "Tlou Energy",
                        "physical and postal address": (
                            "Ground Floor, Victoria House, " "\n132 Independence Avenue, \nGaborone, \nBotswana"
                        ),
                        "sector": "Mining",
                        "board": "Foreign Main Board",
                    },
                    {
                        "counter": "SHUMBA ENERGY",
                        "name": "Shumba Energy Limited",
                        "physical and postal address": (
                            "P.O.Box 70311, \nGaborone, \nBotswana, \nPlot 2780 Manong "
                            "Close,Extension 9, \nGaborone, \nIFS Court, Bank Street, "
                            "TwentyEight Cybercity, Ebène 72201, Republic of Mauritius "
                            "\nTel. +230 467 3000"
                        ),
                        "sector": "Mining",
                        "board": "Foreign Main Board",
                    },
                    {
                        "counter": "INVESTEC",
                        "name": "Investec Limited",
                        "physical and postal address": "100 Grayston Drive, Sandown, Sandton, 2196, South Africa",
                        "sector": "Financial Services",
                        "board": "Foreign Main Board",
                    },
                    {
                        "counter": "CA SALES",
                        "name": "CA Sales Holdings Limited",
                        "physical and postal address": (
                            "1st Floor Building C, West End Office Park \n254 Hall "
                            "Street \nCenturion,0157 \nSouth Africa"
                        ),
                        "sector": "Retail & Wholesale",
                        "board": "Foreign Main Board",
                    },
                    {
                        "counter": "ANGLO",
                        "name": "Anglo American Plc",
                        "physical and postal address": ("20 Carlton House Terrace,London SW1Y 5AN, " "United Kingdom"),
                        "sector": "Mining",
                        "board": "Foreign Main Board",
                    },
                    {
                        "counter": "LUC",
                        "name": "Lucara Diamond Corp",
                        "physical and postal address": (
                            "Suite 2000 – 885 West Georgia StreetVancouver, \n" "British Columbia Canada V6C 3E8"
                        ),
                        "sector": "Mining",
                        "board": "Foreign Venture Capital Board",
                    },
                    {
                        "counter": "GAIA",
                        "name": "GAIA Renewables 1 Limited",
                        "physical and postal address": (
                            "146 Campground Road \nNewlands " "\nCape Town \nSouth Africa, \n7780"
                        ),
                        "sector": "Financial Services",
                        "board": "Foreign Investment Entity",
                    },
                ],
            )

    def test_cluster_tile_iii(self):
        with self.open_resource("test32.html") as f:
            response = ExtractTextResponse(
                url="https://www.cse-india.com/listi/database",
                status=200,
                body=f.read(),
            )
            results = extract_by_keywords(
                response,
                keywords=(
                    Keyword("cse scrip code"),
                    Keyword("company name"),
                    Keyword("company status"),
                    Keyword("date of listing"),
                    Keyword("address"),
                    Keyword("state"),
                    Keyword("www"),
                ),
                value_filters={
                    Keyword("address"): (Text("contact no"),),
                    Keyword("state"): (Text("name of the directors"),),
                    Keyword("www"): (Text("close"),),
                },
                required_fields=(
                    Keyword("cse scrip code"),
                    Keyword("company name"),
                    Keyword("company status"),
                    Keyword("date of listing"),
                ),
                tiles_mode=True,
                debug_mode=True,
            )
            self.assertEqual(len(results), 50)
            self.assertEqual(
                results[0],
                {
                    "cse scrip code": "013186",
                    "company name": "20TH CENTURY - ZURICH INDIA MUTUAL FUND",
                    "company status": "Suspended",
                    "date of listing": "27-05-1994",
                },
            )
            self.assertEqual(
                results[26],
                {
                    "address": "65 IDA JEEDIMETLA HYDERABAD 500 855",
                    "cse scrip code": "021060",
                    "company name": "ADILAKSHMI ENTERPRISES LIMITED",
                    "company status": "Active",
                    "date of listing": "01-04-1994",
                    "state": "ANDHRA PRADESH",
                    "www": "www.kljplastics.in",
                },
            )
            self.assertEqual(
                results[-1],
                {
                    "address": "NILHAT HANSE 11 R N MUKHERJEE ROAD CALCUTTA 700 001",
                    "company name": "ALIPURDUAR TEA CO LTD",
                    "company status": "Active",
                    "cse scrip code": "011187",
                    "date of listing": "08-12-1977",
                    "state": "WEST BENGAL",
                },
            )

    def test_cluster_tile_iv(self):
        with self.open_resource("test33.html") as f:
            response = ExtractTextResponse(
                url="https://www.borzamalta.com.mt/links/issuers",
                body=f.read(),
                status=200,
            )
            results = extract_by_keywords(
                response=response,
                keywords=(
                    Keyword("^##"),
                    Keyword("address"),
                    Keyword("contact"),
                    Keyword("telephone"),
                    Keyword("website"),
                ),
                required_fields=(Keyword("title"),),
                tiles_mode=True,
                fill_fields=(Keyword("address"),),
            )
            self.assertEqual(len(results), 102)
            self.assertEqual(
                results[0],
                {
                    "address": "Hyatt Centric Malta, \nTriq Santu Wistin, \nSan Ġiljan \nMalta",
                    "telephone": "+356 22586260",
                    "title": "ACMUS plc",
                    "website": "[www.acmus.mt](https://www.borzamalta.com.mt/links/www.acmus.mt)",
                },
            )
            self.assertEqual(
                results[25],
                {
                    "address": "Eden Place \nSt. George's Bay St. Julians \nSTJ 3310 \nMalta",
                    "contact": "Dr David Zahra, Company Secretary",
                    "title": "Eden Finance plc",
                },
            )
            self.assertEqual(
                results[27],
                {
                    "address": "Level 3 \nValletta Buildings \nSouth Stre \nValletta",
                    "contact": "Dr Malcolm Falzon | Ms Martina Galea",
                    "telephone": "+356 21238989",
                    "title": "Exalco Finance plc",
                    "website": "<https://www.camilleripreziosi.com>",
                },
            )
            self.assertEqual(
                results[51],
                {
                    "address": (
                        "Block 3 Level 0, Trident Park, Mdina Road, Zone 2 \n"
                        "Central Business District, Birkirkara \n"
                        "CBD2010 \n"
                        "Malta"
                    ),
                    "contact": "Dr. Francesca Briffa Polidano, Company Secretary",
                    "telephone": "+356 2092 6000",
                    "title": "Lidion Bank plc",
                    "website": "<https://www.lidionbank.com>",
                },
            )
