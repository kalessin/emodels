import os
import re
from unittest import TestCase
from typing import Dict

from itemloaders.processors import TakeFirst, Join
from scrapy import Item, Field
from scrapy.http import TextResponse

from emodels import config
from emodels.scrapyutils.loader import ExtractItemLoader, ItemSample
from emodels.scrapyutils.response import COMMENT_RE
from emodels.datasets.utils import DatasetFilename, build_response_from_sample_data


class JobItemTest(Item):
    job_title = Field()
    description = Field()
    url = Field()
    employment_type = Field()
    apply_url = Field()
    job_id = Field()
    publication_date = Field()
    category = Field()
    closing_date = Field()
    sublocation = Field()
    postal_code = Field()
    city = Field()
    state = Field()
    country = Field()
    response = Field()
    locality = Field()


class JobItemLoader(ExtractItemLoader):
    default_item_class = JobItemTest
    default_output_processor = TakeFirst()


class BusinessSearchItemTest(Item):
    name = Field()
    phone = Field()
    website = Field()
    address = Field()
    profile_url = Field()
    category = Field()
    locality = Field()
    street = Field()
    postal_code = Field()
    address_alt = Field()


class BusinessSearchItemLoader(ExtractItemLoader):
    default_item_class = BusinessSearchItemTest
    default_output_processor = TakeFirst()


class ScrapyUtilsTests(TestCase):
    jobs_result_file: DatasetFilename[ItemSample] = DatasetFilename(
        os.path.join(config.EMODELS_REPOSITORY, "items/JobItemTest/0.jl.gz")
    )
    business_result_file: DatasetFilename[ItemSample] = DatasetFilename(
        os.path.join(config.EMODELS_REPOSITORY, "items/BusinessSearchItemTest/0.jl.gz")
    )
    samples_file: DatasetFilename[ItemSample] = DatasetFilename(
        os.path.join(os.path.dirname(__file__), "samples.jl.gz")
    )
    samples: Dict[str, TextResponse]
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        cls.samples = {d["url"]: build_response_from_sample_data(d) for d in cls.samples_file}

    def tearDown(self):
        for col in "jobs", "business":
            fname = getattr(self, f"{col}_result_file")
            dname = os.path.dirname(fname)
            try:
                for f in os.listdir(dname):
                    os.remove(os.path.join(dname, f))
            except FileNotFoundError:
                pass

    def test_case_one(self):
        response = self.samples["https://careers.und.edu/jobs/job21.html"]
        loader = JobItemLoader(response=response)
        loader.add_text_re("job_title", tid="#job_title_2_2")
        loader.add_text_re("employment_type", tid="#employment_type_2_2_0_0")
        loader.add_text_re("job_id", tid="#requisition_identifier_2_2_0")
        loader.add_text_re("description", r"(###\s+.+?)\*\*apply now\*\*", flags=re.S | re.I)

        item = loader.load_item()

        self.assertFalse(COMMENT_RE.findall(item["description"]))

        self.assertEqual(
            item["description"][:80],
            "### Student Athlete Support Services Coord \n\n * __ 492556 \n\n\n\n * __ Grand Forks,",
        )
        self.assertEqual(
            item["description"][-80:],
            "arning skills.\n\n\n\n**Please note, all employment postings close at 11:55pm CST.**",
        )

        self.assertEqual(
            response.markdown[slice(*loader.extract_indexes["job_title"])], "Student Athlete Support Services Coord"
        )
        self.assertEqual(response.markdown[slice(*loader.extract_indexes["job_id"])], "492556")
        self.assertEqual(response.markdown[slice(*loader.extract_indexes["employment_type"])], "Full-time Staff")

        data = next(self.jobs_result_file)

        self.assertFalse(COMMENT_RE.findall(data["markdown"]))

        self.assertEqual(
            data["markdown"][slice(*data["indexes"]["job_title"])], "Student Athlete Support Services Coord"
        )
        self.assertEqual(data["markdown"][slice(*data["indexes"]["job_id"])], "492556")
        self.assertEqual(data["markdown"][slice(*data["indexes"]["employment_type"])], "Full-time Staff")
        self.assertEqual(data["markdown"][slice(*data["indexes"]["description"])], item["description"])

        self.assertTrue(response.text_re(tid=".job-field job-title"))

    def test_case_one_css(self):
        response = self.samples["https://careers.und.edu/jobs/job21.html"]
        loader = JobItemLoader(response=response)
        loader.add_css("job_title", "#job_title_2_2::text")
        loader.add_css("employment_type", "#employment_type_2_2_0_0::text")
        loader.add_css("job_id", "#requisition_identifier_2_2_0::text")

        loader.load_item()

        self.assertEqual(
            response.markdown[slice(*loader.extract_indexes["job_title"])], "Student Athlete Support Services Coord"
        )
        self.assertEqual(response.markdown[slice(*loader.extract_indexes["job_id"])], "492556")
        self.assertEqual(response.markdown[slice(*loader.extract_indexes["employment_type"])], "Full-time Staff")

        data: ItemSample = next(DatasetFilename(self.jobs_result_file))

        self.assertFalse(COMMENT_RE.findall(data["markdown"]))

        self.assertEqual(
            data["markdown"][slice(*data["indexes"]["job_title"])], "Student Athlete Support Services Coord"
        )
        self.assertEqual(data["markdown"][slice(*data["indexes"]["job_id"])], "492556")
        self.assertEqual(data["markdown"][slice(*data["indexes"]["employment_type"])], "Full-time Staff")

    def test_case_one_xpath(self):
        response = self.samples["https://careers.und.edu/jobs/job21.html"]
        loader = JobItemLoader(response=response)
        loader.add_xpath("job_title", "//*[@id='job_title_2_2']/text()")
        loader.add_xpath("employment_type", "//*[@id='employment_type_2_2_0_0']/text()")
        loader.add_xpath("job_id", "//*[@id='requisition_identifier_2_2_0']/text()")

        loader.load_item()

        self.assertEqual(
            response.markdown[slice(*loader.extract_indexes["job_title"])], "Student Athlete Support Services Coord"
        )
        self.assertEqual(response.markdown[slice(*loader.extract_indexes["job_id"])], "492556")
        self.assertEqual(response.markdown[slice(*loader.extract_indexes["employment_type"])], "Full-time Staff")

        data: ItemSample = next(DatasetFilename(self.jobs_result_file))

        self.assertFalse(COMMENT_RE.findall(data["markdown"]))

        self.assertEqual(
            data["markdown"][slice(*data["indexes"]["job_title"])], "Student Athlete Support Services Coord"
        )
        self.assertEqual(data["markdown"][slice(*data["indexes"]["job_id"])], "492556")
        self.assertEqual(data["markdown"][slice(*data["indexes"]["employment_type"])], "Full-time Staff")

    def test_case_two(self):
        response = self.samples["https://yell.com/result.html"]

        for r in response.css_split(".businessCapsule--mainRow"):
            loader = BusinessSearchItemLoader(response=r)
            loader.add_text_re("name", r"##(.+)")
            loader.add_text_re("phone", r"Tel([\s\d]+)", tid="#telephone")
            loader.add_text_re("website", r"Website\]\((.+?)\)")
            loader.add_text_re("address", r"\[(?:.+\|)?(.+)\]\(.+view=map")
            loader.add_text_re("profile_url", r"\[More info .+\]\((http.+?\d+/)")
            loader.add_text_re(
                "category",
                tid=".businessCapsule--classification",
            )
            loader.add_text_re("address_alt", reg=r"(?:.+\|)?(.+?),?", tid="#addressLocality")
            loader.add_text_re("street", reg=r"(?:.+\|)?(.+?),?", tid="#streetAddress")
            loader.load_item()

        extracted = []
        for d in DatasetFilename(self.business_result_file):
            extracted.append({attr: d["markdown"][slice(*d["indexes"][attr])] for attr in d["indexes"]})

        self.assertEqual(len(extracted), 25)
        self.assertEqual(len([e for e in extracted if "name" in e]), 25)
        self.assertEqual(len([e for e in extracted if "category" in e]), 25)
        categories = [e["category"] for e in extracted if "category" in e]
        self.assertEqual(categories.count("Solicitors"), 24)
        self.assertEqual(categories.count("Personal Injury"), 1)
        self.assertEqual(len([e for e in extracted if "phone" in e]), 25)
        self.assertEqual(len([e for e in extracted if "website" in e]), 20)
        self.assertEqual(len([e for e in extracted if "address" in e]), 24)
        self.assertFalse("address" in extracted[1])
        self.assertEqual(len([e for e in extracted if "street" in e]), 24)
        self.assertEqual(len([e for e in extracted if "profile_url" in e]), 25)

        self.assertEqual(extracted[0]["name"], "Craig Wood Solicitors")
        self.assertEqual(extracted[1]["category"], "Solicitors")
        self.assertEqual(extracted[2]["website"], "http://www.greyandcosolicitors.co.uk")
        self.assertEqual(extracted[3]["phone"], "01463 225544")
        self.assertEqual(extracted[4]["address"], "3 Ardconnel Terrace, Inverness, IV2 3AE")
        self.assertEqual(extracted[4]["address_alt"], "3 Ardconnel Terrace, Inverness,")
        self.assertEqual(
            extracted[5]["profile_url"], "https://yell.com/biz/jack-gowans-and-marc-dickson-inverness-901395225/"
        )
        self.assertEqual(extracted[7]["street"], "York House, 20, Church St,")

    def test_case_two_css_xpath(self):
        response = self.samples["https://yell.com/result.html"]

        for r in response.css_split(".businessCapsule--mainRow"):
            loader = BusinessSearchItemLoader(response=r)
            loader.add_css("name", "h2::text")
            loader.add_css("phone", "[itemprop='telephone']::text")
            loader.add_xpath("website", "//a[contains(text(), 'Website')]/@href")
            loader.add_xpath("address", "//*[@itemprop='address']//text()", Join())
            loader.add_xpath(
                "profile_url", "//a[contains(text(), 'More info')]/@href", Join(), lambda x: response.urljoin(x)
            )
            loader.add_css("category", ".businessCapsule--classification::text")
            loader.add_css("locality", "[itemprop='addressLocality']::text")
            loader.add_css("street", "[itemprop='streetAddress']::text", Join(), lambda x: x.strip(", "))
            loader.add_css("postal_code", "[itemprop='postalCode']::text")
            loader.load_item()

        extracted = []
        for d in DatasetFilename(self.business_result_file):
            extracted.append({attr: d["markdown"][slice(*d["indexes"][attr])] for attr in d["indexes"]})

        self.assertEqual(len(extracted), 25)
        self.assertEqual(len([e for e in extracted if "name" in e]), 25)
        self.assertEqual(len([e for e in extracted if "category" in e]), 25)
        categories = [e["category"] for e in extracted if "category" in e]
        self.assertEqual(categories.count("Solicitors"), 24)
        self.assertEqual(categories.count("Personal Injury"), 1)
        self.assertEqual(len([e for e in extracted if "phone" in e]), 25)
        self.assertEqual(len([e for e in extracted if "website" in e]), 20)
        self.assertEqual(len([e for e in extracted if "address" in e]), 25)
        self.assertEqual(len([e for e in extracted if "locality" in e]), 24)
        self.assertEqual(len([e for e in extracted if "street" in e]), 24)
        self.assertEqual(len([e for e in extracted if "postal_code" in e]), 24)
        self.assertEqual(len([e for e in extracted if "profile_url" in e]), 25)

        self.assertEqual(extracted[0]["name"], "Craig Wood Solicitors")
        self.assertEqual(extracted[1]["category"], "Solicitors")
        self.assertEqual(extracted[2]["website"], "http://www.greyandcosolicitors.co.uk")
        self.assertEqual(extracted[3]["phone"], "01463 225544")
        self.assertEqual(extracted[4]["address"], "3 Ardconnel Terrace, Inverness, IV2 3AE")
        self.assertEqual(
            extracted[5]["profile_url"], "https://yell.com/biz/jack-gowans-and-marc-dickson-inverness-901395225/"
        )
        self.assertEqual(extracted[6]["locality"], "Inverness")
        self.assertEqual(extracted[7]["street"], "York House, 20, Church St")
        self.assertEqual(extracted[8]["postal_code"], "IV1 1DF")

    def test_case_three(self):
        response = self.samples["https://npc.isolvedhire.com/jobs/857557.html"]
        self.assertEqual(
            response.text_re(tid=".job-items"),
            [("Holbrook, AZ, USA", 456, 473), ("13.85", 480, 485), ("Hourly", 492, 498), ("Part Time", 505, 514)],
        )

        loader = JobItemLoader(response=response)

        loader.add_text_re("locality", tid=".job-items")
        loader.add_text_re("employment_type", tid=".job-items", idx=3)

        item = loader.load_item()
        self.assertEqual(item["locality"], "Holbrook, AZ, USA")
        self.assertEqual(item["employment_type"], "Part Time")
