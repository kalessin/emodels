import os
import re
import gzip
import json
from unittest import TestCase

from itemloaders.processors import TakeFirst
from scrapy import Item, Field
from scrapy.http import TextResponse

os.environ["EMODELS_SAVE_EXTRACT_ITEMS"] = "1"
os.environ["EMODELS_DIR"] = os.path.dirname(__file__)

from emodels import config
from emodels.scrapyutils import ExtractItemLoader, COMMENT_RE, ExtractTextResponse

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), 'samples')


class JobItem(Item):
    job_title = Field()
    description = Field()
    description_as_html = Field()
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


class JobItemLoader(ExtractItemLoader):
    default_item_class = JobItem
    default_output_processor = TakeFirst()


class BusinessSearchItem(Item):
    name = Field()
    phone = Field()
    website = Field()
    address = Field()
    profile_url = Field()
    category = Field()


class BusinessSearchItemLoader(ExtractItemLoader):
    default_item_class = BusinessSearchItem
    default_output_processor = TakeFirst()


class ScrapyUtilsTests(TestCase):

    jobs_result_file = os.path.join(config.EMODELS_DIR, "items/JobItem.jl.gz")
    business_result_file = os.path.join(config.EMODELS_DIR, "items/BusinessSearchItem.jl.gz")

    def tearDown(self):
        for col in "jobs", "business":
            fname = getattr(self, f"{col}_result_file")
            if os.path.isfile(fname):
                os.remove(fname)

    def test_itemloader_base(self):
        sample_file = os.path.join(SAMPLES_DIR, "job21.html")
        body = open(sample_file).read().encode("utf8")
        tresponse = TextResponse(url="https://careers.und.edu/jobs/job21.html", body=body, status=200)
        loader = JobItemLoader(response=tresponse)
        loader.add_text_id("job_title", "job_title_2_2")
        loader.add_text_id("employment_type", "employment_type_2_2_0_0")
        loader.add_text_id("job_id", "requisition_identifier_2_2_0")
        loader.add_text_re("description", "(###\s+.+?)\*\*apply now\*\*", re.S | re.I)
        loader.add_text_re_as_html("description_as_html", "(###\s+.+?)\*\*apply now\*\*", re.S | re.I)

        response = loader.context["response"]
        self.assertEqual(response.markdown[slice(*loader.extract_indexes["job_title"])], 'Student Athlete Support Services Coord')
        self.assertEqual(response.markdown[slice(*loader.extract_indexes["job_id"])], '492556')
        self.assertEqual(response.markdown[slice(*loader.extract_indexes["employment_type"])], 'Full-time Staff')

        item = loader.load_item()
        
        self.assertFalse(COMMENT_RE.findall(item["description"]))
        self.assertFalse(COMMENT_RE.findall(item["description_as_html"]))

        self.assertEqual(item["description"][:80], "###  Student Athlete Support Services Coord  \n\n\n  * __ 492556  \n\n\n\n\n  * __ Grand")
        self.assertEqual(item["description"][-80:], "arning skills.\n\n\n\n**Please note, all employment postings close at 11:55pm CST.**")
        self.assertEqual(item["description_as_html"][:80], "<h3>Student Athlete Support Services Coord</h3>\n\n<ul>\n<li><p>__ 492556  </p></li")
        self.assertEqual(item["description_as_html"][-80:], "><strong>Please note, all employment postings close at 11:55pm CST.</strong></p>")

        with gzip.open(self.jobs_result_file, "rt") as fz:
            data = json.loads(next(fz))

        self.assertFalse(COMMENT_RE.findall(data["markdown"]))

        self.assertEqual(data["markdown"][slice(*data["indexes"]["job_title"])], 'Student Athlete Support Services Coord')
        self.assertEqual(data["markdown"][slice(*data["indexes"]["job_id"])], '492556')
        self.assertEqual(data["markdown"][slice(*data["indexes"]["employment_type"])], 'Full-time Staff')

        self.assertEqual(data["markdown"][slice(*data["indexes"]["description"])], item["description"])
        self.assertEqual(data["markdown"][slice(*data["indexes"]["description_as_html"])], item["description"])

    def test_split_text_re(self):
        sample_file = os.path.join(SAMPLES_DIR, "yell.html")
        body = open(sample_file).read().encode("utf8")
        response = ExtractTextResponse(url="https://yell.com/result.html", body=body, status=200)

        for r in response.css_split(".businessCapsule--mainRow"):
            loader = BusinessSearchItemLoader(response=r)
            loader.add_text_re("name", r"##(.+)")
            loader.add_text_re("phone", r"Tel([\s\d]+)")
            loader.add_text_re("website", r"Website\]\((.+?)\)")
            loader.add_text_re("address", r"\[.+\|(.+)\]\(.+view=map", re.S)
            loader.add_text_re("profile_url", r"\[More info .+\]\((http.+?\d+/)")
            loader.add_text_re(
                "category",
                r"##.+\]\(.+\)(?:.+with Yell)?(.+?)(?:###.+)?\[ Website",
                re.S,
            )   
            loader.load_item()

        extracted = []
        with gzip.open(self.business_result_file, "rt") as fz:
            for l in fz:
                d = json.loads(l)
                extracted.append({attr: d["markdown"][slice(*d["indexes"][attr])] for attr in d["indexes"]})
        
        self.assertEqual(len(extracted), 25)
        self.assertEqual(extracted[0]["name"], "Craig Wood Solicitors")
        self.assertEqual(extracted[1]["category"], "Solicitors")
        self.assertEqual(extracted[2]["website"], "http://www.greyandcosolicitors.co.uk")
        self.assertEqual(extracted[3]["phone"], "01463 225544")
        self.assertEqual(extracted[4]["address"], "3 Ardconnel Terrace,  Inverness, IV2 3AE")
        self.assertEqual(extracted[5]["profile_url"], 'https://yell.com/biz/jack-gowans-and-marc-dickson-inverness-901395225/')

