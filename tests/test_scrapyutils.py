import os
import re
import gzip
import json
from unittest import TestCase

from itemloaders.processors import TakeFirst
from scrapy import Item, Field

os.environ["EMODELS_SAVE_EXTRACT_ITEMS"] = "1"
os.environ["EMODELS_DIR"] = os.path.dirname(__file__)

from emodels import config
from emodels.scrapyutils import ExtractItemLoader, ExtractTextResponse, COMMENT_RE

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


class ScrapyUtilsTests(TestCase):

    jobs_result_file = os.path.join(config.EMODELS_DIR, "items/JobItem.jl.gz")

    def tearDown(self):
        if os.path.isfile(self.jobs_result_file):
            os.remove(self.jobs_result_file)

    def test_extract_job(self):
        sample_file = os.path.join(SAMPLES_DIR, "job21.html")
        body = open(sample_file).read().encode("utf8")
        response = ExtractTextResponse(url="https://careers.und.edu/jobs/job21.html", body=body, status=200)
        loader = JobItemLoader(response=response)
        loader.add_text_id("job_title", "job_title_2_2")
        loader.add_text_id("employment_type", "employment_type_2_2_0_0")
        loader.add_text_id("job_id", "requisition_identifier_2_2_0")
        loader.add_text_re("description", "(###\s+.+?)\*\*apply now\*\*", re.S | re.I)
        loader.add_text_re_as_html("description_as_html", "(###\s+.+?)\*\*apply now\*\*", re.S | re.I)

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
