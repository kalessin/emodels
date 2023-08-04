from unittest import TestCase

from emodels.scrapyutils.response import ExtractTextResponse
from emodels import html2text


class Html2TextTests(TestCase):
    def test_ids(self):
        html = b"""
<div id="did">
<p>first line</p>
<p class="pc0">this is a line</p>
<p id="pid1">this is a line with id</p>
<p id="pid2">this is another line with id</p>
</div>
<p>A final line<p>
        """
        response = ExtractTextResponse(url="http://example.com/example1.html", status=200, body=html)
        response._add_extra_ids(["did"])
        expected = """first line <!--#did-->

this is a line <!--#did-->

this is a line with id <!--#pid1-->

this is another line with id <!--#pid2-->

A final line
"""

        self.assertEqual(expected, response.markdown_ids)

    def test_ids_ii(self):
        html = b"""
<div class="did">
<p>first line</p>
<p class="pc0">this is a line</p>
<p id="pid1">this is a line with id</p>
<p id="pid2">this is another line with id</p>
</div>
        """
        response = ExtractTextResponse(url="http://example.com/example1.html", status=200, body=html)
        expected = """first line

this is a line

this is a line with id <!--#pid1-->

this is another line with id <!--#pid2-->
"""

        self.assertEqual(expected, response.markdown_ids)

    def test_classes(self):
        html = b"""
<div class="did">
<p>this is a line</p>
<p class="pc1">this is a line with class</p>
<p id="pid2">this is a line with id</p>
</div>
<p>A final line<p>
        """
        response = ExtractTextResponse(url="http://example.com/example2.html", status=200, body=html)
        response._add_extra_classes(["did"])
        expected = """this is a line <!--.did-->

this is a line with class <!--.pc1-->

this is a line with id <!--.did-->

A final line
"""

        self.assertEqual(expected, response.markdown_classes)

    def test_classes_ii(self):
        html = b"""
<div id="did">
<p>this is a line</p>
<p class="pc1">this is a line with class</p>
<p id="pid2">this is a line with id</p>
</div>
        """
        response = ExtractTextResponse(url="http://example.com/example2.html", status=200, body=html)
        expected = """this is a line

this is a line with class <!--.pc1-->

this is a line with id
"""

        self.assertEqual(expected, response.markdown_classes)

    def test_tables_plain(self):
        html = b"""
<table><tr><td>Data 1</td><td>Data 2</td></tr>
<tr><td>Data 3</td><td>Data 4</td></tr>
<tr><td>Data 5</td><td>Data 6</td></tr>
</table>
"""

        response = ExtractTextResponse(url="http://example.com/example2.html", status=200, body=html)
        expected = """| Data 1| Data 2|
| Data 3| Data 4|
| Data 5| Data 6|
"""
        self.assertEqual(expected, response.markdown)

    def test_tables_with_header(self):
        html = b"""
<table><tr><th>Head 1</th><th>Head 2</th></tr>
<tr><td>Data 1</td><td>Data 2</td></tr>
<tr><td>Data 3</td><td>Data 4</td></tr>
</table>

"""

        response = ExtractTextResponse(url="http://example.com/example2.html", status=200, body=html)
        expected = """| Head 1| Head 2|
| ---|---|
| Data 1| Data 2|
| Data 3| Data 4|
"""
        self.assertEqual(expected, response.markdown)

    def test_tables_with_br_line_breaks(self):
        html = b"""
<table><tr><td>Data 1</td><td>Data 2</td></tr>
<tr><td>Data 3</td><td>Data 4</td></tr>
<tr><td>Data 5</td><td><p>Data 6</p></td></tr>
<tr><td>Data 7</td><td>Data 8</td></tr>
</table>
"""
        saved = html2text.config.LINE_BREAK_WITHIN_TABLE
        html2text.config.LINE_BREAK_WITHIN_TABLE = "<br>"
        response = ExtractTextResponse(url="http://example.com/example2.html", status=200, body=html)
        expected = """| Data 1| Data 2|
| Data 3| Data 4|
| Data 5| <br><br>Data 6<br><br>|
| Data 7| Data 8|
"""
        self.assertEqual(expected, response.markdown)
        html2text.config.LINE_BREAK_WITHIN_TABLE = saved

    def test_tables_with_line_breaks_default(self):
        html = b"""
<table><tr><td>Data 1</td><td>Data 2</td></tr>
<tr><td>Data 3</td><td>Data 4</td></tr>
<tr><td>Data 5</td><td><p>Data 6</p></td></tr>
<tr><td>Data 7</td><td>Data 8</td></tr>
</table>
"""
        response = ExtractTextResponse(url="http://example.com/example2.html", status=200, body=html)
        expected = """| Data 1| Data 2|
| Data 3| Data 4|
| Data 5| \n\nData 6

|
| Data 7| Data 8|
"""
        self.assertEqual(expected, response.markdown)

    def test_entities(self):
        html = b"""<div>There&nbsp;are&nbsp;spaces</div>"""
        response = ExtractTextResponse(url="http://example.com/example2.html", status=200, body=html)
        self.assertEqual(response.markdown, "There are spaces\n")

        html = b"""<div>There&amp;nbsp;are&amp;nbsp;spaces</div>"""
        response = ExtractTextResponse(url="http://example.com/example2.html", status=200, body=html)
        self.assertEqual(response.markdown, "There&nbsp;are&nbsp;spaces\n")

        html = b"""<div>There&nbspare&nbspspaces</div>"""
        response = ExtractTextResponse(url="http://example.com/example2.html", status=200, body=html)
        self.assertEqual("There are spaces\n", response.markdown)

    def test_ids_delayed(self):
        html = b"""
<div id="did">
<span>
<p>first line</p>
<p class="pc0">this is a line</p>
<p id="pid1">this is a line with id</p>
<p id="pid2">this is another line with id</p>
</span>
</div>
<p>A final line<p>
        """
        response = ExtractTextResponse(url="http://example.com/example1.html", status=200, body=html)
        response._add_extra_ids(["did"])
        expected = """first line <!--#did-->

this is a line <!--#did-->

this is a line with id <!--#pid1-->

this is another line with id <!--#pid2-->

A final line
"""

        self.assertEqual(expected, response.markdown_ids)

    def test_classes_delayed(self):
        html = b"""
<div class="did">
<span>
<p>this is a line</p>
<p class="pc1">this is a line with class</p>
<p id="pid2">this is a line with id</p>
</span>
</div>
<p>A final line<p>
        """
        response = ExtractTextResponse(url="http://example.com/example2.html", status=200, body=html)
        response._add_extra_classes(["did"])
        expected = """this is a line <!--.did-->

this is a line with class <!--.pc1-->

this is a line with id <!--.did-->

A final line
"""

        self.assertEqual(expected, response.markdown_classes)
