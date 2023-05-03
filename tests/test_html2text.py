from unittest import TestCase

from emodels.scrapyutils import ExtractTextResponse
from emodels import html2text


class Html2TextTests(TestCase):
    def test_ids(self):
        html = b"""
<div id="did">
<p class="pc0">this is a line</p>
<p id="pid1">this is a line with id</p>
<p id="pid2">this is another line with id</p>
</div>
        """
        response = ExtractTextResponse(url="http://example.com/example1.html", status=200, body=html)
        expected = """this is a line

this is a line with id <!--#pid1-->

this is another line with id <!--#pid2-->
"""

        self.assertEqual(response.markdown_ids, expected)

        expected_html = """<p>this is a line</p>
<p>this is a line with id</p>
<p>this is another line with id</p>"""
        self.assertEqual(response.markdown_to_html(), expected_html)

    def test_classes(self):
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

        self.assertEqual(response.markdown_classes, expected)

        expected_html = """<p>this is a line</p>
<p>this is a line with class</p>
<p>this is a line with id</p>"""
        self.assertEqual(response.markdown_to_html(), expected_html)

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
        self.assertEqual(response.markdown, expected)

        expected_html = """<table>
<tbody>
<tr>
<td>Data 1</td>
<td>Data 2</td>
</tr>
<tr>
<td>Data 3</td>
<td>Data 4</td>
</tr>
<tr>
<td>Data 5</td>
<td>Data 6</td>
</tr>
</tbody>
</table>"""
        self.assertEqual(response.markdown_to_html(), expected_html)

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
        self.assertEqual(response.markdown, expected)

        expected_html = """<table>
<thead>
<tr>
<th>Head 1</th>
<th>Head 2</th>
</tr>
</thead>
<tbody>
<tr>
<td>Data 1</td>
<td>Data 2</td>
</tr>
<tr>
<td>Data 3</td>
<td>Data 4</td>
</tr>
</tbody>
</table>"""
        self.assertEqual(response.markdown_to_html(), expected_html)

    def test_tables_with_br_line_breaks(self):
        html = b"""
<table><tr><td>Data 1</td><td>Data 2</td></tr>
<tr><td>Data 3</td><td>Data 4</td></tr>
<tr><td>Data 5</td><td><p>Data 6</p></td></tr>
<tr><td>Data 7</td><td>Data 8</td></tr>
</table>
"""
        saved = html2text.config.ENABLE_BR_WITHIN_TABLE
        html2text.config.ENABLE_BR_WITHIN_TABLE = True
        response = ExtractTextResponse(url="http://example.com/example2.html", status=200, body=html)
        expected = """| Data 1| Data 2|
| Data 3| Data 4|
| Data 5| <br><br>Data 6<br><br>|
| Data 7| Data 8|
"""
        self.assertEqual(response.markdown, expected)
        html2text.config.ENABLE_BR_WITHIN_TABLE = saved

        expected_html = """<table>
<tbody>
<tr>
<td>Data 1</td>
<td>Data 2</td>
</tr>
<tr>
<td>Data 3</td>
<td>Data 4</td>
</tr>
<tr>
<td>Data 5</td>
<td><br><br>Data 6<br><br></td>
</tr>
<tr>
<td>Data 7</td>
<td>Data 8</td>
</tr>
</tbody>
</table>"""
        self.assertEqual(response.markdown_to_html(), expected_html)

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
| Data 5| 

Data 6

|
| Data 7| Data 8|
"""
        self.assertEqual(response.markdown, expected)
