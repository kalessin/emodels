# Markdown primitives & tuning knobs

How the extraction actually works under the modes, the `ExtractTextResponse` primitives you test
with, and the spider arguments that tune recall/noise.

## `ExtractTextResponse` — the markdown domain

[emodels/scrapyutils/response.py](../../../emodels/scrapyutils/response.py). A `TextResponse` subclass
(so all of scrapy's `.css`/`.xpath`/`.text` still work) that adds the **markdown view** the clustering
engine operates on:

- **`.markdown`** — the response rendered to markdown (`html2text`, `bodywidth=0`), then cleaned
  (whitespace/commas normalized, link URLs tidied). This is what `item`/`tiles`/`combined` see — when
  reasoning about field locations, look at *this*, not the browser DOM.
- **`.markdown_ids` / `.markdown_classes`** — markdown annotated with `<!--#id-->` / `<!--.class-->`
  comments so a selector can be restricted to a labelled region (powers `text_re`'s `tid`).
- **`.text_re(reg=None, tid=None, flags=0, skip_prefix=DEFAULT_SKIP_PREFIX, idx=0, optimize=False)`** —
  the low-level markdown selector. Returns `[(extracted, start, end), ...]` for regex `reg` over the
  markdown. `tid` restricts to a region by `#id`/`itemprop` or `.class`. `skip_prefix` (default
  `[^a-zA-Z0-9$]*`) strips leading punctuation from the match. This is the primitive
  `additional_regexes` uses per field.
- **`.css_split(sel)` / `.xpath_split(sel)`** — split one response into several sub-responses (one per
  matched element), e.g. to isolate repeated blocks before extraction.

The matching `ItemLoader` extension `ExtractItemLoader.add_text_re(...)`
([emodels/scrapyutils/loader.py], documented in the [README](../../../README.md#markdown-selectors))
is the *manual* counterpart — a markdown selector you hardcode in a loader, which also feeds dataset
generation when `EMODELS_SAVE_EXTRACT_ITEMS=1`.

## Testing selectors / inspecting the markdown (`scrapy shell`)

```python
$ scrapy shell <url>          # or fetch(<url>) inside the shell
>>> from emodels.scrapyutils.response import ExtractTextResponse
>>> response = response.replace(cls=ExtractTextResponse)
>>> print(response.markdown)              # see exactly what the clustering engine sees
>>> response.text_re(r"ISIN[:\s]+([A-Z0-9]{12})")     # test a field regex
>>> response.text_re(r"(.+)", tid="#price")           # restrict to an id/itemprop/class region
```

Iterate `text_re` patterns here, then feed the good ones to the spider via `additional_regexes`. For
blocked/JS pages, fetch production-aligned bytes through the same transport the spider will use (your
scraping fetch/proxy path), not the browser, since rendering/bans can change the bytes (and thus the
markdown).

## Tuning knobs (spider arguments → the engines)

Set these on the `ExtractionSpiderParams` subclass (or per-site config). They flow into
`extract_by_keywords` / `parse_tables_from_response` (see their docstrings in
[cluster.py](../../../emodels/extract/cluster.py) / [table.py](../../../emodels/extract/table.py)).

- **`fields`** — the target fields/keywords. In markdown terms a field is a label to search; you can
  even use markup as a field, e.g. `^#` matches a markdown heading (title), `^##` a sub-heading.
- **`listing_fields`** — separate, usually smaller, column set for table extraction (see
  [extract-modes.md](extract-modes.md)).
- **`required_fields`** — results missing all of these are down-scored/dropped (default: all `fields`
  are required). Loosen when many records legitimately lack some field.
- **`constraints` / `constraints_overrides`** — `{field: pattern}` a value must match or the field is
  dropped. `pattern` is a regex **or** a special keyword: `date_type`, `url_type`. Prefer
  `constraints_overrides` (merged onto the defaults) over replacing `constraints` wholesale.
- **`value_filters`** — `{field: (regex, ...)}`; candidates whose value matches are down-scored
  during extraction (noise suppression at scoring time).
- **`drop_fields`** — fields removed from the *final* result (post-processing), unlike `value_filters`
  which acts during extraction.
- **`drop_items`** — `{field: (regex, ...)}`; if a final item's field matches, the whole item is
  dropped.
- **`dedupe_keywords`** — fields used to detect duplicate records (defaults to `fields`).
- **`additional_regexes`** — `{field: (regex | (regex|None, tid), ...)}`; precise per-field markdown
  selectors (via `text_re`) layered on top of the keyword search. First alternative that extracts
  wins. Use when label-search alone is unreliable for a field.
- **`fill_fields`** — fields with no explicit label in the markdown but inferable from the text
  *between* labelled fields; must also appear in `fields` to fix ordering.
- **`multiline_fields`** — `{field: max_lines}`; allow a field to span several lines (default is the
  keyword's own line only), e.g. `{"address": 4}`.
- **`max_tables`** — how many tables to consider (listing/combined); raise when one dataset spans
  several differently-headed tables.
- **`debug_mode`** — dumps a `markdown{N}.md` per visited page and enables verbose algorithm tracing;
  invaluable while tuning the above.

## Required subclass methods (recap)

- **`validate_result(result, columns) -> bool`** — accept/reject a candidate record (e.g. require a
  minimum number of present fields, reject obviously-wrong values). Returning `False` discards it.
- **`adapt_result(result, response)`** — post-process a accepted record: rename fields, normalize
  values, drop conditionals. The base `_adapt_result` also resolves markdown links, urljoins a `url`
  field, and applies `drop_fields` around your `adapt_result`.
