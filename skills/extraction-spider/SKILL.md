---
name: extraction-spider
description: Use ONLY when the user explicitly asks to build/extend a scraping spider with the emodels extraction method (emodels.extract.spider.ExtractionSpider) — the markdown/text + clustering approach (NOT xpath/css). Covers what it is, when it's the right tool vs. a cheaper structured source (JSON/API), the extraction modes (listing/item/tiles/any/hybrid/combined), how to create a concrete spider, and the markdown primitives (ExtractTextResponse) behind it. Requires the user to provide the fields to extract. Tightly coupled to site analysis — analyze the site before choosing this method and its mode. This is a base skill, to be extended later with worked examples.
---

# Extraction spiders (emodels `ExtractionSpider`)

`emodels.extract.spider.ExtractionSpider` extracts items in the **text (markdown) domain** instead of
the html-source domain. Rather than xpath/css selectors tied to a site's HTML, it renders each
response to markdown (via `ExtractTextResponse`) and locates fields by **keyword/label search,
regex, and clustering** over that text. The payoff (see the emodels [README](../../README.md)):
markdown exposes *labelling and structural patterns common across sites*, so the same extraction
code generalizes across many sites and survives HTML-layout changes — and the same primitives can
generate datasets to train extractive-QA transformer models.

It is an **extraction engine**, not a crawler: you (or a base crawler that embeds it) drive the
crawl and call `extract_items_from_page(response)`.

## Gate — do not start unless both hold

1. **The user explicitly asked for this method** (pre- or post-analysis). Don't pick it on your own.
2. **The user provided the target fields.** Extraction here is keyword/label driven — you need the
   field names (e.g. `isin`, `name`, `address`, `phone`). No fields → ask for them first.

## FIRST: is there a cheaper, more reliable source? (analysis directive)

Before committing to markdown extraction, **analyze the site first** — its page types, data sources,
hidden JSON/XHR APIs, and pagination — and check for a *structured* source. Markdown/clustering
extraction is powerful but heuristic; **if clean structured data exists, prefer it** and say so:

- A **JSON/XHR API** or embedded JSON (e.g. `__NEXT_DATA__`, JSON-LD, a backing data endpoint) →
  hit the API / parse the JSON directly. When a site already serves the data as structured JSON,
  that path is simpler and more reliable than markdown extraction — prefer it.
- A clean, stable **HTML structure with reliable ids/classes** for a single site → a plain scrapy
  spider with css/xpath may be simpler to reason about.
- A downloadable structured file (CSV/XLS) → parse it.

**Use `ExtractionSpider` when:** the data lives in semi-structured human-readable text with no stable
selectors; and/or you want extraction that **generalizes across many similar sites** and is robust to
layout changes; and/or you're building toward an EQA training dataset. Recommend the cheaper path
when one clearly exists — the user can still choose markdown extraction.

## Extraction modes (set via `extract_mode`)

Picking the mode IS the core of the analysis. One-liners (full guidance:
[references/extract-modes.md](references/extract-modes.md)):

- **`listing`** — page has an HTML `<table>` of rows; parse it by matching column headers to your
  fields. (Operates on real `<table>` elements via xpath, guided by your `fields`.)
- **`item`** — one item per page; locate fields by label/keyword in the markdown (clustering).
- **`tiles`** — many repeated item blocks ("tiles") on one page; cluster repeated field patterns.
- **`any`** — try `listing`, fall back to `item` if nothing was extracted.
- **`hybrid`** — `listing` first, then follow each row's item URL (`item_url_template` or a `url`
  field) and complete the record from the item page.
- **`combined`** — run table + keyword extraction on the **same** page and merge (data split between a
  table and surrounding text; forces `max_tables ≥ 2`).

If unsure which mode, analyze first; `any` is a reasonable default when a page might be either a
listing or a single item. Don't try to grasp every case up front — this skill will gain
mode-specific worked examples over time.

## Create a concrete spider (core procedure)

`ExtractionSpider` is abstract: a subclass must implement `validate_result` and `adapt_result`, and
configure arguments through a `ExtractionSpiderParams` subclass — a pydantic `BaseModel`: add a
`Field(...)` per new or overridden argument, and read them at runtime as `self.args.<name>`.

```python
from typing import Tuple
from emodels.extract.spider import ExtractionSpider, ExtractionSpiderParams
from emodels.extract.utils import Keyword, Result, Text

class MyParams(ExtractionSpiderParams):
    # override only changed defaults / new args; everything else is inherited
    fields: Tuple[Keyword, ...] = (Keyword("name"), Keyword("isin"), Keyword("address"))
    # extract_mode default is "listing"; set "item"/"tiles"/... per analysis

class MySpider(ExtractionSpider):
    name = "mysite"
    args: MyParams

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.args = MyParams.from_spider(self)   # see "late-mutation" note below

    def start_requests(self):
        yield self.make_request(self.start_url, callback=self.parse)

    def parse(self, response):
        yield from self.extract_items_from_page(response)   # the engine entry point

    def validate_result(self, result: Result, columns: Tuple[Keyword, ...]) -> bool:
        # return False to reject a candidate result (e.g. too few fields). Required.
        return sum(c in result for c in columns) >= 2

    def adapt_result(self, result: Result, response):
        # rename/clean/drop fields, post-process values. Required (can be a no-op `pass`).
        ...
```

If `ExtractionSpider` is embedded in a larger crawl framework, you'll often subclass an intermediate
base from that framework — one that already drives the crawl and implements `validate_result` /
`adapt_result` — rather than `ExtractionSpider` directly.

> **Late-mutation note:** `ExtractionSpider.__init__` builds `self.args` from class attributes + `-a`.
> If anything sets the arguments as *instance attributes after* base `__init__` (a runtime per-site
> config system, a mid-MRO mixin, post-`__init__` logic), rebuild at the end:
> `self.args = MyParams.from_spider(self)` reading instance attrs.

## Tuning knobs (reduce noise / improve recall)

`constraints`, `constraints_overrides`, `value_filters`, `required_fields`, `dedupe_keywords`,
`additional_regexes`, `fill_fields`, `multiline_fields`, `listing_fields`, `drop_fields`,
`drop_items`, `max_tables`. What each does and when to reach for it, plus the `ExtractTextResponse`
primitives (`markdown`, `text_re`/`tid`, the `add_text_re` markdown selector) and how to **test
selectors in `scrapy shell`**: [references/markdown-extraction.md](references/markdown-extraction.md).

## References

- [references/extract-modes.md](references/extract-modes.md) — per-mode analysis & selection guidance.
- [references/markdown-extraction.md](references/markdown-extraction.md) — `ExtractTextResponse`
  primitives, the tuning args, selector testing, dataset generation.
- emodels [README](../../README.md) — the markdown-domain rationale and low-level selector model
  (`scrapyutils`, `add_text_re`, datasets/transformers).

Before choosing the method or a mode, do a thorough **site analysis** — page types, data sources,
hidden JSON/XHR APIs, pagination, anti-bot — and prefer a structured source when one clearly exists.
