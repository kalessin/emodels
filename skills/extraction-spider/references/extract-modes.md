# Extraction modes — analysis & selection

`extract_mode` selects which algorithm `extract_items_from_page` runs
([emodels/extract/spider.py](../../../emodels/extract/spider.py)). Choosing it is the heart of the
analysis. Two engines underlie the modes:

- **Table engine** (`emodels/extract/table.py`, `parse_tables_from_response`) — operates on real HTML
  `<table>` elements via **xpath**: finds tables whose header row best matches your `fields`, then
  parses rows into records. Used by `listing`/`hybrid`/`combined`.
- **Clustering engine** (`emodels/extract/cluster.py`, `extract_by_keywords`) — operates on the
  **markdown** text: finds your field keywords/labels and clusters nearby text into records. Used by
  `item`/`tiles`/`combined`.

So `listing` is *not* markdown clustering — it needs genuine `<table>` markup. If a "listing" is built
from divs/cards rather than a `<table>`, treat it as **tiles** (repeated blocks) instead.

## The modes

### `listing` (default)
- **When:** the page contains an HTML `<table>` (or several) whose columns correspond to your fields.
- **How:** `parse_tables_from_response` scores candidate tables by header↔`fields` overlap, parses
  the best one(s). Needs **≥ 2 columns**.
- **Knobs:** `listing_fields` (use a *different* field set for the table than for item extraction —
  handy in hybrid/combined to keep listing less noisy); `max_tables` (raise when one logical dataset
  spans multiple tables with different headers); `required_fields`, `dedupe_keywords`.

### `item`
- **When:** one record per page (a profile/detail/quote page), data laid out as labelled text.
- **How:** `extract_by_keywords` over the markdown — finds each field's label and the value on the
  same line; clusters fields into a single best record.
- **Knobs:** `additional_regexes` (precise per-field markdown regexes), `multiline_fields` (let a
  field span N lines, e.g. `{"address": 4}`), `fill_fields`, `value_filters`, `constraints`.

### `tiles`
- **When:** many repeated item blocks on one page (cards, search-result tiles, div-based "tables").
- **How:** `extract_by_keywords(tiles_mode=True)` — KMeans-clusters repeated field patterns into one
  record per tile; returns all tiles matching the pattern. `tiles_mode_tolerance` controls how many
  required fields a tile may miss.
- **Knobs:** same as `item`, plus tolerance. Use `^#`/`^##` style keywords to anchor on markdown
  headings when tiles are heading-delimited.

### `any`
- **When:** a page might be either a listing or a single item, or you're unsure.
- **How:** run `listing` first; if it yields nothing, fall back to `item`. Good default when analysis
  is incomplete.

### `hybrid`
- **When:** a listing page links to per-item detail pages and you want fields from both.
- **How:** `listing` extraction first; for each row, build the item URL from `item_url_template`
  (a python format string fed the row dict) or a `url` field, follow it, and complete the record from
  the item page (`parse_items_from_response`). Dedup is deferred to the item stage.
- **Knobs:** `item_url_template`, `listing_fields` (listing) vs `fields` (item page).

### `combined`
- **When:** a single page holds part of the data in a `<table>` and the rest in surrounding labelled
  text, and you want them merged into one record.
- **How:** runs table extraction + keyword extraction on the **same** response and merges; forces
  `max_tables = max(2, max_tables)`. Returns a single combined `Result`.
- **Knobs:** `max_tables`, `dedupe_keywords`, `value_filters`.

## Picking a mode from analysis

```
Is the target data in an HTML <table>?
├─ yes, one record per row, all fields in the table .............. listing
│   └─ also need fields from each row's detail page? ............. hybrid
│   └─ some fields in the table, some in page text? ............. combined
├─ no <table>, but many repeated blocks/cards on the page ........ tiles
├─ no, single record per page (profile/detail) ................... item
└─ unsure / page type varies ..................................... any
```

Always confirm against the rendered **markdown** (not the browser DOM) — that's what the clustering
engine sees. `response.replace(cls=ExtractTextResponse).markdown` in `scrapy shell` shows it (see
[markdown-extraction.md](markdown-extraction.md)).

## Fields vs listing_fields

- `fields` — the fields to extract; used for item/tiles/combined and as the table columns in listing
  if `listing_fields` is empty.
- `listing_fields` — when set, used as the table columns for listing/hybrid extraction, while `fields`
  is used for the item-page/keyword extraction. Use this when the listing table has a smaller/cleaner
  column set than the per-item page.
