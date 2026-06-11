# Claude Code skills

[Claude Code](https://claude.com/claude-code) skills shipped with e-models. They give Claude working
knowledge of the emodels extraction tooling so it can help you build and tune it correctly.

Skills are not loaded automatically — copy the ones you want into your **personal** Claude scope:

```
cp -r skills/extraction-spider ~/.claude/skills/
```

(or symlink it, if you'd rather track this checkout). Each skill is a self-contained directory; only
its `name` + `description` stay in context until Claude decides it's relevant, so installing a skill
you don't use costs nothing.

## Available skills

- **extraction-spider** — building and extending scraping spiders with the emodels extraction method
  (`emodels.extract.spider.ExtractionSpider`): the markdown/text + clustering approach (not xpath/css)
  built on `ExtractTextResponse`. Covers when it's the right tool vs. a cheaper structured source, the
  extraction modes (listing/item/tiles/any/hybrid/combined), how to create a concrete spider, and the
  markdown primitives and tuning knobs behind it. Tightly coupled to site analysis. See
  [extraction-spider/SKILL.md](extraction-spider/SKILL.md).
