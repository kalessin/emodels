e-models
========


Suite of tools to assist in the build of extraction models with scrapy spiders

Installation:

```
$ pip install e-models
```

## scrapyutils module


scrapyutils module provides two classes, one for extending `scrapy.http.TextResponse` and another for
extending `scrapy.loader.ItemLoader`. The extensions provide methods that:

1. Allow to extract item data in the text (markdown) domain instead of the html source domain.
2. The main purpose of this approach is the generation of datasets suitable for training transformer models for text extraction (aka extractive question answering, EQA)
3. As a secondary objective, it provides an alternative approach to xpath and css selectors for extraction of data from the html source, that may be more suitable and readable for humans.

Usage:

Instead of subclass your item loaders from `scrapy.Item`, use `emodels.scrapyutils.ExtractItemLoader`. This action will not affect the working of itemloaders and will enable the properties just
described above. In addition, in order to save the collected extraction data, it is required to set the environment variable `EMODELS_SAVE_EXTRACT_ITEMS` to 1. The collected
extraction data will be incrementally stored at `<user home folder>/.datasets/items/<item class name>.jl.gz`. The base folder `<user home folder>/.datasets` is the default one. You can
customize it via the environment variable `EMODELS_DIR`.

So, in order to maintain dataset consistence you should choose the same item class name for same item schema, even accross multiple projects. And avoid to repeat it among items with different
schema.
