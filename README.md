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
3. As a secondary objective, it provides an alternative kind of selector to xpath and css selectors for extraction of data from the html source, that may be more suitable and readable for humans.
4. In many situations, and specially when there is not an id or a class to spot accurately the text, the expresion in terms of regular expressions in the domain of markdown can be simpler.
5. The additional cost of maintenance is zero. Even if you are introducing a new kind of selector that may be unknown for most people, this only characterizes single lines and does not condition
   any other line of the spider code. If the selector still works, it is not needed to be changed. If it doesn't work anymore, it can be replaced by common and known selectors without affecting
   anything else.

#### Set up

Instead of subclass your item loaders from `scrapy.loader.ItemLoader`, use `emodels.scrapyutils.loader.ExtractItemLoader`. This action will not affect the working of itemloaders and will enable the
properties just described above. In addition, in order to save the collected extraction data, it is required to set the environment variable `EMODELS_SAVE_EXTRACT_ITEMS` to 1. The collected
extraction data will be stored at `<user home folder>/.datasets/items/<item class name>/<sequence number>.jl.gz`. The base folder `<user home folder>/.datasets` is the default one. You can
customize it via the environment variable `EMODELS_REPOSITORY`.

So, in order to maintain a clean and organized dataset, only enable extract items saving when you are sure you have the correct extraction selectors. Then run locally:

```
EMODELS_SAVE_EXTRACT_ITEMS=1 scrapy crawl myspider
```

In addition, in order to have your dataset organized, you may want to choose the same item class name for same item schema, even accross multiple projects. And avoid to repeat it among items with different
schema. However, in general you will use extraction data from all classes of items at same time in order to train a transformer model, as this is the way how transformers learn to generalize. In addition
avoid to create multiple collections for same item class and target site, as they will not provide diversity and you may commit the mistake of having same sample distribution in train and test datasets.

At the end you will have a transformer model that is suited to extract any kind of item, as they are trained not to extract "data from x item" but instead to recognize and extract based on fields.
So, even if you didn't train the transformer to extract a specific item class, it will do great if you trained it to extract its fields, if it already learned to extract same fields from
other item classes. You only need to ask the correct question. For example, given an html page as a context, you can ask the model: `which is the phone number?`. You don't need to specify
which kind of data (a business? a person? an organization?) you expect to find there.

For instructions on how to create datasets for training your transformers, go to the datasets module sections.

#### Markdown Selectors

The new kind of selector is based on regular expressions, but it also admits id and class specifications for further guiding the selector extraction. Many examples can be seen in
`tests/test_scrapyutils.py`. This selector does not operate on the html source, but on a text rendered (markdown) version of it. The loader function that applies this selector is
(copied from `emodels/scrapyutils/loaders.py`):


```python
    def add_text_re(
        self,
        attr: str,
        reg: str = "(.+?)",
        tid: Optional[str] = None,
        flags: int = 0,
        skip_prefix: str = DEFAULT_SKIP_PREFIX,
        idx: int = 0,
        *processors,
        **kw,
    ):
        """
        attr - item attribute where selector extracted value will be assigned to.
        reg - Optional regular expression (it is optional because you can also use tid alone)
        tid - Optional css id or class specification (start with either # or .). When this parameter is present,
              regular expression match will be restricted to the text region with the specified id or class. Note:
              when the tid string starts with #, it is also able to match the itemprop attribute, not only the id.
        flags - Optional regular expression flag
        skip_prefix - This prefix is added to the provided regular expression in order to skip it from the result.
              The default one is any non alphanumeric character at begining of the line and in most cases
              you will use this value. Provided for convenience, in order to avoid to repeat it frequently
              in the regular expression parameter, making it more natural.
        idx - Regex selectors only return a single match, and by default it is the first one (idx=0). If you want
              instead to extract a different match, set the appropiate index with this parameter.
        *processors - Extraction processors passed to the method (same as in usual loaders)
        **kw - Additional extract parameters passed to the method (same as in usual loaders)
        """
```

The docstring aim is to be enough explicative, so no need to repeat here.

#### Testing selectors

In most cases, as usual with selectors, you will first need to test different selectors in order to set the definitive one to hardcode in the spider. Typically with css and xpath selectors,
there are three alternatives used by developers:

1. The browser development console
2. The fetch command in the `scrapy shell` console
3. Put testing code in the spider to save responses and open it from a python console

Each one has its pros and cons, and which to use depends on specific needs.

1. For example, the browser development console is readily accesible but frequently the page rendered in the browser is not exactly the one that will be downloaded by the spider
(not only because of rendering itself, but also bans, session issues, etc). However, when applicable, it allows to easily identify id, itemprop and class attributes that can be
readily used in the `add_text_re()` `tid` parameter. In many situations, however, it is not as straighforward as that, and you may get unexpected results, dirty extraction with
undesired characters, etc. However, with experience you will be able to make most of the extraction using only the browser. See instructions on usage of tid parameter in the selector
doctest.

2. Scrapy shell

```
$ scrapy shell
> fetch(<url>)
> from emodels.scrapyutils.response import ExtractTextResponse
> response = response.replace(cls=ExtractTextResponse)
```

The newly created response has available the method `text_re()` for testing extraction with markdown selectors.


3. Put testing code in the spider to save responses and open it from a python console


```

```


## datasets module

This module contains utils for creating datasets and training models.

#### 1. Creating a dataset from collected items.

As we mentioned before, the way to collect extraction data from a spider run is by running the spider with the appropiate environment variable set:

```bash
EMODELS_SAVE_EXTRACT_ITEMS=1 scrapy crawl myspider
```

This will save in the local datasets directory, within the folder `items/<scrapy item name associacted to the loader>`, the extraction data generated from the the fields extracted
with the `add_text_re()` method. Each time you run a spider in this way, a new file will be generated within the mentioned folder.

At any moment, you can build a joint from all the individual files, with the following lines in a python console:

```python
> from emodels.datasets.utils import ExtractDatasetFilename
> eds = ExtractDatasetFilename.build_from_items("myproject/items")
```

The joint dataset will be saved to the dataset file represented by eds variable.

```python
> eds
'/home/myuser/.datasets/myproject/items.jl.gz'
```

This operation also assigns randomly the samples to train/test/validation buckets, according to the
`dataset_ratio` parameter, which by default assigns 70% to training bucket, 30% to test bucket and 0 to validation bucket.

Provided `EMODELS_DATASET_DIR` is the same, you can also recover back the dataset later:

```python
> eds = ExtractDatasetFilename.local_by_name("myproject/items")
```

If not, you can also do:

```python
> eds = ExtractDatasetFilename("/home/myuser/.datasets/myproject/items.jl.gz")
```

#### 2. Preparing dataset to train a HuggingFace transformer model

Convert extract dataset to HuggingFace DatasetDict:

```python
> from emodels.datasets.hugging import to_hfdataset, prepare_datasetdict
> hf = to_hfdataset(eds)
```

And then prepare the dataset for usage in transformers training:

```python
> from transformers import AutoTokenizer
> tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
> hff = prepare_datasetdict(hf, tokenizer)
```

This preparation includes truncation of samples in order to fit to the target HuggingFace model size (in this case, `deepset/roberta-base-squad2`), and set of appropiate sample fields
required for training.


Both `hf` and `hff` in the examples above are instances of HuggingFace DatasetDict class. So you can save them to disk and recover them at any time. I.e:

```python
> hff.save_to_disk("./my_prepared_dataset")
```

Later, for recovering:

```
> from datasets import DatasetDict
> hff = DatasetDict.load_from_disk("./my_prepared_dataset")
```

(Notice that the `datasets` module here is not the same as `emodels.datasets` module. The former comes from the HuggingFace package.)

#### 3. Get trainer and do the train stuff.

```python
> from emodels.datasets.hugging import get_qatransformer_trainer
> model, trainer, test_data = get_qatransformer_trainer(hff, "deepset/roberta-base-squad2", "mytransformer_cache_dir")
> trainer.train()   # this will take long time
...

> trainer.evaluate(test_data)
```

Once trained, the model can be saved:

```python
> model.save_pretrained("./mytunned_model")
```

Save the tokenizer along with the model, so everything is packed in the same place:


```python
> tokenizer.save_pretrained("./mytunned_model")
```

This will avoid the requirement to manually load the tokenizer on each further operation.

And later be recovered:

```python
> from transformers import AutoModelForQuestionAnswering
> model = AutoModelForQuestionAnswering.from_pretrained("./mytunned_model")
```

#### 4. Extracting with the model.

```python
> from transformers import AutoTokenizer
> tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
> from emodels.datasets.hugging import QuestionAnswerer
> question_answerer = QuestionAnswerer("./mytunned_model")
> question_answerer.answer(question="Which is the name?", context=<target markdown>)
```

#### 5. Evaluating the model.

Optionally, evaluate the extraction with the base untunned model:

```python
> from emodels.datasets.hugging import evaluate
> evaluate(eds.iter(), model_path="deepset/roberta-base-squad2")
```

This will return a score dictionary, one item per dataset bucket. Then, do the same for the tunned model:


```python
> evaluate(eds.iter(), model="./mytunned_model")
```

if everything went ok, the score of the tunned model should be bigger than the pretrained one.

This evaluation is different than the one performed by trainer.evaluate() above. While the last evaluates by comparing
indexes results, which are the output and target of the model, the one here evaluates actual results as extracted text.

Results however should be similar.

## html2text module

This is a modified and tuned copy of the html2text python library, with modifications required to work for the purpose of this package. You won't usually need to import this module directly. It is
mostly there for correct working of the other modules in this package.
