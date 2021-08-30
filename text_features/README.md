# Synopsis

text feature extraction for English and German. Parts of LIWC and more.

# Installation

```
    $ virtualenv --python=python3 venv/
    $ source venv/bin/activate
    (venv) $ pip install -r requirements.txt
    (venv) $ python -m nltk.downloader averaged_perceptron_tagger
    (venv) $ python -m nltk.downloader wordnet
    (venv) $ python -m spacy download en_core_web_sm
    (venv) $ python -m spacy download de_core_news_sm
```

# Run

```
import text_features as tf
myText = "This is a happy little first sentence of a short example text. And now we will add to this sentence a second one right here. Hope you enjoyed reading."
fex = tf.PsyText(language="eng")
feat = fex.extract_from_string(myText)
```

# documentation

* [feature description](features.md)
* most feature names depend on the [word category specifications](dictionaries/word_categories_eng.json)
    * i.e. for each key specified in this dict a feature is generated
    * the key's `_pos` suffix indicates, that POS labels and not word tokens have to match. This suffix is not taken over into the feature name.
* **n.b.** since word category specification elements are language-specific, features sets are too.

# external sources

* the sentiment dictionaries are based on SentiWord (English) and SentiWS (German)
* they are imported to `dictionaries/sentiment_X.pkl` by `prepare_sentiment_dict.py`

# TODO

* change English text processing pipeline from `nltk` to `spaCy`.
