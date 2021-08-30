import os
import json
import pickle
import re
from collections import Counter
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
import spacy

class PsyText():

    r"""psycholinguistic text feature extractor"""

    def __init__(self, language):

        ''' initialize feature extractor.
        Args:
        language: ISO 639-3 language code (deu, eng)
        '''

        self.language=language
        with open("{}/text_features/dictionaries/word_categories_{}.json".format(pwd(), language), "r") as h:
            self.param = json.load(h)
            if "map_pos" not in self.param:
                self.param["map_pos"] = {}
            if "ratios" not in self.param:
                self.param["ratios"] = {}
        with open("{}/text_features/dictionaries/sentiment_{}.pkl".format(pwd(), language), "rb") as h:
            self.senti = pickle.load(h)
        with open("{}/text_features/dictionaries/mappings_{}.json".format(pwd(), language), "r") as h:
            self.mapping = json.load(h)

        if language == "deu":
            self.nlp = spacy.load("de_core_news_sm")
        #elif language == "eng":
        #   self.nlp = spacy.load("en_core_web_sm")
        else:
            self.nlp = None


    def extract_from_index_with_grouping(self, index, column, grouping, normalize=True, cache_path=None):

        ''' extracts linguistic features from text in column COLUMN of dataframe INDEX.
        Args:
        index: (pd.Dataframe)
        column: (str) column name from which to take text. Assumes cells to contain text STRINGS!
        grouping: (str or list/array) column name from where to take grouping or grouping list of same
            length as index.shape[0]
        normalize: (boolean) normalize counts to text length (w/o punctuation; within each grouping level)
        cache_path:  (str) file name (pickle) were to store resulting dataframe
        df: (pd.DataFrame) with psycholinguistic features (1 row per grouping level)
            and index given by grouping levels
            with respective grouping level
        '''

        if cache_path and os.path.isfile(cache_path):
            df = pd.read_pickle(cache_path)
            return df
        
        if isinstance(grouping, str):
            grp = index[grouping].to_numpy()
        else:
            grp = np.asarray(grouping)

        levels = sorted(set(grp))

        df = pd.DataFrame()
        for lev in levels:
            i = np.where(grp==lev)[0]
            text = index[column].iloc[i].to_list()
            text = " ".join(text)
            dfl = self.extract_from_string(text, normalize=normalize)
            df = pd.concat([df, dfl], ignore_index=True)

        df.index = levels

        if cache_path:
            df.to_pickle(cache_path)

        return df
        
            
    def extract_from_string(self, text, normalize=True, cache_path=None):

        ''' extracts linguistic features from string
        Args:
        text: (str) text string
        normalize: (boolean) normalize counts to text length (w/o punctuation)
        cache_path: (str) file name (pickle) were to store resulting dataframe
        Returns:
        df: (pd.DataFrame) with 1 row of psycholinguistic features
        '''
        
        if cache_path and os.path.isfile(cache_path):
            df = pd.read_pickle(cache_path)
            return df

        df = text_features(text, nrm=normalize,
                           lng=self.language, param=self.param,
                           senti=self.senti, mapping=self.mapping,
                           nlp=self.nlp)

        if cache_path:
            df.to_pickle(cache_path)

        return df
        

def text_features(s, lng="eng", nrm=True, param=None, senti=None, mapping=None, nlp=None):

    ''' text features, partly from LIWC 
    Args:
    s: (string) of text
    lng: (string) "eng" or "deu"
    nrm: (boolean) normalize counts to text length (w/o punctuation)
    param: (dict) of word categories
    senti: (dict) sentiment
    mapping: (dict) POS mapping
    Returns
    df: (pd dataframe) with 1 row of features
    '''
    
    # parameters: word categories, sentiment, pos-mappings
    if param is None:
        with open("{}/text_features/dictionaries/word_categories_{}.json".format(pwd(), lng), "r") as h:
            param = json.load(h)
    if senti is None:
        with open("{}/text_features/dictionaries/sentiment_{}.pkl".format(pwd(), lng), "rb") as h:
            senti = pickle.load(h)
    if mapping is None:
        with open("{}/text_features/dictionaries/mappings_{}.json".format(pwd(), lng), "r") as h:
            mapping = json.load(h)
    if nlp is None:
        if lng == "deu":
            nlp = spacy.load("de_core_news_sm")
        # not yet required for English
        #else:
        #    nlp = spacy.load("en_core_web_sm")
        
    # tokenization, lemmatization, pos tagging
    if lng == "eng":
        tup, lemma_cat = process_eng(s, param, mapping, nlp)
    elif lng == "deu":
        tup, lemma_cat = process_deu(s, param, mapping, nlp)

    #!x
    #input(tup)
        
    # event counter
    cc = init_counter(param)
    
    for x in tup:
        cc = update_counter(cc, x, param, senti)

    # infer features
    cc = inferrable_features(cc, param)
        
    # normalize counts
    if nrm:
        cc = normalize_counter(cc)

    # add global ratios (type/token etc)
    cc = add_ratios(cc, lemma_cat)

    # normalize, convert to dataframe
    df = counter2df(cc, nrm, param)
    return df


def process_deu(s, param, mapping, nlp):

    '''
    German text processing
    Args:
    s: (str) input text
    param: (dict) word categories
    mapping: (dict) POS mapping
    Returns:
    tup: [(token, pos, lemma, mc), ...]  (mc is pos major class)
    lemma_cat: dict cw|fw -> list of corresp lemmas
    '''

    lemma_cat = {"fw": [],
                 "cw": []}
    
    doc = nlp(s)
    tup = []

    for token in doc:
        tok = token.text.lower()
        pos = token.pos_
        lem = token.lemma_.lower()
        # major class
        if pos in mapping["pos"]:
            mc = mapping["pos"][pos]
        else:
            mc = ""
            
        # pos mapping (e.g. PRON to PREL)
        pos = pos_map(pos, tok, param["map_pos"])
                
        tup.append((token.text, pos, lem, mc))

        if pos in param["content_words_pos"]:
            lemma_cat["cw"].append(lem)
        elif re.search("[A-Z]+", pos):
            lemma_cat["fw"].append(lem)

    #input(tup)
            
    return tup, lemma_cat
    

def process_eng(s, param, mapping, nlp):

    '''
    English text processing
    Args:
    s: (str) input text
    param: (dict) word categories
    mapping: (dict) POS mapping
    Returns:
    tup: [(token, pos, lemma, mc), ...]   (mc is pos major class)
    lemma_cat: dict cw|fw -> list of corresp lemmas
    '''
    
    text = nltk.word_tokenize(s)
    tup = nltk.pos_tag(text)
    tup, lemma_cat = lemmatize_eng(tup, param, mapping)

    #input(tup)
    return tup, lemma_cat


def pos_map(pos, tok, pm):
    ''' map POS label e.g. to more fine-grained category '''
    if pos in pm and tok in pm[pos]["tok"]:
        return pm[pos]["tar"]
    return pos


def lemmatize_eng(tup, param, mapping):

    ''' adds lemma and major class POS to (text, pos) tuples
    for later lookups in sentiment lexicon etc.
    Args:
    tup: [(token, pos), ...]
    Returns:
    tup: [(token, pos, lemma, mc), ...]
    lemma_cat: dict cw|fw -> list of corresp lemmas
    '''

    lemma_cat = {"fw": [],
                 "cw": []}
    
    myLemmatizer = nltk.stem.WordNetLemmatizer()
    for i in range(len(tup)):

        tok, pos = tup[i][0], tup[i][1]
        pos = pos_map(pos, tok, param["map_pos"])
        
        # lemma
        lem = myLemmatizer.lemmatize(tok)
        lem = lem.lower()
        # pos major class: n, v, a, r, ""
        if pos in mapping["pos"]:
            mc = mapping["pos"][pos]
        else:
            mc = ""
        tup[i] += (lem, mc)

        # lemma type sets
        if pos in param["content_words_pos"] and \
           tok not in param["auxiliary_verbs"]:
            lemma_cat["cw"].append(lem)
        elif re.search("[A-Z]+", pos):
            lemma_cat["fw"].append(lem)
            
    return tup, lemma_cat


def init_counter(param):

    ''' initialize Counter object;
    needed so that output df has constant number of columns
    regardless whether events have been observed or not '''

    cc = Counter()
    for x in param:
        # needed for English verb disambiguation only
        if x == "auxiliary_verbs":
            continue
        # skip pos mapping rules and later ratio calculations
        if x in ["map_pos", "ratios"]:
            continue
        u = re.sub("_pos$", "", x)
        cc[u] = 0

    for x in ["sentiment_negative", "sentiment_positive", "sentiment"]:
        cc[x] = 0

    return cc


def update_counter(cc, x, param, senti):

    ''' update counter fro keys in params
    Args:
    cc: (Counter) counter
    x: (tuple) (token, pos, lemma, majorPosClass)
    param: (dict) word categories
    senti: (dict) sentiment
    Returns:
    cc with incremented counts
    '''
    
    tok, pos, lem, mc = x[0].lower(), x[1], x[2], x[3]
    senti_key = lem + "#" + mc
    senti_key2 = tok.lower() + "#" + mc
    
    if not re.search("[A-Z0-9]+", pos):
        return cc

    cc["N"] += 1

    # word categories    
    if len(tok) >= param["longer_than"]:
        cc["longer_than"] += 1

    for x in param:
        
        if type(param[x]) is not list:
            continue
        if re.search("_pos$", x):

            # special treatment for auxiliary verbs
            # (not specially POS-marked in eng)
            if x == "content_words_pos" and tok in param["auxiliary_verbs"]:
                continue
            
            if pos in param[x]:
                u = re.sub("_pos$", "", x)
                if u not in cc:
                    continue
                cc[u] += 1
        else:
            if (tok in param[x] or lem in param[x]):
                if x not in cc:
                    continue
                cc[x] += 1

    # sentiment
    sk = None
    if senti_key in senti:
        sk = senti_key
    elif senti_key2 in senti:
        sk = senti_key2
    if sk:
        cc["sentiment"] += 1
        if senti[sk] > 0:
            cc["sentiment_positive"] += senti[sk]
        else:
            cc["sentiment_negative"] += np.abs(senti[sk])
            
    return cc


def inferrable_features(cc, param):

    ''' adds features that can be inferred from others '''
    
    cc["function_words"] = cc["N"] - cc["content_words"]
    cc["main_verbs"] == cc["verbs"] - cc["auxiliary_verbs"]

    # aggregate over pronouns_1s, pronouns_1p, pronouns_2, pronouns_3
    cc["pronouns_1s"] = 0
    cc["pronouns_1p"] = 0
    cc["pronouns_2"] = 0
    cc["pronouns_3"] = 0
    
    for u in ["personal", "possessive"]:
        for v in ["1s", "1p", "2", "2s", "2p", "3s", "3p"]:
            src_key = f"{u}_pronouns_{v}"
            if src_key not in cc:
                continue
            if v == "1s":
                key = "pronouns_1s"
            elif v == "1p":
                key = "pronouns_1p"
            elif re.search("2", v):
                key = "pronouns_2"
            else:
                key = "pronouns_3"
            cc[key] += cc[src_key]
    
    # count ratios
    for p in param["ratios"]:
        nom = cc[param["ratios"][p]["nom"]]
        denom = cc[param["ratios"][p]["denom"]]
        key = f"{p}_ratio"
        if denom == 0:
            cc[key] = 0
        else:
            cc[key] = nom/denom
                 
    return cc

    
def normalize_counter(cc):

    ''' normalize counts to text length '''

    for x in cc:
        if x in ["N"] or re.search("_ratio$", x):
            continue
        cc[x] = cc[x] / cc["N"]

    return cc


def add_ratios(cc, lemma_cat):

    ''' adds type/token ratios for all, function, and content words '''
    
    for typ in ["cw", "fw"]:
        fld = "{}_typeTokenRatio".format(typ)
        if len(lemma_cat[typ]) > 0:
            cc[fld] = len(set(lemma_cat[typ])) / len(lemma_cat[typ])
        else:
            cc[fld] = 0

    lem = lemma_cat["cw"]
    lem.extend(lemma_cat["fw"])

    if len(lem) > 0:
        cc["typeTokenRatio"] = len(set(lem)) / len(lem)
    else:
        cc["typeTokenRatio"] = 0

    return cc


def counter2df(cc, nrm, param):

    ''' complete counter, normalize, and transform to dataframe '''

    for x in cc:
        cc[x] = [cc[x]]
            
    # to dataframe
    df = pd.DataFrame.from_dict(cc)
    df = df.reindex(columns=sorted(cc.keys()))

    return df

    
def pwd():

    ''' this file's directory '''
    
    return os.path.dirname(os.path.abspath(__file__))
