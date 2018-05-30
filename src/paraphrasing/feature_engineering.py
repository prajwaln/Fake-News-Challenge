import os
import re
import nltk
import numpy as np
from sklearn import feature_extraction
from tqdm import tqdm
import operator, math
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

_wnl = nltk.WordNetLemmatizer()


def normalize_word(w):
    return _wnl.lemmatize(w).lower()


def get_tokenized_lemmas(s):
    return [normalize_word(t) for t in nltk.word_tokenize(s)]


def clean(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric

    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()


def remove_stopwords(l):
    # Removes stopwords from a list of tokens
    return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]


def gen_or_load_feats(feat_fn, headlines, bodies, feature_file):
    dir = os.path.dirname(feature_file)
    if not os.path.exists(dir):
        os.makedirs(dir)
    create = False
    if not os.path.isfile(feature_file): create = True
    else:
        X = np.load(feature_file)
        if X.shape[0] != len(headlines): create = True
    if create:
        print("Creating {}...".format(feature_file))
        feats = feat_fn(headlines, bodies)
        np.save(feature_file, feats)
        X = np.load(feature_file)

    return X


def get_commonwords(l):
    # Identifies common, homogeneous words across multiple articles
    # Added by Julian
    counts = {}
    for headline, body in l:
        article = headline + body
        article_words = {}
        for word in article:
            if word not in article_words: article_words[word] = 0
            article_words[word] += 1
        for word, count in article_words.items():
            if word not in counts: counts[word] = []
            counts[word].append(count/len(article))
    for word, c_list in counts.items():
        mean = sum(c_list)/len(l)
        sd = math.sqrt(sum([(x-mean)**2 for x in c_list])/len(l))
        counts[word] = sd/mean
    sorted_counts = sorted(counts.items(), key=operator.itemgetter(1))
    return [k for k,v in sorted_counts[:int(0.005 * len(sorted_counts))]]

# Paraphrase related code ++
# Added by Prajwal

def tag(sentence):
    words = word_tokenize(sentence)
    words = pos_tag(words)
    return words

def paraphraseable(tag):
    return tag.startswith('NN') or tag == 'VB' or tag.startswith('JJ')

def pos(tag):
    if tag.startswith('NN'):
        return wn.NOUN
    elif tag.startswith('V'):
        return wn.VERB

def synonyms(word, tag):
    lemma_lists = [ss.lemmas() for ss in wn.synsets(word, pos(tag))]
    lemmas = [lemma.name() for lemma in sum(lemma_lists, [])]
    return set(lemmas)

# Paraphrase related code --   
    
def paraphrase_line(headline):
    # Extends the words in headline with paraphrases for each original word
    # Added by Prajwal
    hl_set = set(headline)
    paraphrases = []
    for (w, t) in tag(headline):
        if paraphraseable(t):
            syns = synonyms(w, t)
            if syns:
                if len(syns) > 1:
                    new_words = syns - hl_set                    
                    paraphrases.extend(new_words)  
    return paraphrases  

def word_overlap_features(headlines, bodies):
    X = []
    articles = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        clean_headline = get_tokenized_lemmas(clean_headline)
        clean_body = get_tokenized_lemmas(clean_body)
        articles.append((clean_headline, clean_body))
    common_words = get_commonwords(articles)
    for clean_headline, clean_body in articles:
        clean_headline = [x for x in clean_headline if x not in common_words]
        clean_body = [x for x in clean_body if x not in common_words]
        features = [
            len(set(clean_headline).intersection(clean_body)) / float(len(set(clean_headline).union(clean_body)))]
        X.append(features)
    return X


def refuting_features(headlines, bodies):
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        # 'refute',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_headline = get_tokenized_lemmas(clean_headline)
        features = [1 if word in clean_headline else 0 for word in _refuting_words]
        X.append(features)
    return X


def polarity_features(headlines, bodies):
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        # 'refute',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]

    def calculate_polarity(text):
        tokens = get_tokenized_lemmas(text)
        return sum([t in _refuting_words for t in tokens]) % 2
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        features = []
        features.append(calculate_polarity(clean_headline))
        features.append(calculate_polarity(clean_body))
        X.append(features)
    return np.array(X)


def ngrams(input, n):
    input = input.split(' ')
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def chargrams(input, n):
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def append_chargrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in chargrams(" ".join(remove_stopwords(text_headline.split())), size)]
    grams_hits = 0
    grams_early_hits = 0
    grams_first_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
        if gram in text_body[:100]:
            grams_first_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    features.append(grams_first_hits)
    return features


def append_ngrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in ngrams(text_headline, size)]
    grams_hits = 0
    grams_early_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    return features


def hand_features(headlines, bodies):

    def binary_co_occurence(headline, body):
        # Count how many times a token in the title
        # appears in the body text.
        bin_count = 0
        bin_count_early = 0
        for headline_token in headline.split(" "):
            if headline_token in body:
                bin_count += 1
            if headline_token in body[:255]:
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def binary_co_occurence_stops(headline, body):
        # Count how many times a token in the title
        # appears in the body text. Stopwords in the title
        # are ignored.
        bin_count = 0
        bin_count_early = 0
        for headline_token in remove_stopwords(headline.split(" ")):
            if headline_token in body:
                bin_count += 1
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def binary_co_occurence_paraphrase(headline, body):
        # Count how many times a word in the headline or its paraphrase
        # appears in the body text.
        # Added by Prajwal
        bin_count = 0
        bin_count_early = 0
        for headline_token in paraphrase_line(headline):
            if headline_token in body:
                bin_count += 1
            if headline_token in body[:255]:
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def count_grams(headline, body):
        # Count how many times an n-gram of the title
        # appears in the entire body, and intro paragraph

        features = []
        features = append_chargrams(features, clean_headline, clean_body, 2)
        features = append_chargrams(features, clean_headline, clean_body, 8)
        features = append_chargrams(features, clean_headline, clean_body, 4)
        features = append_chargrams(features, clean_headline, clean_body, 16)
        features = append_ngrams(features, clean_headline, clean_body, 2)
        features = append_ngrams(features, clean_headline, clean_body, 3)
        features = append_ngrams(features, clean_headline, clean_body, 4)
        features = append_ngrams(features, clean_headline, clean_body, 5)
        features = append_ngrams(features, clean_headline, clean_body, 6)
        return features

    X = []
    articles = []
    lemmas = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        articles.append((clean_headline, clean_body))
        clean_headline_l = get_tokenized_lemmas(clean_headline)
        clean_body_l = get_tokenized_lemmas(clean_body)
        lemmas.append((clean_headline_l, clean_body_l))
    common_words = r'(\b{}\b)'.format('\\b|\\b'.join(get_commonwords(lemmas)))
    for clean_headline, clean_body in articles:
        clean_headline = re.sub(common_words, '', clean_headline)
        clean_body = re.sub(common_words, '', clean_body)
        X.append(binary_co_occurence(clean_headline, clean_body)
                 + binary_co_occurence_stops(clean_headline, clean_body)
                 + binary_co_occurence_paraphrase(clean_headline, clean_body)
                 + count_grams(clean_headline, clean_body))


    return X
