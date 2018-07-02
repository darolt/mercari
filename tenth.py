import multiprocessing as mp
import pandas as pd
from time import time
from scipy.sparse import csr_matrix
import os
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import gc
from sklearn.base import BaseEstimator, TransformerMixin
import re
from pandas.api.types import is_numeric_dtype, is_categorical_dtype

os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['JOBLIB_START_METHOD'] = 'forkserver'

INPUT_PATH = r'../input'


def dameraulevenshtein(seq1, seq2):
    """Calculate the Damerau-Levenshtein distance between sequences.

    This method has not been modified from the original.
    Source: http://mwh.geek.nz/2009/04/26/python-damerau-levenshtein-distance/

    This distance is the number of additions, deletions, substitutions,
    and transpositions needed to transform the first sequence into the
    second. Although generally used with strings, any sequences of
    comparable objects will work.

    Transpositions are exchanges of *consecutive* characters; all other
    operations are self-explanatory.

    This implementation is O(N*M) time and O(M) space, for N and M the
    lengths of the two sequences.

    >>> dameraulevenshtein('ba', 'abc')
    2
    >>> dameraulevenshtein('fee', 'deed')
    2

    It works with arbitrary sequences too:
    >>> dameraulevenshtein('abcd', ['b', 'a', 'c', 'd', 'e'])
    2
    """
    # codesnippet:D0DE4716-B6E6-4161-9219-2903BF8F547F
    # Conceptually, this is based on a len(seq1) + 1 * len(seq2) + 1 matrix.
    # However, only the current and two previous rows are needed at once,
    # so we only store those.
    oneago = None
    thisrow = list(range(1, len(seq2) + 1)) + [0]
    for x in range(len(seq1)):
        # Python lists wrap around for negative indices, so put the
        # leftmost column at the *end* of the list. This matches with
        # the zero-indexed strings and saves extra calculation.
        twoago, oneago, thisrow = (oneago, thisrow, [0] * len(seq2) + [x + 1])
        for y in range(len(seq2)):
            delcost = oneago[y] + 1
            addcost = thisrow[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            thisrow[y] = min(delcost, addcost, subcost)
            # This block deals with transpositions
            if (x > 0 and y > 0 and seq1[x] == seq2[y - 1]
                    and seq1[x - 1] == seq2[y] and seq1[x] != seq2[y]):
                thisrow[y] = min(thisrow[y], twoago[y - 2] + 1)
    return thisrow[len(seq2) - 1]


class SymSpell:
    def __init__(self, max_edit_distance=3, verbose=0):
        self.max_edit_distance = max_edit_distance
        self.verbose = verbose
        # 0: top suggestion
        # 1: all suggestions of smallest edit distance
        # 2: all suggestions <= max_edit_distance (slower, no early termination)

        self.dictionary = {}
        self.longest_word_length = 0

    def get_deletes_list(self, w):
        """given a word, derive strings with up to max_edit_distance characters
           deleted"""

        deletes = []
        queue = [w]
        for d in range(self.max_edit_distance):
            temp_queue = []
            for word in queue:
                if len(word) > 1:
                    for c in range(len(word)):  # character index
                        word_minus_c = word[:c] + word[c + 1:]
                        if word_minus_c not in deletes:
                            deletes.append(word_minus_c)
                        if word_minus_c not in temp_queue:
                            temp_queue.append(word_minus_c)
            queue = temp_queue

        return deletes

    def create_dictionary_entry(self, w):
        '''add word and its derived deletions to dictionary'''
        # check if word is already in dictionary
        # dictionary entries are in the form: (list of suggested corrections,
        # frequency of word in corpus)
        new_real_word_added = False
        if w in self.dictionary:
            # increment count of word in corpus
            self.dictionary[w] = (self.dictionary[w][0], self.dictionary[w][1] + 1)
        else:
            self.dictionary[w] = ([], 1)
            self.longest_word_length = max(self.longest_word_length, len(w))

        if self.dictionary[w][1] == 1:
            # first appearance of word in corpus
            # n.b. word may already be in dictionary as a derived word
            # (deleting character from a real word)
            # but counter of frequency of word in corpus is not incremented
            # in those cases)
            new_real_word_added = True
            deletes = self.get_deletes_list(w)
            for item in deletes:
                if item in self.dictionary:
                    # add (correct) word to delete's suggested correction list
                    self.dictionary[item][0].append(w)
                else:
                    # note frequency of word in corpus is not incremented
                    self.dictionary[item] = ([w], 0)

        return new_real_word_added

    def create_dictionary_from_arr(self, arr, token_pattern=r'[a-z]+'):
        total_word_count = 0
        unique_word_count = 0

        for line in arr:
            # separate by words by non-alphabetical characters
            words = re.findall(token_pattern, line.lower())
            for word in words:
                total_word_count += 1
                if self.create_dictionary_entry(word):
                    unique_word_count += 1

        print("total words processed: %i" % total_word_count)
        print("total unique words in corpus: %i" % unique_word_count)
        print("total items in dictionary (corpus words and deletions): %i" % len(self.dictionary))
        print("  edit distance for deletions: %i" % self.max_edit_distance)
        print("  length of longest word in corpus: %i" % self.longest_word_length)
        return self.dictionary

    def create_dictionary(self, fname):
        total_word_count = 0
        unique_word_count = 0

        with open(fname) as file:
            for line in file:
                # separate by words by non-alphabetical characters
                words = re.findall('[a-z]+', line.lower())
                for word in words:
                    total_word_count += 1
                    if self.create_dictionary_entry(word):
                        unique_word_count += 1

        print("total words processed: %i" % total_word_count)
        print("total unique words in corpus: %i" % unique_word_count)
        print("total items in dictionary (corpus words and deletions): %i" % len(self.dictionary))
        print("  edit distance for deletions: %i" % self.max_edit_distance)
        print("  length of longest word in corpus: %i" % self.longest_word_length)
        return self.dictionary

    def get_suggestions(self, string, silent=False):
        """return list of suggested corrections for potentially incorrectly
           spelled word"""
        if (len(string) - self.longest_word_length) > self.max_edit_distance:
            if not silent:
                print("no items in dictionary within maximum edit distance")
            return []

        suggest_dict = {}
        min_suggest_len = float('inf')

        queue = [string]
        q_dictionary = {}  # items other than string that we've checked

        while len(queue) > 0:
            q_item = queue[0]  # pop
            queue = queue[1:]

            # early exit
            if ((self.verbose < 2) and (len(suggest_dict) > 0) and
                    ((len(string) - len(q_item)) > min_suggest_len)):
                break

            # process queue item
            if (q_item in self.dictionary) and (q_item not in suggest_dict):
                if self.dictionary[q_item][1] > 0:
                    # word is in dictionary, and is a word from the corpus, and
                    # not already in suggestion list so add to suggestion
                    # dictionary, indexed by the word with value (frequency in
                    # corpus, edit distance)
                    # note q_items that are not the input string are shorter
                    # than input string since only deletes are added (unless
                    # manual dictionary corrections are added)
                    assert len(string) >= len(q_item)
                    suggest_dict[q_item] = (self.dictionary[q_item][1],
                                            len(string) - len(q_item))
                    # early exit
                    if (self.verbose < 2) and (len(string) == len(q_item)):
                        break
                    elif (len(string) - len(q_item)) < min_suggest_len:
                        min_suggest_len = len(string) - len(q_item)

                # the suggested corrections for q_item as stored in
                # dictionary (whether or not q_item itself is a valid word
                # or merely a delete) can be valid corrections
                for sc_item in self.dictionary[q_item][0]:
                    if sc_item not in suggest_dict:

                        # compute edit distance
                        # suggested items should always be longer
                        # (unless manual corrections are added)
                        assert len(sc_item) > len(q_item)

                        # q_items that are not input should be shorter
                        # than original string
                        # (unless manual corrections added)
                        assert len(q_item) <= len(string)

                        if len(q_item) == len(string):
                            assert q_item == string
                            item_dist = len(sc_item) - len(q_item)

                        # item in suggestions list should not be the same as
                        # the string itself
                        assert sc_item != string

                        # calculate edit distance using, for example,
                        # Damerau-Levenshtein distance
                        item_dist = dameraulevenshtein(sc_item, string)

                        # do not add words with greater edit distance if
                        # verbose setting not on
                        if (self.verbose < 2) and (item_dist > min_suggest_len):
                            pass
                        elif item_dist <= self.max_edit_distance:
                            assert sc_item in self.dictionary  # should already be in dictionary if in suggestion list
                            suggest_dict[sc_item] = (self.dictionary[sc_item][1], item_dist)
                            if item_dist < min_suggest_len:
                                min_suggest_len = item_dist

                        # depending on order words are processed, some words
                        # with different edit distances may be entered into
                        # suggestions; trim suggestion dictionary if verbose
                        # setting not on
                        if self.verbose < 2:
                            suggest_dict = {k: v for k, v in suggest_dict.items() if v[1] <= min_suggest_len}

            # now generate deletes (e.g. a substring of string or of a delete)
            # from the queue item
            # as additional items to check -- add to end of queue
            assert len(string) >= len(q_item)

            # do not add words with greater edit distance if verbose setting
            # is not on
            if (self.verbose < 2) and ((len(string) - len(q_item)) > min_suggest_len):
                pass
            elif (len(string) - len(q_item)) < self.max_edit_distance and len(q_item) > 1:
                for c in range(len(q_item)):  # character index
                    word_minus_c = q_item[:c] + q_item[c + 1:]
                    if word_minus_c not in q_dictionary:
                        queue.append(word_minus_c)
                        q_dictionary[word_minus_c] = None  # arbitrary value, just to identify we checked this

        # queue is now empty: convert suggestions in dictionary to
        # list for output
        if not silent and self.verbose != 0:
            print("number of possible corrections: %i" % len(suggest_dict))
            print("  edit distance for deletions: %i" % self.max_edit_distance)

        # output option 1
        # sort results by ascending order of edit distance and descending
        # order of frequency
        #     and return list of suggested word corrections only:
        # return sorted(suggest_dict, key = lambda x:
        #               (suggest_dict[x][1], -suggest_dict[x][0]))

        # output option 2
        # return list of suggestions with (correction,
        #                                  (frequency in corpus, edit distance)):
        as_list = suggest_dict.items()
        # outlist = sorted(as_list, key=lambda (term, (freq, dist)): (dist, -freq))
        outlist = sorted(as_list, key=lambda x: (x[1][1], -x[1][0]))

        if self.verbose == 0:
            return outlist[0]
        else:
            return outlist

        '''
        Option 1:
        ['file', 'five', 'fire', 'fine', ...]

        Option 2:
        [('file', (5, 0)),
         ('five', (67, 1)),
         ('fire', (54, 1)),
         ('fine', (17, 1))...]  
        '''

    def best_word(self, s, silent=False):
        try:
            return self.get_suggestions(s, silent)[0]
        except:
            return None


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field, start_time=time()):
        self.field = field
        self.start_time = start_time

    def fit(self, x, y=None):
        return self

    def transform(self, dataframe):
        print(f'[{time()-self.start_time}] select {self.field}')
        dt = dataframe[self.field].dtype
        if is_categorical_dtype(dt):
            return dataframe[self.field].cat.codes[:, None]
        elif is_numeric_dtype(dt):
            return dataframe[self.field][:, None]
        else:
            return dataframe[self.field]


class DropColumnsByDf(BaseEstimator, TransformerMixin):
    def __init__(self, min_df=1, max_df=1.0):
        self.min_df = min_df
        self.max_df = max_df

    def fit(self, X, y=None):
        m = X.tocsc()
        self.nnz_cols = ((m != 0).sum(axis=0) >= self.min_df).A1
        if self.max_df < 1.0:
            max_df = m.shape[0] * self.max_df
            self.nnz_cols = self.nnz_cols & ((m != 0).sum(axis=0) <= max_df).A1
        return self

    def transform(self, X, y=None):
        m = X.tocsc()
        return m[:, self.nnz_cols]


def get_rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(np.expm1(y_true), np.expm1(y_pred)))


def split_cat(text):
    try:
        cats = text.split("/")
        return cats[0], cats[1], cats[2], cats[0] + '/' + cats[1]
    except:
        print("no category")
        return 'other', 'other', 'other', 'other/other'


def brands_filling(dataset):
    vc = dataset['brand_name'].value_counts()
    brands = vc[vc > 0].index
    brand_word = r"[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+"

    many_w_brands = brands[brands.str.contains(' ')]
    one_w_brands = brands[~brands.str.contains(' ')]

    ss2 = SymSpell(max_edit_distance=0)
    ss2.create_dictionary_from_arr(many_w_brands, token_pattern=r'.+')

    ss1 = SymSpell(max_edit_distance=0)
    ss1.create_dictionary_from_arr(one_w_brands, token_pattern=r'.+')

    two_words_re = re.compile(r"(?=(\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+))")

    def find_in_str_ss2(row):
        for doc_word in two_words_re.finditer(row):
            print(doc_word)
            suggestion = ss2.best_word(doc_word.group(1), silent=True)
            if suggestion is not None:
                return doc_word.group(1)
        return ''

    def find_in_list_ss1(list):
        for doc_word in list:
            suggestion = ss1.best_word(doc_word, silent=True)
            if suggestion is not None:
                return doc_word
        return ''

    def find_in_list_ss2(list):
        for doc_word in list:
            suggestion = ss2.best_word(doc_word, silent=True)
            if suggestion is not None:
                return doc_word
        return ''

    print(f"Before empty brand_name: {len(dataset[dataset['brand_name'] == ''].index)}")

    n_name = dataset[dataset['brand_name'] == '']['name'].str.findall(
        pat=r"^[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+")
    dataset.loc[dataset['brand_name'] == '', 'brand_name'] = [find_in_list_ss2(row) for row in n_name]

    n_desc = dataset[dataset['brand_name'] == '']['item_description'].str.findall(
        pat=r"^[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+")
    dataset.loc[dataset['brand_name'] == '', 'brand_name'] = [find_in_list_ss2(row) for row in n_desc]

    n_name = dataset[dataset['brand_name'] == '']['name'].str.findall(pat=brand_word)
    dataset.loc[dataset['brand_name'] == '', 'brand_name'] = [find_in_list_ss1(row) for row in n_name]

    desc_lower = dataset[dataset['brand_name'] == '']['item_description'].str.findall(pat=brand_word)
    dataset.loc[dataset['brand_name'] == '', 'brand_name'] = [find_in_list_ss1(row) for row in desc_lower]

    print(f"After empty brand_name: {len(dataset[dataset['brand_name'] == ''].index)}")

    del ss1, ss2
    gc.collect()


def preprocess_regex(dataset, start_time=time()):
    karats_regex = r'(\d)([\s-]?)(karat|karats|carat|carats|kt)([^\w])'
    karats_repl = r'\1k\4'

    unit_regex = r'(\d+)[\s-]([a-z]{2})(\s)'
    unit_repl = r'\1\2\3'

    dataset['name'] = dataset['name'].str.replace(karats_regex, karats_repl)
    dataset['item_description'] = dataset['item_description'].str.replace(karats_regex, karats_repl)
    print(f'[{time() - start_time}] Karats normalized.')

    dataset['name'] = dataset['name'].str.replace(unit_regex, unit_repl)
    dataset['item_description'] = dataset['item_description'].str.replace(unit_regex, unit_repl)
    print(f'[{time() - start_time}] Units glued.')


def preprocess_pandas(train, test, start_time=time()):
    train = train[train.price > 0.0].reset_index(drop=True)
    print('Train shape without zero price: ', train.shape)

    nrow_train = train.shape[0]
    y_train = np.log1p(train["price"])
    merge: pd.DataFrame = pd.concat([train, test])

    del train
    del test
    gc.collect()

    merge['has_category'] = (merge['category_name'].notnull()).astype('category')
    print(f'[{time() - start_time}] Has_category filled.')

    merge['category_name'] = merge['category_name'] \
        .fillna('other/other/other') \
        .str.lower() \
        .astype(str)
    merge['general_cat'], merge['subcat_1'], merge['subcat_2'], merge['gen_subcat1'] = \
        zip(*merge['category_name'].apply(lambda x: split_cat(x)))
    print(f'[{time() - start_time}] Split categories completed.')

    merge['has_brand'] = (merge['brand_name'].notnull()).astype('category')
    print(f'[{time() - start_time}] Has_brand filled.')

    merge['gencat_cond'] = merge['general_cat'].map(str) + '_' + merge['item_condition_id'].astype(str)
    merge['subcat_1_cond'] = merge['subcat_1'].map(str) + '_' + merge['item_condition_id'].astype(str)
    merge['subcat_2_cond'] = merge['subcat_2'].map(str) + '_' + merge['item_condition_id'].astype(str)
    print(f'[{time() - start_time}] Categories and item_condition_id concancenated.')

    merge['name'] = merge['name'] \
        .fillna('') \
        .str.lower() \
        .astype(str)
    merge['brand_name'] = merge['brand_name'] \
        .fillna('') \
        .str.lower() \
        .astype(str)
    merge['item_description'] = merge['item_description'] \
        .fillna('') \
        .str.lower() \
        .replace(to_replace='No description yet', value='')
    print(f'[{time() - start_time}] Missing filled.')

    preprocess_regex(merge, start_time)

    brands_filling(merge)
    print(f'[{time() - start_time}] Brand name filled.')

    merge['name'] = merge['name'] + ' ' + merge['brand_name']
    print(f'[{time() - start_time}] Name concancenated.')

    merge['item_description'] = merge['item_description'] \
                                + ' ' + merge['name'] \
                                + ' ' + merge['subcat_1'] \
                                + ' ' + merge['subcat_2'] \
                                + ' ' + merge['general_cat'] \
                                + ' ' + merge['brand_name']
    print(f'[{time() - start_time}] Item description concatenated.')

    merge.drop(['price', 'test_id', 'train_id'], axis=1, inplace=True)

    return merge, y_train, nrow_train


def intersect_drop_columns(train: csr_matrix, valid: csr_matrix, min_df=0):
    t = train.tocsc()
    v = valid.tocsc()
    nnz_train = ((t != 0).sum(axis=0) >= min_df).A1
    nnz_valid = ((v != 0).sum(axis=0) >= min_df).A1
    nnz_cols = nnz_train & nnz_valid
    res = t[:, nnz_cols], v[:, nnz_cols]
    return res


if __name__ == '__main__':
    mp.set_start_method('forkserver', True)

    start_time = time()

    train = pd.read_table(os.path.join(INPUT_PATH, 'train.tsv'),
                          engine='c',
                          dtype={'item_condition_id': 'category',
                                 'shipping': 'category'}
                          )
    test = pd.read_table(os.path.join(INPUT_PATH, 'test.tsv'),
                         engine='c',
                         dtype={'item_condition_id': 'category',
                                'shipping': 'category'}
                         )
    print(f'[{time() - start_time}] Finished to load data')
    print('Train shape: ', train.shape)
    print('Test shape: ', test.shape)

    submission: pd.DataFrame = test[['test_id']]

    merge, y_train, nrow_train = preprocess_pandas(train, test, start_time)

    meta_params = {'name_ngram': (1, 2),
                   'name_max_f': 75000,
                   'name_min_df': 10,

                   'category_ngram': (2, 3),
                   'category_token': '.+',
                   'category_min_df': 10,

                   'brand_min_df': 10,

                   'desc_ngram': (1, 3),
                   'desc_max_f': 150000,
                   'desc_max_df': 0.5,
                   'desc_min_df': 10}

    stopwords = frozenset(['the', 'a', 'an', 'is', 'it', 'this', ])
    # 'i', 'so', 'its', 'am', 'are'])

    vectorizer = FeatureUnion([
        ('name', Pipeline([
            ('select', ItemSelector('name', start_time=start_time)),
            ('transform', HashingVectorizer(
                ngram_range=(1, 2),
                n_features=2 ** 27,
                norm='l2',
                lowercase=False,
                stop_words=stopwords
            )),
            ('drop_cols', DropColumnsByDf(min_df=2))
        ])),
        ('category_name', Pipeline([
            ('select', ItemSelector('category_name', start_time=start_time)),
            ('transform', HashingVectorizer(
                ngram_range=(1, 1),
                token_pattern='.+',
                tokenizer=split_cat,
                n_features=2 ** 27,
                norm='l2',
                lowercase=False
            )),
            ('drop_cols', DropColumnsByDf(min_df=2))
        ])),
        ('brand_name', Pipeline([
            ('select', ItemSelector('brand_name', start_time=start_time)),
            ('transform', CountVectorizer(
                token_pattern='.+',
                min_df=2,
                lowercase=False
            )),
        ])),
        ('gencat_cond', Pipeline([
            ('select', ItemSelector('gencat_cond', start_time=start_time)),
            ('transform', CountVectorizer(
                token_pattern='.+',
                min_df=2,
                lowercase=False
            )),
        ])),
        ('subcat_1_cond', Pipeline([
            ('select', ItemSelector('subcat_1_cond', start_time=start_time)),
            ('transform', CountVectorizer(
                token_pattern='.+',
                min_df=2,
                lowercase=False
            )),
        ])),
        ('subcat_2_cond', Pipeline([
            ('select', ItemSelector('subcat_2_cond', start_time=start_time)),
            ('transform', CountVectorizer(
                token_pattern='.+',
                min_df=2,
                lowercase=False
            )),
        ])),
        ('has_brand', Pipeline([
            ('select', ItemSelector('has_brand', start_time=start_time)),
            ('ohe', OneHotEncoder())
        ])),
        ('shipping', Pipeline([
            ('select', ItemSelector('shipping', start_time=start_time)),
            ('ohe', OneHotEncoder())
        ])),
        ('item_condition_id', Pipeline([
            ('select', ItemSelector('item_condition_id', start_time=start_time)),
            ('ohe', OneHotEncoder())
        ])),
        ('item_description', Pipeline([
            ('select', ItemSelector('item_description', start_time=start_time)),
            ('hash', HashingVectorizer(
                ngram_range=(1, 3),
                n_features=2 ** 27,
                dtype=np.float32,
                norm='l2',
                lowercase=False,
                stop_words=stopwords
            )),
            ('drop_cols', DropColumnsByDf(min_df=2)),
        ]))
    ], n_jobs=1)

    sparse_merge = vectorizer.fit_transform(merge)
    print(f'[{time() - start_time}] Merge vectorized')
    print(sparse_merge.shape)

    tfidf_transformer = TfidfTransformer()

    X = tfidf_transformer.fit_transform(sparse_merge)
    print(f'[{time() - start_time}] TF/IDF completed')

    X_train = X[:nrow_train]
    print(X_train.shape)

    X_test = X[nrow_train:]
    del merge
    del sparse_merge
    del vectorizer
    del tfidf_transformer
    gc.collect()

    X_train, X_test = intersect_drop_columns(X_train, X_test, min_df=1)
    print(f'[{time() - start_time}] Drop only in train or test cols: {X_train.shape[1]}')
    gc.collect()

    ridge = Ridge(solver='auto', fit_intercept=True, alpha=0.4, max_iter=200, normalize=False, tol=0.01)
    ridge.fit(X_train, y_train)
    print(f'[{time() - start_time}] Train Ridge completed. Iterations: {ridge.n_iter_}')

    predsR = ridge.predict(X_test)
    print(f'[{time() - start_time}] Predict Ridge completed.')

    ######## FM_FTRL
    import time
    start_time = time.time()
    import numpy as np
    import pandas as pd
    
    from sklearn.preprocessing import LabelEncoder, LabelBinarizer
    from sklearn.cross_validation import train_test_split
    from scipy.sparse import csr_matrix, lil_matrix, coo_matrix, hstack
    from keras.preprocessing.text import text_to_word_sequence
    from keras.preprocessing.sequence import pad_sequences
    from keras.layers import Input, Dropout, Dense, BatchNormalization, Add, \
         concatenate, GRU, Embedding, Flatten, Conv1D, MaxPooling1D, Maximum, ELU
    from keras.optimizers import Adam
    from keras.models import Model
    from keras import backend as K
    from nltk.corpus import stopwords
    from nltk.stem.snowball import SnowballStemmer
    import tensorflow as tf
    import string, re
    import warnings
    from collections import OrderedDict
    from six.moves import range
    from six.moves import zip
    
    from wordbatch.models import FM_FTRL
    import lightgbm as lgb
    
    SUBMIT = True 
    
    stemmer = SnowballStemmer("english")
    # set seed
    np.random.seed(123)
    
    # I believe that these stopwords may have impact on the price, so they are not removed
    stopwords_exceptions = {'off', 'no', 'not', 'don', 'once', 'only'}
    actual_stopwords = set(stopwords.words('english')) - stopwords_exceptions
    
    def rmse(y_true, y_pred):
      return tf.sqrt(tf.reduce_mean(tf.squared_difference(y_true, y_pred)))
    
    def print_time(text):
      time_diff = int((time.time() - start_time))
      minutes, seconds = (time_diff//60, time_diff%60)
      print("[%d:%d] %s" %(minutes, seconds, text))
    
    pat_1st_isnumber = re.compile('^[0-9]')
    pat_date = re.compile('^[1-2][0-9]{2}$')
    pat_twodigit = re.compile('^[0-9]{2}$')
    pat_threedigit = re.compile('^[0-9]{3}$')
    pat_number = re.compile('^[0-9]+(point)?[0-9]*$')
    pat0 = re.compile('^zero$')
    pat1 = re.compile('^one$')
    pat2 = re.compile('^two$')
    pat3 = re.compile('^three$')
    pat4 = re.compile('^four$')
    pat5 = re.compile('^five$')
    pat6 = re.compile('^six$')
    pat7 = re.compile('^seven$')
    pat8 = re.compile('^eight$')
    pat9 = re.compile('^nine$')
    
    def trans(token):
      return token
      if token is None:
        return token
      #if pat_1st_isnumber.match(token[0]):
      #  if pat_date.match(token):
      #    return 'specialdate'
      #  if pat_threedigit.match(token):
      #    return 'specialthreedigitnumber'
      #  if pat_twodigit.match(token):
      #    return 'specialwodigitnumber'
      #  return token
      if token[0] in ['z', 'o', 't', 'f', 's', 'n', 'e']:
        if pat0.match(token):
          return '0'
        if pat1.match(token):
          return '1'
        if pat2.match(token):
          return '2'
        if pat3.match(token):
          return '3'
        if pat4.match(token):
          return '4'
        if pat5.match(token):
          return '5'
        if pat6.match(token):
          return '6'
        if pat7.match(token):
          return '7'
        if pat8.match(token):
          return '8'
        if pat9.match(token):
          return '9'
      return token
    
    pat_float = re.compile(r'([0-9]+)[\.\/]([0-9]+)')
    pat_inches = re.compile(r'([0-9]+)[\s]*(\"\"|\'\'|\")[\.\!\s\)]')
    pat_dis = re.compile(r'([0-9]+)(\"\"|\'\'|\"|m|cm|inches|inch|in|mm|pc)[\.\!\s\)]')
    pat_mass = re.compile(r'([0-9]+)(g|kg|lbs)[\.\!\s\)]')
    pat_vol = re.compile(r'([0-9]+)(ml|liters|milliliters|oz|gal|ct)[\)\.\!\s]')
    pat_per = re.compile(r'([0-9]+)[\s]*(%)[\.\!\s\)]')
    pat_sep_num = re.compile(r'([0-9])([^0-9\s\.])')
    pat_splitter = re.compile(r'\s*[^a-z0-9\s]+\s*')
    
    def my_text_to_word_sequence(text):
        text = text.lower() + '.'
        text = pat_inches.sub(r'\1 inches ', text)
        #text = pat_dis.sub(r'\1 specialdistance ', text)
        #text = pat_mass.sub(r'\1 specialmass ', text)
        #text = pat_vol.sub(r'\1 specialvolume ', text)
        text = pat_sep_num.sub(r'\1 \2', text)
        text = pat_per.sub(r'\1 percent ', text)
        text = pat_float.sub(r'\1point\2', text)
        #text = re.sub(r'[-+]?([0-9]+)[\.\/]([0-9]+)', r'\1point\2', text)
        #text = re.sub(r'([0-9]+)(\"\"|\'\'|\")[\.\!\s\)]', r'\1 inches ', text)
        #text = re.sub(r'([0-9]+)(m|cm|inches|inch|in|mm|pc)[\.\!\s\)]', r'\1 \2 ', text)
        #text = re.sub(r'([0-9]+)(g|kg|lbs)[\.\!\s\)]', r'\1 \2 ', text)
        #text = re.sub(r'([0-9]+)(ml|liters|milliliters|oz|gal|ct)[\)\.\!\s]', r'\1 \2 ', text)
        #text = re.sub(r'([0-9]+)[\s]*(%)[\.\!\s\)]', r'\1 percent ', text)
    
        #text = re.sub(r'\s[0-9][0-9\.\/])*[\s]*(""|m|cm|inches|inch|in|mm)[\.\!\s]', ' specialdistance ', text)
        #text = re.sub(r'\s[0-9][0-9\.\/]*[\s]*(g|kg|lbs)[\.\!\s]', ' specialmass ', text)
        #text = re.sub(r'\s[0-9][0-9\.\/]*[\s]*(%)[\.\!\s]', ' specialpercent ', text)
        #text = re.sub(r'\s[0-9][0-9\.\/]*[\s]*(ml|liters|milliliters|oz|gal|ct)[\.\!\s]', ' specialvolume ', text)
        text = pat_splitter.sub(' ', text)
        seq = text.split(' ')
        return [i for i in seq[:-1] if i and i not in actual_stopwords]
    
    class StemmerTokenizer(object):
        def __init__(self, num_words=None,
                     lower=True,
                     **kwargs):
            # Legacy support
            if 'nb_words' in kwargs:
                warnings.warn('The `nb_words` argument in `Tokenizer` '
                              'has been renamed `num_words`.')
                num_words = kwargs.pop('nb_words')
            if kwargs:
                raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))
    
            self.word_counts = OrderedDict()
            self.lower = lower
            self.num_words = num_words
            self.word_stem = {}
            self.bigram_counts = {}
            self.trigram_counts = {}
    
        def fit_on_texts(self, sequences, enable_bigrams=False):
            for seq in sequences:
                for i, w_original in enumerate(seq):
                    if w_original in self.word_stem:
                      w = self.word_stem[w_original]
                    else:
                      w = stemmer.stem(w_original)
                      self.word_stem[w_original] = w
                    if w in self.word_counts:
                        self.word_counts[w] += 1
                    else:
                        self.word_counts[w] = 1
                    if i==0 or not enable_bigrams:
                      continue
                    concat = seq[i-1] + '_' + w
                    if concat in self.bigram_counts:
                        self.bigram_counts[concat] += 1
                    else:
                        self.bigram_counts[concat] = 1
                    if i<2:
                      continue
                    concat = seq[i-2] + '_' + seq[i-1] + '_' + w
                    #if concat in self.trigram_counts:
                    #    self.trigram_counts[concat] += 1
                    #else:
                    #    self.trigram_counts[concat] = 1
    
            wcounts = list(self.word_counts.items())
            wcounts.sort(key=lambda x: x[1], reverse=True)
            sorted_voc = [wc[0] for wc in wcounts]
            # note that index 0 is reserved, never assigned to an existing word
            self.word_index = dict(list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))
            if enable_bigrams:
              # order bigram_counts by count
              bicounts = list(self.bigram_counts.items())
              bicounts.sort(key=lambda x: x[1], reverse=True)
              print(bicounts[:40])
              self.bigram_counts = dict(bicounts[:700])
              self.bigram_index = {k:i for i, k in enumerate(self.bigram_counts.keys())}
    
              #tricounts = list(self.trigram_counts.items())
              #tricounts.sort(key=lambda x: x[1], reverse=True)
              #print(tricounts[:40])
              #self.trigram_counts = dict(tricounts[:350])
              #self.trigram_index = {k:i for i, k in enumerate(self.trigram_counts.keys())}
    
        def get_word_index(self, w):
          word_idx = self.word_index.get(w)
          if word_idx is None:
            if w in self.word_stem:
              w = self.word_stem[w] 
            else:
              w = stemmer.stem(w)
            if pat_date.match(w):
              return (self.offset-2, w)
            if pat_number.match(w):
              return (self.offset-1, w)
            w = trans(w)
            word_idx = self.word_index.get(w)
          return (word_idx, w)
    
        def texts_to_sequences(self, sequences):
            self.offset = len(self.word_index) + 2
            #self.offset2 = self.offset + len(self.bigram_index)
            num_words = self.offset + len(self.bigram_index)
            self.x_row, self.x_col = [], []
            res = []
            for vect in self.texts_to_sequences_generator(sequences):
                res.append(vect)
    
            x_data = np.ones(len(self.x_row), dtype=np.float64)
            self.x = coo_matrix((x_data, (self.x_row, self.x_col)), shape=(len(sequences), num_words))
            del self.x_row
            del self.x_col
    
            self.x = self.x.tocsr()
            return res
    
        def texts_to_sequences_generator(self, sequences):
            for seq_idx, seq in enumerate(sequences):
              vect = []
              idx_to_add = set()
              prev2_word = None
              prev2_idx = None
              prev_word = None
              prev_idx = None
              for w in seq:
                word_idx, w = self.get_word_index(w) 
                # look for trigram
                #if prev2_idx and prev_idx and word_idx:
                #  concat = prev2_word + '_' + prev_word + '_' + w
                #  if concat in self.trigram_index:
                #    trigram_idx = self.offset+len(self.bigram_index) + self.trigram_index[concat]
                #    idx_to_add.add(trigram_idx)
    
                # look for bigram
                if prev_idx and word_idx:
                  concat = prev_word + '_' + w
                  if concat in self.bigram_index:
                    bigram_idx = self.offset+self.bigram_index[concat]
                    idx_to_add.add(bigram_idx)
    
                prev2_word, prev2_idx = (prev_word, prev_idx)
                # look for unigram
                if word_idx is not None:
                  vect.append(word_idx)
                  idx_to_add.add(word_idx)
                  prev_word, prev_idx = (w, word_idx)
                else:
                  prev_word, prev_idx = (None, None)
    
              self.x_row.extend([seq_idx for idx in idx_to_add])
              self.x_col.extend([idx for idx in idx_to_add])
              yield vect
    
    def new_rnn_model(lr=0.001, decay=0.0):
      # Inputs
      name = Input(shape=[X_train['name'].shape[1]], name="name")
      item_desc = Input(shape=[X_train['item_desc'].shape[1]], name="item_desc")
      brand_name = Input(shape=[1], name="brand_name")
      category = Input(shape=[1], name="category")
      #has_pic = Input(shape=[1], name="has_pic")
      #has_brand_on_name  = Input(shape=[1], name="has_brand_on_name")
      #has_brand_on_descp = Input(shape=[1], name="has_brand_on_descp")
    
      item_condition = Input(shape=[1], name="item_condition")
      shipping = Input(shape=[1], name='shipping')
      desc_len = Input(shape=[1], name="desc_len")
      name_len = Input(shape=[1], name="name_len")
      subcat_0 = Input(shape=[1], name="subcat_0")
      subcat_1 = Input(shape=[1], name="subcat_1")
      subcat_2 = Input(shape=[1], name="subcat_2")
    
      # Embeddings layers (adjust outputs to help model)
      # vocab size is 1.5x bigger for item_desc
      emb_name = Embedding(MAX_TEXT_NAME, 20)(name)
      emb_item_desc = Embedding(MAX_TEXT_DESCP, 30)(item_desc)
    
      emb_brand_name = Embedding(MAX_BRAND, 30)(brand_name)
      emb_category = Embedding(MAX_CATEGORY, 10)(category)
      emb_subcat_0 = Embedding(MAX_SUBCAT_0, 5)(subcat_0)
      emb_subcat_1 = Embedding(MAX_SUBCAT_1, 10)(subcat_1)
      emb_subcat_2 = Embedding(MAX_SUBCAT_2, 15)(subcat_2)
      emb_item_condition = Embedding(MAX_CONDITION, 5)(item_condition)
      emb_desc_len = Embedding(MAX_DESC_LEN, 5)(desc_len)
      emb_name_len = Embedding(MAX_NAME_LEN, 5)(name_len)
    
      #l_cov1 = ELU()(Conv1D(16, 3)(emb_item_desc))
      #l_pool1 = MaxPooling1D(2)(l_cov1)
      #l_cov2 = ELU()(Conv1D(32, 3)(l_pool1))
      #l_pool2 = MaxPooling1D(2)(l_cov2)
      #l_cov3 = ELU()(Conv1D(64, 3)(l_pool2))
      #l_pool3 = MaxPooling1D(4)(l_cov3)
      #l_flat1 = Flatten()(l_pool3)
    
      #l_cov12 = ELU()(Conv1D(32, 3)(emb_name))
      #l_pool12 = MaxPooling1D(2)(l_cov12)
      #l_cov22 = ELU()(Conv1D(64, 3)(l_pool12))
      #l_flat2 = Flatten()(l_cov22)
    
      rnn_layer1 = GRU(10) (emb_item_desc) # 10
      rnn_layer2 = GRU(8) (emb_name) # 8
    
      # main layers
      main_l = concatenate([
                 Flatten() (emb_brand_name),
                 Flatten() (emb_category),
                 Flatten() (emb_item_condition),
                 Flatten() (emb_desc_len),
                 Flatten() (emb_name_len),
                 Flatten() (emb_subcat_0),
                 Flatten() (emb_subcat_1),
                 Flatten() (emb_subcat_2),
                 rnn_layer1,
                 rnn_layer2,
                 #l_flat1,
                 #l_flat2,
                 shipping,
               ])
      layer_width = 32
      main_l = Dropout(0.1)(ELU()(Dense(512) (BatchNormalization()(main_l))))
      l1 = Dropout(0.1)(ELU()(Dense(layer_width) (BatchNormalization()(main_l))))
      l2 = Dropout(0.1)(ELU()(Dense(layer_width) (BatchNormalization()(l1))))
      ls = Add()([l1, l2])
      l3 = Dropout(0.1)(ELU()(Dense(layer_width) (BatchNormalization()(ls))))
      ls = Add()([ls, l3])
      l4 = Dropout(0.1)(ELU()(Dense(layer_width) (BatchNormalization()(ls))))
      ls = Add()([ls, l4])
      l5 = Dropout(0.1)(ELU()(Dense(layer_width) (BatchNormalization()(ls))))
      ls = Add()([ls, l5])
      l6 = Dropout(0.1)(ELU()(Dense(layer_width) (BatchNormalization()(ls))))
    
      #main_l = Dropout(0.1)(Dense(300, activation='relu') (main_l))
      #main_l = Dropout(0.1)(Dense(100, activation='relu') (main_l))
      #main_l = Dropout(0.1)(Dense(50, activation='relu') (main_l))
      #main_l = Dropout(0.1)(Dense(20, activation='relu') (main_l))
    
      # the output layer.
      output = Dense(1, activation="linear") (l6)
      
      model = Model([name,
                     item_desc, 
                     brand_name , 
                     item_condition, 
                     category, 
                     #has_pic, 
                     #has_brand_on_name, 
                     #has_brand_on_descp,
                     shipping, 
                     desc_len, 
                     name_len, 
                     subcat_0, 
                     subcat_1, 
                     subcat_2],
                     output
                   )
    
      optimizer = Adam(lr=lr, decay=decay)
      model.compile(loss=rmse, optimizer=optimizer)
    
      return model
    
    def get_rnn_data(dataset):
      return {
            'name': pad_sequences(
                      dataset.seq_name,
                      maxlen=MAX_NAME_SEQ,
                      truncating='post'
                    ),
            'item_desc': pad_sequences(
                           dataset.seq_descp,
                           maxlen=MAX_ITEM_DESC_SEQ,
                           truncating='post'
                         ),
            'brand_name':     dataset.brand_name.values,
            'category':       dataset.category.values,
            'subcat_0':       dataset.subcat_0.values,
            'subcat_1':       dataset.subcat_1.values,
            'subcat_2':       dataset.subcat_2.values,
            'item_condition': dataset.item_condition_id.values,
            'shipping':       dataset[["shipping"]].values,
            'name_len':       dataset[["name_len"]].values,
            'desc_len':       dataset[["desc_len"]].values,
            #'has_pic' : np.array(dataset[["has_pic"]]),
            #'has_brand_on_name'  : np.array(dataset[["has_brand_on_name"]]),
            #'has_brand_on_descp' : np.array(dataset[["has_brand_on_descp"]]),
      }
    
    def rmsle(Y, Y_pred):
        assert Y.shape == Y_pred.shape
        return np.sqrt(np.mean(np.square(Y_pred - Y )))
    
    def split_cat(text):
        try: return text.split("/")
        except: return ("No Label", "No Label", "No Label")
    
    # this one is faster (20 secs)
    def brandfinder(name):
        for x in name.split(' '):
            if x in all_brands:
                return x
        return 'missing' 
    
    # brands_per_cat
    # fake products are taken as originals (e.g. Victoria's Secret)
    # channel (in electronics) is taken as Channel
    # missing should be skipped 
    def slow_brandfinder(params):
        name = params[0]
        cat = params[1]
        try: 
            brands_this_cat = brands_per_cat0[cat]
            namesplit = name.lower().split(' ')
            for x in namesplit:
                if x in brands_this_cat:
                    return x
            return 'missing'
        except:
            return 'missing'
        #print(brands_this_cat)
    
    ### PHASE 1: prepare inputs
    train_df = pd.read_table('../input/train.tsv', engine='c')
    test_df = pd.read_table('../input/test.tsv', engine='c')
    test_ids = test_df.test_id
    #train_df = train_df[:10000]
    print(train_df.shape, test_df.shape)
    
    train_df = train_df.drop(train_df[(train_df.price < 3.0)].index)
    
    # Scale target variable to log.
    train_df["target"] = np.log1p(train_df.price)
    
    # Split training examples into train/dev examples.
    train_size = 0.999 if SUBMIT==True else 0.9
    train_df, dev_df = train_test_split(train_df, random_state=123, train_size=train_size)
    
    # Calculate number of train/dev/test examples.
    n_trains = train_df.shape[0]
    n_devs = dev_df.shape[0]
    n_tests = test_df.shape[0]
    print("Training on", n_trains, "examples")
    print("Validating on", n_devs, "examples")
    print("Testing on", n_tests, "examples")
    
    # Concatenate train - dev - test data for easy to handle
    full_df = pd.concat([train_df, dev_df, test_df])
    del train_df
    del dev_df
    del test_df
    full_df['name'] = full_df['name'] \
        .fillna('') \
        .str.lower() \
        .astype(str)
    full_df['brand_name'] = full_df['brand_name'] \
        .fillna('') \
        .str.lower() \
        .astype(str)
    full_df['item_description'] = full_df['item_description'] \
        .fillna('') \
        .str.lower() \
        .replace(to_replace='No description yet', value='') \
        .astype(str)
    
    brands_filling(full_df)

    #full_set = pd.concat([train_df,test_df])
    all_brands = set(full_df['brand_name'].astype(str).values)
    #full_df.brand_name.fillna(value='missing', inplace=True)
    
    # split category name into 3 parts
    full_df['subcat_0'], full_df['subcat_1'], full_df['subcat_2'] = \
      zip(*full_df['category_name'].apply(lambda x: split_cat(x)))
    
    #brands_per_cat0 = full_df.groupby('subcat_0')['brand_name'].apply(lambda x: set(x.str.lower())).to_dict()
    
    #premissing = len(full_df.loc[full_df['brand_name'] == 'missing'])
    #full_df.loc[full_df.brand_name == 'missing', 'brand_name'] = \
    #  full_df[full_df.brand_name == 'missing']['name'].apply(brandfinder)
    #  #full_df[full_df.brand_name == 'missing'][['name', 'subcat_0']].apply(slow_brandfinder, axis=1)
    #
    #found = premissing - len(full_df.loc[full_df['brand_name'] == 'missing'])
    #print("Found %d missing brands." %(found))
    
    #full_df.loc[full_df.brand_name != 'missing', 'name'] = \
    #  full_df[full_df.brand_name != 'missing'].name.str.cat(full_df[full_df.brand_name != 'missing'].brand_name, sep=' ').astype(str)
    
    print_time('Filling missing data')
    full_df.category_name.fillna(value="missing", inplace=True)
    full_df.item_description.fillna(value="", inplace=True)
    full_df.item_description.replace('No description yet', "", inplace=True)
    #full_df['has_brand_on_name']  = full_df.apply(lambda row: row['brand_name'] in row['name'], axis=1).astype(int)
    #full_df['has_brand_on_descp'] = full_df.apply(lambda row: row['brand_name'] in row['item_description'], axis=1).astype(int)
    #full_df['has_pic'] = full_df.item_description.str.contains('photo|image|picture|pics', case=False).astype(int)
    
    print_time('Processing categorical data...')
    
    #le = LabelEncoder()
    #
    #full_df['category'] = le.fit_transform(full_df.category_name)
    #full_df.brand_name = le.fit_transform(full_df.brand_name)
    #full_df.subcat_0 = le.fit_transform(full_df.subcat_0)
    #full_df.subcat_1 = le.fit_transform(full_df.subcat_1)
    #full_df.subcat_2 = le.fit_transform(full_df.subcat_2)
    #
    #del le
    
    print_time('Transforming text data to sequences...')
    full_df['descp_word_seq'] = full_df.item_description.apply(my_text_to_word_sequence)
    full_df['name_word_seq'] = full_df.name.apply(my_text_to_word_sequence)
    print_time('Finished word to sequence of words (name and descp).')
    
    min_freq = 6
    
    # validation set
    tokenizer = StemmerTokenizer()
    if SUBMIT:
      tokenizer.fit_on_texts(full_df[n_trains+n_devs:].descp_word_seq)
    else:
      tokenizer.fit_on_texts(full_df[n_trains:n_trains+n_devs].descp_word_seq)
    test_word_set_descp = {trans(k) for k in set(tokenizer.word_index.keys())}
    print_time('Finished fitting on validation/test set (description).')
    del tokenizer
    tokenizer = StemmerTokenizer()
    if SUBMIT:
      tokenizer.fit_on_texts(full_df[n_trains+n_devs:].name_word_seq)
    else:
      tokenizer.fit_on_texts(full_df[n_trains:n_trains+n_devs].name_word_seq)
    test_word_set_name = {trans(k) for k in set(tokenizer.word_index.keys())}
    print_time('Finished fitting on validation/test set (name).')
    del tokenizer
    
    # training set
    # name 
    tokenizer_name = StemmerTokenizer()
    tokenizer_name.fit_on_texts(full_df[:n_trains].name_word_seq, enable_bigrams=True)
    word_set = set(tokenizer_name.word_index.keys())
    print("Original number of tokens: %d" % (len(word_set)))
    
    # finding patterns
    def find_patterns(word_set, tokenizer):
      new_set = set()
      new_count = {}
      for k in word_set:
        new_word = trans(k)
        new_set.add(new_word)
        if new_word in new_count:
          new_count[new_word] += tokenizer.word_counts[k]
        else:
          new_count[new_word] = tokenizer.word_counts[k]
      return (new_set, new_count)
    
    word_set, new_count = find_patterns(word_set, tokenizer_name)
      
    #remove words that appear only on train set or only on test set
    word_set = {k for k in word_set if k in test_word_set_name}
    print("number of tokens that are in train and test sets: %d" % (len(word_set)))
    #remove words that appear less than min_freq times. Number to tune
    frequent_word_set = {k for k in word_set if new_count[k]>min_freq}
    word_set = {k for k in word_set if k in frequent_word_set}
    print("number of tokens that appear more than X times: %d" % (len(word_set)))
    # reorder word_index
    tokenizer_name.word_index = {k:i for i, k in enumerate(word_set)}
    
    # descp
    tokenizer_descp = StemmerTokenizer()
    tokenizer_descp.fit_on_texts(full_df[:n_trains].descp_word_seq, enable_bigrams=True)
    word_set = set(tokenizer_descp.word_index.keys())
    print("Original number of tokens: %d" % (len(word_set)))
    
    # finding patterns
    word_set, new_count = find_patterns(word_set, tokenizer_descp)
    
    #remove words that appear only on train set or only on test set
    word_set = {k for k in word_set if k in test_word_set_descp}
    print("number of tokens that are in train and test sets: %d" % (len(word_set)))
    
    #remove words that appear less than min_freq times. Number to tune
    frequent_word_set = {k for k in word_set if new_count[k]>min_freq}
    word_set = {k for k in word_set if k in frequent_word_set}
    print("number of tokens that appear more than X times: %d" % (len(word_set)))
    
    
    # reorder word_index
    tokenizer_descp.word_index = {k:i for i,(k) in enumerate(word_set)}
    
    # Transforming
    print_time('Transforming full dataset...')
    full_df['seq_descp'] = tokenizer_descp.texts_to_sequences(full_df.descp_word_seq)
    MAX_TEXT_DESCP = max(tokenizer_descp.word_index.values()) + 2 + 1
    print_time('Finished transforming full dataset (description).')
    
    full_df['seq_name'] = tokenizer_name.texts_to_sequences(full_df.name_word_seq)
    MAX_TEXT_NAME = max(tokenizer_name.word_index.values()) + 2 + 1
    print_time('Finished transforming full dataset (name).')
    
    del full_df['name_word_seq']
    del full_df['descp_word_seq']
    
    print_time('Sequences created.')
    
    
    print_time('Counting words...')
    full_df['desc_len'] = full_df['seq_descp'].apply(len)
    full_df['name_len'] = full_df['seq_name'].apply(len)
    print_time('Done.')
    
    #MAX_NAME_SEQ = 8
    #MAX_ITEM_DESC_SEQ = 30
    #MAX_CATEGORY = np.max(full_df.category.max()) + 1
    #MAX_BRAND = np.max(full_df.brand_name.max()) + 1
    #MAX_CONDITION = np.max(full_df.item_condition_id.max()) + 1
    #MAX_DESC_LEN = np.max(full_df.desc_len.max()) + 1
    #MAX_NAME_LEN = np.max(full_df.name_len.max()) + 1
    #MAX_SUBCAT_0 = np.max(full_df.subcat_0.max()) + 1
    #MAX_SUBCAT_1 = np.max(full_df.subcat_1.max()) + 1
    #MAX_SUBCAT_2 = np.max(full_df.subcat_2.max()) + 1
    
    #### PHASE 2: RNN
    #np.random.seed(123)
    #  
    ## Set hyper parameters for the model.
    #BATCH_SIZE = 512 * 3
    #epochs = 2
    #
    ## Calculate learning rate decay.
    #exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
    #steps = (n_trains/ BATCH_SIZE) * epochs
    #lr_init, lr_fin = 0.02, 0.005
    #lr_decay = exp_decay(lr_init, lr_fin, steps)
    #
    #X_train = get_rnn_data(full_df.iloc[:n_trains])
    #Y_train = full_df.iloc[:n_trains].target.values.reshape(-1, 1)
    #
    #X_dev = get_rnn_data(full_df.iloc[n_trains:n_trains+n_devs])
    #Y_dev = full_df.iloc[n_trains:n_trains+n_devs].target.values.reshape(-1, 1)
    #
    # Create model and fit it with training dataset.
    #rnn_model = new_rnn_model(lr=lr_init, decay=lr_decay)
    
    #rnn_model.fit(
    #            X_train,
    #            Y_train,
    #            epochs=4,
    #            batch_size=BATCH_SIZE,
    #            validation_data=(X_dev, Y_dev),
    #            verbose=1,
    #          )
    #
    #X_test = get_rnn_data(full_df[n_trains+n_devs:])
    #
    #if SUBMIT:
    #  preds_rnn_test = rnn_model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
    #else:
    #  print_time('Evaluating the model on validation data...')
    #  X_dev = get_rnn_data(full_df.iloc[n_trains:n_trains+n_devs])
    #  Y_dev = full_df.iloc[n_trains:n_trains+n_devs].target.values.reshape(-1, 1)
    #  preds_rnn_dev = rnn_model.predict(X_dev, batch_size=BATCH_SIZE)
    #  print(" RMSLE error:", rmsle(Y_dev, preds_rnn_dev))
    #
    #del rnn_model
    
    print_time('RNN finished')
    ### END RNN
    
    ### PHASE 3: FM_FTRL
    print_time('Creating sparse matrix.')
    X_name = tokenizer_name.x
    del tokenizer_name
    print_time('Done with seq_to_matrix for seq_name.')
    
    X_descp = tokenizer_descp.x
    del tokenizer_descp
    print_time('Done with seq_to_matrix for seq_descp.')
    
    def quant_name(x):
      if x == 0:
        return 0
      if x <= 6:
        return 1
      else:
        return 2
    
    def quant_descp(x):
      if x == 0:
        return 0
      if x <= 9:
        return 1
      if x <= 18:
        return 2
      if x <= 36:
        return 3
      if x <= 64:
        return 4
      if x <= 86:
        return 5
      else:
        return 6
    
    full_df['name_len'] = full_df.name_len.apply(quant_name)
    full_df['desc_len'] = full_df.desc_len.apply(quant_descp)
    
    lb = LabelBinarizer(sparse_output=True)
    X_brand = lb.fit_transform(full_df.brand_name)
    X_category = lb.fit_transform(full_df.category_name)
    X_cat0 = lb.fit_transform(full_df.subcat_0)
    X_cat1 = lb.fit_transform(full_df.subcat_1)
    X_cat2 = lb.fit_transform(full_df.subcat_2)
    X_condition = lb.fit_transform(full_df.item_condition_id)
    X_name_len = lb.fit_transform(full_df.name_len)
    X_desc_len = lb.fit_transform(full_df.desc_len)
    del lb
    
    X_shipping = csr_matrix(pd.get_dummies(full_df.shipping, sparse=True).values)
    #X_has_pic = csr_matrix(pd.get_dummies(full_df.has_pic, sparse=True).values)
    #X_has_brand_on_name = csr_matrix(pd.get_dummies(full_df.has_brand_on_name, sparse=True).values)
    #X_has_brand_on_descp = csr_matrix(pd.get_dummies(full_df.has_brand_on_descp, sparse=True).values)
    
    sparse_merge = hstack((
                     X_name,
                     X_descp,
                     X_brand,
                     X_category,
                     X_cat0,
                     X_cat1,
                     X_cat2,
                     X_condition,
                     X_shipping,
                     #X_has_pic,
                     #X_has_brand_on_name,
                     #X_has_brand_on_descp,
                     X_name_len,
                     X_desc_len,
                   )).tocsr()
    
    X_train = sparse_merge[:n_trains]
    X_dev = sparse_merge[n_trains:n_trains+n_devs]
    X_test = sparse_merge[n_trains+n_devs:]
    
    Y_train = full_df.iloc[:n_trains].target.values.reshape(-1)
    Y_dev = full_df[n_trains:n_trains+n_devs].target
    
    print_time('Fitting FM_FTRL.')
    model = FM_FTRL(alpha=0.01, beta=0.01, L1=1, L2=0.001, \
                    D=sparse_merge.shape[1], alpha_fm=0.021, \
                    L2_fm=0.0, init_fm=0.01, D_fm=400, e_noise=0.0001, \
                    iters=25, inv_link="identity", threads=1)
    
    model.fit(X_train, Y_train)
    
    if SUBMIT:
      preds_fm_test = model.predict(X=X_test)
    else:
      preds_fm_dev = model.predict(X=X_dev)
      print(rmsle(Y_dev, preds_fm_dev))
    
    del model
    print_time('FM_FTRL finished')

    
    ########

    avg = 0.5*preds_fm_test + 0.5*predsR

    submission = pd.DataFrame({
          "test_id": test_ids,
          "price": np.expm1(preds_equal.reshape(-1)),
      
        })
    submission.to_csv("submission.csv", index=False)

