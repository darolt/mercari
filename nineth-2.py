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
#from nltk.stem.snowball import SnowballStemmer
import tensorflow as tf
import string, re
import warnings
from collections import OrderedDict
from six.moves import range
from six.moves import zip

from wordbatch.models import FM_FTRL
import lightgbm as lgb

SUBMIT = True 

#stemmer = SnowballStemmer("english")
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
#pat_dis = re.compile(r'([0-9]+)(m|cm|inches|inch|in|mm|pc)[\.\!\s\)]')
#pat_mass = re.compile(r'([0-9]+)(g|kg|lbs)[\.\!\s\)]')
#pat_vol = re.compile(r'([0-9]+)(ml|liters|milliliters|oz|gal|ct)[\)\.\!\s]')
pat_per = re.compile(r'([0-9]+)[\s]*(%)[\.\!\s\)]')
pat_sep_num = re.compile(r'([0-9])([^0-9\s\.])')
pat_splitter = re.compile(r'\s*[^a-z0-9\s]+\s*')

def my_text_to_word_sequence(text):
    text = text.lower() + '.'
    text = pat_inches.sub(r'\1 inches ', text)
    #text = pat_dis.sub(r'\1 \2 ', text)
    #text = pat_mass.sub(r'\1 \2 ', text)
    #text = pat_vol.sub(r'\1 \2 ', text)
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
            for i, w in enumerate(seq):
                #if w_original in self.word_stem:
                #  w = self.word_stem[w_original]
                #else:
                #  w = stemmer.stem(w_original)
                #  self.word_stem[w_original] = w
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
                #if i<2:
                #  continue
                #concat = seq[i-2] + '_' + seq[i-1] + '_' + w
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
          self.bigram_counts = dict(bicounts[:1000])
          self.bigram_index = {k:i for i, k in enumerate(self.bigram_counts.keys())}

          #tricounts = list(self.trigram_counts.items())
          #tricounts.sort(key=lambda x: x[1], reverse=True)
          #print(tricounts[:40])
          #self.trigram_counts = dict(tricounts[:250])
          #self.trigram_index = {k:i for i, k in enumerate(self.trigram_counts.keys())}

    def texts_to_sequences(self, sequences):
        self.offset = len(self.word_index) + 2
        #self.offset2 = self.offset + len(self.bigram_index)
        num_words = self.offset + len(self.bigram_index)
        #self.x = lil_matrix((len(sequences), num_words), dtype=np.float64)
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

    def get_word_index(self, w):
      word_idx = self.word_index.get(w)
      if word_idx is None:
        #if w in self.word_stem:
        #  w = self.word_stem[w] 
        #else:
        #  w = stemmer.stem(w)
        if pat_date.match(w):
          return (self.offset-2, w)
        if pat_number.match(w):
          return (self.offset-1, w)
        w = trans(w)
        word_idx = self.word_index.get(w)
      return (word_idx, w)

    def texts_to_sequences_generator(self, sequences):
        for seq_idx, seq in enumerate(sequences):
          vect = []
          idx_to_add = set()
          prev_word = None
          prev_idx = None
          for w in seq:
            word_idx, w = self.get_word_index(w) 
            # look for bigram
            if prev_idx and word_idx:
              concat = prev_word + '_' + w
              if concat in self.bigram_index:
                bigram_idx = self.offset+self.bigram_index[concat]
                vect.append(word_idx)
                idx_to_add.add(word_idx)
                idx_to_add.add(bigram_idx)                
                prev_word, prev_idx = (None, None)
                continue

            # look for unigram
            if word_idx is not None:
              vect.append(word_idx)
              idx_to_add.add(word_idx)

            prev_word, prev_idx = (w, word_idx)

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

#full_set = pd.concat([train_df,test_df])
all_brands = set(full_df['brand_name'].astype(str).values)
full_df.brand_name.fillna(value='missing', inplace=True)

# split category name into 3 parts
full_df['subcat_0'], full_df['subcat_1'], full_df['subcat_2'] = \
  zip(*full_df['category_name'].apply(lambda x: split_cat(x)))

#brands_per_cat0 = full_df.groupby('subcat_0')['brand_name'].apply(lambda x: set(x.str.lower())).to_dict()

premissing = len(full_df.loc[full_df['brand_name'] == 'missing'])
full_df.loc[full_df.brand_name == 'missing', 'brand_name'] = \
  full_df[full_df.brand_name == 'missing']['name'].apply(brandfinder)
  #full_df[full_df.brand_name == 'missing'][['name', 'subcat_0']].apply(slow_brandfinder, axis=1)

found = premissing - len(full_df.loc[full_df['brand_name'] == 'missing'])
print("Found %d missing brands." %(found))

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

le = LabelEncoder()

full_df['category'] = le.fit_transform(full_df.category_name)
full_df.brand_name = le.fit_transform(full_df.brand_name)
full_df.subcat_0 = le.fit_transform(full_df.subcat_0)
full_df.subcat_1 = le.fit_transform(full_df.subcat_1)
full_df.subcat_2 = le.fit_transform(full_df.subcat_2)

del le

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

MAX_NAME_SEQ = 8
MAX_ITEM_DESC_SEQ = 30
MAX_CATEGORY = np.max(full_df.category.max()) + 1
MAX_BRAND = np.max(full_df.brand_name.max()) + 1
MAX_CONDITION = np.max(full_df.item_condition_id.max()) + 1
MAX_DESC_LEN = np.max(full_df.desc_len.max()) + 1
MAX_NAME_LEN = np.max(full_df.name_len.max()) + 1
MAX_SUBCAT_0 = np.max(full_df.subcat_0.max()) + 1
MAX_SUBCAT_1 = np.max(full_df.subcat_1.max()) + 1
MAX_SUBCAT_2 = np.max(full_df.subcat_2.max()) + 1

### PHASE 2: RNN
np.random.seed(123)
  
# Set hyper parameters for the model.
BATCH_SIZE = 512 * 3
epochs = 2

# Calculate learning rate decay.
exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
steps = (n_trains/ BATCH_SIZE) * epochs
lr_init, lr_fin = 0.02, 0.005
lr_decay = exp_decay(lr_init, lr_fin, steps)

X_train = get_rnn_data(full_df.iloc[:n_trains])
Y_train = full_df.iloc[:n_trains].target.values.reshape(-1, 1)

X_dev = get_rnn_data(full_df.iloc[n_trains:n_trains+n_devs])
Y_dev = full_df.iloc[n_trains:n_trains+n_devs].target.values.reshape(-1, 1)

# Create model and fit it with training dataset.
rnn_model = new_rnn_model(lr=lr_init, decay=lr_decay)

rnn_model.fit(
            X_train,
            Y_train,
            epochs=3,
            batch_size=BATCH_SIZE,
            validation_data=(X_dev, Y_dev),
            verbose=1,
          )

X_test = get_rnn_data(full_df[n_trains+n_devs:])

if SUBMIT:
  preds_rnn_test = rnn_model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
else:
  print_time('Evaluating the model on validation data...')
  X_dev = get_rnn_data(full_df.iloc[n_trains:n_trains+n_devs])
  Y_dev = full_df.iloc[n_trains:n_trains+n_devs].target.values.reshape(-1, 1)
  preds_rnn_dev = rnn_model.predict(X_dev, batch_size=BATCH_SIZE)
  print(" RMSLE error:", rmsle(Y_dev, preds_rnn_dev))

del rnn_model

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
                D=sparse_merge.shape[1], alpha_fm=0.018, \
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

#print_time('Fitting LGB...') 
#params = {
#      'learning_rate': 0.6,
#      'application': 'regression',
#      'num_leaves': 31,
#      'verbosity': -1,
#      'metric': 'RMSE',
#      'data_random_seed': 1,
#      'bagging_fraction': 0.6,
#      'bagging_freq': 5,
#      'feature_fraction': 0.65,
#      'nthread': 4,
#      'min_data_in_leaf': 100,
#      'max_bin': 31,
#}
#
#d_train = lgb.Dataset(X_train, label=Y_train)
#watchlist = [d_train]
#d_valid = lgb.Dataset(X_dev, label=Y_dev)
#watchlist = [d_train, d_valid]
#
#model = lgb.train(params, train_set=d_train, num_boost_round=1000, valid_sets=watchlist, \
#                  early_stopping_rounds=100, verbose_eval=100)
#
#if SUBMIT:
#  preds_lgb_test = model.predict(X_test)
#else:
#  preds_lgb_dev = model.predict(X_dev)
#  print(rmsle(Y_dev, preds_lgb_dev))
#
#print_time('LGB done.') 

# SUBMIT
if SUBMIT:
  preds_equal = 0.5*preds_rnn_test.reshape((-1)) + 0.5*preds_fm_test
  submission = pd.DataFrame({
          "test_id": test_ids,
          "price": np.expm1(preds_equal.reshape(-1)),
      })
  submission.to_csv("submission.csv", index=False)
else:
  print_time('RMSLE half/half:')
  preds_equal = 0.5*preds_rnn_dev.reshape((-1)) + 0.5*preds_fm_dev
  print(rmsle(Y_dev, preds_equal))

