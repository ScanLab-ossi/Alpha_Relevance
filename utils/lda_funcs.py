import pickle
import pandas as pd
import numpy as np
import emoji
import regex
import re
import string
from ast import literal_eval

#Natural Language Processing (NLP)
import torch
import spacy
import gensim
from spacy.tokenizer import Tokenizer
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from gensim.parsing.preprocessing import STOPWORDS as SW
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sparse_cos_sim
from sklearn.model_selection import GridSearchCV
from sentence_transformers import SentenceTransformer,util #  util.cos_sim
from scipy.sparse import vstack



from pprint import pprint
from wordcloud import STOPWORDS
stopwords = set(STOPWORDS)
from tqdm import tqdm
tqdm.pandas()
from tqdm import tqdm_notebook as tqdm
#Warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'


# Create a an ALL STOP WORDS list and a Tokenizer
nlp = spacy.load('en_core_web_lg')
# Tokenizer
tokenizer = Tokenizer(nlp.vocab)
# Custom stopwords
custom_stopwords = ['hi','\n','\n\n', '&amp;', ' ', '.', '-', 'got', "it's", 'it’s', "i'm", 'i’m', 'im', 'want', 'like', '$', '@','objective','opportunity','project']

# Customize stop words by adding to the default list
STOP_WORDS = nlp.Defaults.stop_words.union(custom_stopwords)

# ALL_STOP_WORDS = spacy + gensim + wordcloud
ALL_STOP_WORDS = STOP_WORDS.union(SW).union(stopwords)

# Lemmatizer functions
def give_emoji_n_url_free_text(text):
    """
    Removes emoji's and special characters as registered trademarks , tm etc which can be found in publications
    Accepts:
        Text
    Returns:
        Text - emoji/special characters free
    """
    emoji_list = emoji.distinct_emoji_list(text)
    clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])
    pattern = r"http\S+"
    clean_text = re.sub(pattern, "", clean_text) # REMOVE LINKs
    return clean_text

    # preprocesss text function
def preprocess(text):
    """
    Parses a string into a list of semantic units (words)
    Args:
        text (str): The string that the function will tokenize.
    Returns:
        list: tokens parsed out
    """
    # Removing url's
    tokens = give_emoji_n_url_free_text(text) #remove emojis
    tokens = re.sub('[^a-zA-Z 0-9]', '', tokens)
    tokens = re.sub('[%s]' % re.escape(string.punctuation), '', tokens) # Remove punctuation
    tokens = re.sub('\w*\d\w*', '', tokens) # Remove words containing numbers
    tokens = re.sub('@*!*\$*', '', tokens) # Remove @ ! $

    # Make text lowercase and split it
    tokens = tokens.lower()
    return tokens

def remove_stop_words(doc):
  words = [word for word in doc.split() if word.lower() not in ALL_STOP_WORDS]
  new_text = " ".join(words)
  return new_text

def get_lemmas(text):
    '''Used to lemmatize the processed text'''
    lemmas = []

    doc = nlp(text)

    for token in doc:
        if ((token.is_stop == False) and (token.is_punct == False)) and (token.pos_ != 'PRON'):
            lemmas.append(token.lemma_)
    lemmas_to_text = " ".join(lemmas)
    return lemmas_to_text

def preprocess_removestops_lemmatize(doc):
    '''
    Performs a complete pipeline of:
                                    preprocessing text,
                                    removing stopwords (costume stop words included alsp),
                                    and finally lemmatizing the text.
    '''
    text_p = preprocess(doc)
    text_p_restps = remove_stop_words(text_p)
    lemm_text = get_lemmas(text_p_restps)
    return lemm_text


def get_n_keywords_with_tfidf(corpus:list,n= 5):
    # initialize the TfidfVectorizer
    pubsvectorizer = TfidfVectorizer(max_df=0.99, min_df=1,max_features = 10000,dtype=np.float32,ngram_range=(1, 3))

    # fit the vectorizer to the corpus
    X = pubsvectorizer.fit_transform(corpus)

    # get the feature names (i.e., the keywords)
    feature_names = pubsvectorizer.get_feature_names()
    list_of_keywords = []
    with Pool() as p:
        list_of_keywords = p.map(get_keywords_for_doc, [(X[i], feature_names, n) for i in range(X.shape[0])])
    return list_of_keywords

def get_keywords_for_doc(params):
    doc, feature_names, n = params
    doc = doc.toarray()[0]
    top_indices = doc.argsort()[-n:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    final_words =  " ".join(top_features)
    return final_words

def compute_coherence_values(dictionary, corpus, texts,cores, limit=10, start=2, step=1):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the
    LDA model with respective number of topics
    """
    coherence_values_topic = []
    model_list_topic = []
    for num_topics in tqdm(range(start, limit, step)):
        model = LdaMulticore(corpus=corpus,
                             num_topics=num_topics,
                             id2word=id2word,
                             workers=cores-1,
                             passes=20,
                             random_state=12,
                             decay=0.5,
                            chunksize=2000,)
        model_list_topic.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values_topic.append(coherencemodel.get_coherence())
    best_model = model_list_topic[np.argmax(coherence_values_topic)]
    return best_model

def get_most_n_dominant_words_from_LDA(text,model,dictionary,n_keywords,word_bank):
  topics_n_prob = model.get_document_topics(dictionary.doc2bow(text))
  topics_n_wordsnum = [(a,round(b*n_keywords)) if round(b*n_keywords) >0 else (a,1) for a,b in topics_n_prob]
  li = []
  for topic,num in topics_n_wordsnum:
    li+= word_bank[topic][0:num]
  li = [value for value in li if value != '   ']
  return " ".join(li)
def literal(x):
    if x != '':
        return literal_eval(x)
    return x