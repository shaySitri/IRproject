from flask import Flask, request, jsonify
import sys
from collections import Counter, OrderedDict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import re
from operator import itemgetter
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from time import time
from timeit import timeit
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import math
from itertools import chain
import time
import csv
from inverted_index_gcp import *

import hashlib
def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()

nltk.download('stopwords')
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links", "may", "first","follower","see", "history","people", "one", "two","part", "thumb", "including", "second", "following","many", "however", "would", "became"]
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
all_stopwords = english_stopwords.union(corpus_stopwords)


# global variables
idx_body = None
idx_title = None
idx_anchor = None
bm25 = None
pageView_dict = {}
pageRank_dict = {}
title_dict = {}
bodyLenDl = 0
maxi = 1
max_rank = 0

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        global title_dict
        with open('./titleDict/titleDict.pkl', 'rb') as f:
            title_dict = pickle.load(f)
        global idx_body
        idx_body = InvertedIndex.read_index("./indexBody/postings_gcp", "body_index")
        idx_body.norm = {}
        with open('./normDict/norm.pkl', 'rb') as f:
            idx_body.norm = pickle.load(f)
        global bodyLenDl
        bodyLenDl = len(idx_body.DL)
        global bm25
        bm25 = BM25_from_index(idx_body)
        global idx_anchor
        idx_anchor = InvertedIndex.read_index("./indexAnchor/postings_gcp", "index_anchor")
        global idx_title
        idx_title = InvertedIndex.read_index("./indexTitle/postings_gcp", "index_title")
        global pageView_dict
        with open('./pageView/pageviews.pkl', 'rb') as f:
            pageView_dict = pickle.load(f)
        global pageRank_dict
        reader = csv.reader(open('./pageRank/pageRank.csv', 'r'))
        global max_rank
        max_rank = 0
        for row in reader:
            id = row[0]
            val = float(row[1])
            pageRank_dict[id] = val
            if val > max_rank:
                max_rank = val
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


# Mutual Functios for both BM-25 and cosine similiarity: #
def get_posting_iter(index, term ,dir1):
    """
    This function returning the posting list of given term.

    Parameters:
    ----------
    index: inverted index
    dir1: directory to read from
    term: term to read
    """
    pls = index.read_posting_list(term, dir1)
    return pls


# BM-25 class: #
class BM25_from_index:
    """
    Best Match 25.
    ----------
    k1 : float, default 1.2
    b : float, default 0.7
    index: inverted index
    """
    def __init__(self, index, k1=1.2, b=0.7):
        self.b = b
        self.k1 = k1
        self.index = index
        self.N = len(index.DL)
        s = 0
        for val in index.DL.items():
            s = s + int(val[1])
        self.AVGDL = s / self.N

    def calc_idf(self, list_of_tokens):
        """
        This function calculate the idf values according to the BM25 idf formula for each term in the query.

        Parameters:
        -----------
        list_of_tokens: list of token representing the query. For example: ['look', 'blue', 'sky']

        Returns:
        -----------
        idf: dictionary of idf scores. As follows:
                                                    key: term
                                                    value: bm25 idf score
        """
        idf = {}
        for term in list_of_tokens:
            n_ti = self.index.df.get(term,0)
            if n_ti != 0:
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass
        return idf


    def small_search(self, query_tok,dir1,dic_pls ,dic_docs):
        """
        This function calculate the bm25 score for given query and document.
        We need to check only documents which are 'candidates' for a given query.
        This function return a dictionary of scores as the following:
                                                                    key: query_id
                                                                    value: score: float, bm25 score.

        Parameters:
        -----------
        query_tok: list of token representing the query. For example: ['look', 'blue', 'sky']
        dic_pls: dict of posting lists.

        Returns:
        -----------
        all_scores: dictionary of scores
        """

        global maxi
        maxi = 1
        all_scores = {}
        idf = self.calc_idf(query_tok)
        for term,pls in dic_pls.items():
            df = self.index.df.get(term,0)
            if df != 0:
                idf_term = idf[term]
                for doc,tf in pls:
                    if doc in dic_docs:
                        if tf > 4:
                            doc_len = self.index.DL[doc]
                            numerator = idf_term * tf * (self.k1 + 1)
                            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
                            score = all_scores.get(doc, 0) + (numerator / denominator)
                            all_scores[doc] = score
                            if maxi < score:
                                maxi = score
        return all_scores

    def search(self, query_tok,dir1,dic_pls):
        """
            This function calculate the bm25 score for given query and document.
            We need to check only documents which are 'candidates' for a given query.
            This function return a dictionary of scores as the following:
                                                                        key: query_id
                                                                        value: score: float, bm25 score.

            Parameters:
            -----------
            query_tok: list of token representing the query. For example: ['look', 'blue', 'sky']
            dic_pls: dict of posting lists.

            Returns:
            -----------
            all_scores: dictionary of scores
            """
        global maxi
        maxi = 1
        all_scores = {}
        idf = self.calc_idf(query_tok)
        for term,pls in dic_pls.items():
            df = self.index.df.get(term,0)
            if df != 0:
                idf_term = idf[term]
                for doc,tf in pls:
                    if tf > 4:
                        doc_len = self.index.DL[doc]
                        numerator = idf_term * tf * (self.k1 + 1)
                        denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
                        score = all_scores.get(doc, 0) + (numerator / denominator)
                        all_scores[doc] = score
                        if maxi < score:
                            maxi = score
        return all_scores




# cosine: #
def get_topN_score_for_queries_body_cosine(query_tok, index, dir1,dic_pls={}):
    """
    Generate a dictionary that gathers for every query its topN score.

    Parameters:
    -----------
    queries_to_search: a dictionary of queries as follows:
                                                        key: query_id
                                                        value: list of tokens.
    index:           inverted index loaded from the corresponding files.
    N: Integer. How many documents to retrieve. This argument is passed to the topN function. By default N = 3, for the topN function.

    Returns:
    -----------
    return: a dictionary of queries and topN pairs as follows:
                                                        key: query_id
                                                        value: list of pairs in the following format:(doc_id, score).
    """
    df_dict = {}
    tf_dict = {}
    idf_dict = {}
    new_query = []
    for q in query_tok:
        if q not in all_stopwords:
            df = index.df.get(q, 0)
            if df != 0:
                new_query.append(q)
                df_dict[q] = df
    qNfactor = 0
    tf_counter = Counter(new_query)
    for term in tf_counter:
        tf =  tf_counter[term]/len(new_query)
        tf_dict[term] = tf
        idf = math.log(bodyLenDl / (df_dict[term]), 10)  # smoothing
        idf_dict[term] = idf
        qNfactor += ((tf * idf) ** 2)
    qNfactor = qNfactor ** 0.5

    sim = {}
    for qi in set(new_query):
        idf = idf_dict[qi]
        tf = tf_dict[qi]
        if not dic_pls:
            list_of_doc = get_posting_iter(index, qi, dir1)
        else:
            list_of_doc = dic_pls[qi]
        for doc, freq in list_of_doc:
            tf_doc = (freq / index.DL[doc]) * idf
            sim[doc] = sim.get(doc, 0) + (tf * idf) * tf_doc/ (qNfactor * (index.norm[doc] ** 0.5))
    return sim




def get_topN_score_for_queries(query, index,dir1):
    """
    Generate a dictionary that gathers for every query its topN score.

    Parameters:
    -----------
    queries_to_search: a dictionary of queries as follows:
                                                        key: query_id
                                                        value: list of tokens.
    index:           inverted index loaded from the corresponding files.
    N: Integer. How many documents to retrieve. This argument is passed to the topN function. By default N = 3, for the topN function.

    Returns:
    -----------
    return: a dictionary of queries and topN pairs as follows:
                                                        key: query_id
                                                        value: list of pairs in the following format:(doc_id, score).
    """
    query_tok = [token.group() for token in RE_WORD.finditer(query.lower())]
    query_tok = set(query_tok)
    sim = {}
    for qi in query_tok:
        if qi not in all_stopwords:
            df = index.df.get(qi, 0)
            if df != 0:
                list_of_doc = get_posting_iter(index, qi,dir1)
                for doc, freq in list_of_doc:
                    sim[doc] = sim.get(doc, 0) + 1
    return sim

# merge
def try_merge_results(title_scores, dic_pls, body_scores_cosine, anchor_scores, num,query_tok ,title_weight=0.21, body_weight_BM25=0.41, body_weight_cosine=0.1, anchor_weight=0.1):
    """
    merging scores function

    """
    query_ = {}
    for doc in anchor_scores:
        score = doc[1] * anchor_weight
        query_[doc[0]] = (query_.get(doc[0], 0) + score)

    for doc in title_scores:
        score = doc[1] * title_weight / num
        query_[doc[0]] = (query_.get(doc[0], 0) + score)

    body_scores_BM25 = bm25.small_search(query_tok,'./indexBody/postings_gcp', dic_pls,query_)
    for doc in query_:
        score = body_scores_BM25.get(doc, 0) * body_weight_BM25 + (pageView_dict[doc] * 0.2 / 181126232)
        query_[doc] = query_.get(doc, 0) + (score/maxi)
        score = body_scores_cosine.get(doc, 0) * body_weight_cosine
        query_[doc] = query_.get(doc, 0) + score

    return query_

@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    dic_pls = {}
    query1 = [token.group() for token in RE_WORD.finditer(query.lower())]  ### this function can be better - (time)
    query_tok = set(query1)
    count = 0
    for term in query_tok:
        if term not in all_stopwords:
            exist_df = idx_body.df.get(term, 0)
            if exist_df != 0:
                count += 1
                pls = get_posting_iter(idx_body, term, './indexBody/postings_gcp')
                dic_pls[term] = pls

    title_dic = get_topN_score_for_queries(query, idx_title, './indexTitle/postings_gcp')
    _temp_t = sorted([(doc_id, score) for doc_id, score in title_dic.items()], key=lambda x: x[1], reverse=True)[:100]
    cosine_dic = get_topN_score_for_queries_body_cosine(query1, idx_body, './indexBody/postings_gcp', dic_pls)
    anchor_dic = get_topN_score_for_queries(query, idx_anchor,'./indexAnchor/postings_gcp')
    _temp_a = sorted([(doc_id, score) for doc_id, score in anchor_dic.items()], key=lambda x: x[1], reverse=True)[:100]
    merge = try_merge_results(_temp_t,dic_pls,cosine_dic,_temp_a,count,query_tok)
    temp = sorted([(doc_id, score) for doc_id, score in merge.items()], key=lambda x: x[1], reverse=True)[:70]

    for tup in temp:
        res.append((tup[0], title_dict[tup[0]]))
    # END SOLUTION

    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    query_tok = [token.group() for token in RE_WORD.finditer(query.lower())]
    dict_body = get_topN_score_for_queries_body_cosine(query_tok, idx_body, './indexBody/postings_gcp')
    temp = sorted([(doc_id, score) for doc_id, score in dict_body.items()], key=lambda x: x[1], reverse=True)[:100]
    for tup in temp:
        res.append((tup[0], title_dict[tup[0]]))
    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO 
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. For example, a document 
        with a title that matches two distinct query words will be ranked before a 
        document with a title that matches only one distinct query word, 
        regardless of the number of times the term appeared in the title (or 
        query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    sim = get_topN_score_for_queries(query, idx_title,'./indexTitle/postings_gcp')
    temp = sorted([(doc_id, score) for doc_id, score in sim.items()], key=lambda x: x[1], reverse=True)
    for tup in temp:
        res.append((tup[0], title_dict[tup[0]]))
    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment 
        3 (GCP part) to do the tokenization and remove stopwords. For example, 
        a document with a anchor text that matches two distinct query words will 
        be ranked before a document with anchor text that matches only one 
        distinct query word, regardless of the number of times the term appeared 
        in the anchor text (or query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    sim = get_topN_score_for_queries(query, idx_anchor,'./indexAnchor/postings_gcp')
    temp = sorted([(doc_id, score) for doc_id, score in sim.items()], key=lambda x: x[1], reverse=True)
    for tup in temp:
        res.append((tup[0], title_dict[tup[0]]))
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    for d_id in wiki_ids:
        res.append(pageRank_dict.get(str(d_id), 0))
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    for d_id in wiki_ids:
        res.append(pageView_dict.get(d_id, 0))
    # END SOLUTION
    return jsonify(res)

if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
