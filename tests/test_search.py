import math
import numpy as np
import pytest
from scipy import sparse
from search_utils import search

#set up data for tests

#simple dtm
@pytest.fixture
def test_dtm():
    row = np.array([0, 0, 0, 1, 1])
    col = np.array([1, 2, 4, 0, 1])
    data = np.array([1, 2, 1, 2, 1])
    test_dtm = sparse.csr_matrix((data, (row, col)),  shape=(2,5))
    return test_dtm

#larger dtm
@pytest.fixture
def corpus_dtm():
    row = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    col = np.array([1, 2, 4, 0, 1, 2, 3, 4])
    data = np.array([1, 2, 1, 2, 1, 1, 1, 3])
    corpus_dtm = sparse.csr_matrix((data, (row, col)),  shape=(3,5))
    return corpus_dtm

#simple qtm
@pytest.fixture
def test_qtm():
    row = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    col = np.array([0, 3, 4, 2, 3, 0, 1, 4])
    data = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    test_qtm = sparse.csr_matrix((data, (row, col)),  shape=(3,5))
    return test_qtm

def test_ld(test_dtm):
    """
    Test ld assignment
    """
    #initialize document lengths
    comments = search.Comments(test_dtm)
    assert comments.get_ld().mean() == 3.5
    #manually change document lengths
    comments.set_ld(np.array([10, 20]))
    assert comments.get_ld().mean() == 15.0

def test_score_small(test_dtm, test_qtm):
    """
    Test algorithm build and score when corpus = comment batch
    """
    #initialize comments and queries
    comments = search.Comments(test_dtm)
    queries = search.Queries(test_qtm)
    #build model
    bm25_model = search.bm25(k = 0.9, b = 0.4)
    rank = bm25_model.build(comments, queries).score(comments)
    results = rank.toarray()
    #check results
    expected = math.log(2/1)*((0.9 + 1) * 1 / ((0.9 * (1 - 0.4 + 0.4 * (4/3.5)))
    + 1)) / 3
    assert pytest.approx(results[0][0]) == pytest.approx(expected)

def test_score_corpus(test_dtm, test_qtm, corpus_dtm):
    """
    Test algorithm build and score when corpus and comment batch are different
    """
    #initialize comments and queries
    comments = search.Comments(test_dtm)
    queries = search.Queries(test_qtm)
    corpus = search.Comments(corpus_dtm)
    #build model
    bm25_model = search.bm25(k = 0.9, b = 0.4)
    rank = bm25_model.build(corpus, queries).score(comments)
    results = rank.toarray()
    #check results
    expected = math.log(3/2)*((0.9 + 1) * 1 / ((0.9 * (1 - 0.4 + 0.4 * (4/4)))
    + 1)) / 3
    assert pytest.approx(results[0][0]) == pytest.approx(expected)

def test_score_vocab(test_dtm, test_qtm, corpus_dtm):
    """
    Test algorithm build and score when corpus and comment batch are different
    and vocabulary is larger
    """
    #initialize comments and queries
    comments = search.Comments(test_dtm)
    queries = search.Queries(test_qtm)
    corpus = search.Comments(corpus_dtm)
    corpus.set_ld(np.array([11, 15, 13]))

    #build model
    bm25_model = search.bm25(k = 0.9, b = 0.4)
    rank = bm25_model.build(corpus, queries).score(comments)
    results = rank.toarray()
    #check results
    expected = math.log(3/2)*((0.9 + 1) * 1 / ((0.9 * (1 - 0.4 + 0.4 * (4/13)))
    + 1)) / 3
    assert pytest.approx(results[0][0]) == pytest.approx(expected)

def test_score_weighted(test_dtm, test_qtm, corpus_dtm):
    """
    Test algorithm build and score when corpus and comment batch are different
    and vocabulary is larger
    """
    #initialize comments and queries
    comments = search.Comments(test_dtm)
    queries = search.Queries(test_qtm)
    corpus = search.Comments(corpus_dtm, np.array([2,1,10]))
    #build model
    bm25_model = search.bm25(k = 0.9, b = 0.4)
    rank = bm25_model.build(corpus, queries).score(comments)
    results = rank.toarray()
    lavg = (4*2 + 3*1 + 5*10)/13
    #check results
    expected = math.log(13/12)*((0.9 + 1) * 1 / ((0.9 * (1 - 0.4 + 0.4 * (4/lavg)))
    + 1)) / 3
    assert pytest.approx(results[0][0]) == pytest.approx(expected)
