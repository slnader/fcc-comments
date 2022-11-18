import math
import numpy as np
from scipy import sparse
from search_utils import search

def main():
    #comment corpus
    row = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    col = np.array([1, 2, 4, 0, 1, 2, 3, 4])
    data = np.array([1, 2, 1, 2, 1, 1, 1, 3])
    corpus_dtm = sparse.csr_matrix((data, (row, col)),  shape=(3,5))

    #comment batch from corpus
    batch_dtm = corpus_dtm[:2,].copy()

    #some queries
    row = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    col = np.array([0, 3, 4, 2, 3, 0, 1, 4])
    data = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    qtm = sparse.csr_matrix((data, (row, col)),  shape=(3,5))

    #initialize comments and queries
    batch = search.Comments(batch_dtm)
    queries = search.Queries(qtm)
    corpus = search.Comments(corpus_dtm)

    #summarize
    print(corpus)
    print(batch)
    print(queries)

    #build model one time
    bm25_v1 = search.bm25(k = 0.9, b = 0.4).build(corpus, queries)
    print(bm25_v1)

    #score one batch at a time
    print(bm25_v1.score(batch).toarray())

    #score whole corpus at once
    print(bm25_v1.score(corpus).toarray())

if __name__ == '__main__':
    main()
