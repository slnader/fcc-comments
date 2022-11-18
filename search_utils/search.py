import math
import numpy as np
from scipy import sparse

class Comments:
    """
    Class for the comment corpus to be ranked
    """

    def __init__(self, dtm, weights = None):
        """
        Initializes instance of comment corpus using sparse csr
        document term matrix.
        Args:
            dtm: scipy sparse csr matrix. Documents are on the rows, terms are
            on the columns.
            weights: np ndarray of frequency weights for each document
            (represents duplicates without repeating data)
        """
        #check for sparse csr format
        if isinstance(dtm, sparse.csr.csr_matrix):
            self.__dtm = dtm
            if weights is None:
                self.__weights = np.ones(dtm.shape[0])
            else:
                self.__weights = weights
            self.__ld = np.asarray(dtm.sum(axis = 1)).flatten()
        else:
            raise ValueError("""Document-term matrix must be in
            sparse csr format.""")

    def __repr__(self):
        """
        Print stats on the comment corpus
        """
        print_string = "Number of documents: " + str(self.__weights.sum()) + \
        "\n" + "Number of terms: " + str(self.__dtm.shape[1]) + "\n" + \
        "Average document length: " + str(np.average(self.__ld,
        weights = self.__weights))

        return print_string

    def set_ld(self, new_ld):
        """
        Option to set word lengths from outside metadata if dtm does not
        include full vocabulary
        Args:
            new_ld: np.array, len(new_ld) = dtm.shape[0]
        """
        if (len(new_ld) == self.__dtm.shape[0]) & (isinstance(new_ld,
        np.ndarray)):
            self.__ld = new_ld
        else:
            raise ValueError("""Word length must be np.ndarray of same length as
            number of documents.""")

    def get_ld(self):
        """
        Returns np.ndarray of document lengths.
        """
        return self.__ld

    def get_dtm(self):
        """
        Returns document-term matrix in sparse csr format.
        """
        return self.__dtm

    def get_weights(self):
        """
        Returns np.ndarray of document weights.
        """
        return self.__weights

class Queries:
    """
    Class for the queries to search over
    """

    def __init__(self, qtm):
        """
        Initializes instance of queries using sparse csr query term matrix.
        Args:
            qtm: scipy sparse csr matrix. Queries are on the rows, terms are
            on the columns
        """
        if isinstance(qtm, sparse.csr.csr_matrix):
            #initialize qtm
            self.__qtm = qtm
            self.__ld = np.asarray(qtm.sum(axis = 1)).flatten()
        else:
            raise ValueError("""Query-term matrix must be in
            sparse csr format.""")

    def __repr__(self):
        """
        Print stats on the queries
        """
        print_string = "Number of queries: " + str(self.__qtm.shape[0]) + \
        "\n" + "Number of terms: " + str(self.__qtm.shape[1]) + "\n" + \
        "Average query length: " + str(self.__ld.mean())

        return print_string

    def get_qtm(self):
        """
        Returns query-term matrix in sparse csr format
        """
        return self.__qtm

class bm25:
    """
    Class for bm25 algorithm to apply to comments and queries
    """

    def __init__(self, k, b):
        """
        BM25 attire variant
        Ranks comments by their relevance to queries using tf*idf weights.
        Args:
            k: controls the rate at which the tf term saturates to its maximum
            as term frequency gets large (larger = slower saturation,
            smaller = faster saturation)
            b: controls normalization by document length (b = 0 = no
            normalization, b = 1 = full normalization)
        """
        self.k = k
        self.b = b
        self.__doc_freq = None
        self.__queries = None
        self.__ndocs = None
        self.__lavg = None

    def __repr__(self):
        """ Print stats on the build """
        print_string = "Number of documents: " + str(self.__ndocs) + \
        "\n" + "Average document length: " + str(self.__lavg) + "\n" + \
        "Average document frequency for terms: " + str(self.__doc_freq.mean())

        return print_string

    def build(self, comments, queries):
        """
        Returns algorithm built with full corpus constants
        Args:
            comments: Comments object containing dtm of full corpus
            queries: Queries Object containing qtm of all queries
        Prereqs:
            ncols(dtm) == ncols(qtm)
        """
        #check matching column dimensions
        qtm = queries.get_qtm()
        dtm = comments.get_dtm()
        if dtm.shape[1] != qtm.shape[1]:
            raise ValueError("""Number of terms must match.""")

        #calculate document frequencies for each query term
        nz_indices = dtm.nonzero()
        doc_weights = comments.get_weights()
        doc_freq = np.asarray([doc_weights[nz_indices[0]
        [np.where(nz_indices[1] == x)]].sum()
        for x in range(qtm.shape[1])])

        #average doc length
        lavg = np.average(comments.get_ld(), weights = doc_weights)

        #store model build
        self.__doc_freq = doc_freq
        self.__queries = queries
        self.__ndocs = np.sum(comments.get_weights())
        self.__lavg = lavg

        return self

    def _compute_idf(self):
        """
        Returns the idf weights in sparse diagonal matrix
        Prereqs:
            build() method has been run
        """

        #Create diagonal matrix of weights
        T = self.__queries.get_qtm().shape[1]
        N = self.__ndocs
        diag = np.zeros(T)
        for i in range(T):
            if self.__doc_freq[i] > 0:
                diag[i] = math.log(N/self.__doc_freq[i])
        weight_mat = sparse.diags(diag, 0)

        return weight_mat

    def _compute_tf(self, comments):
        """
        Returns the tf weights in sparse csr matrix
        Args:
            comments: Comments object containing comments to score
        Prereqs:
            build() method has been run
        """

        #get document lengths
        document_lengths = comments.get_ld()
        dtm = comments.get_dtm()

        #calculate term frequency weights
        tf_array = []
        rows, cols = dtm.nonzero()
        for row, col in zip(rows, cols):
            new_value = ((self.k + 1) * dtm[row, col]) / \
            (dtm[row, col] + self.k * (1 - self.b + self.b *
            (document_lengths[row] / self.__lavg)))
            tf_array.append(new_value)

        #put into csr matrix
        freq_mat = sparse.csr_matrix((tf_array, (rows, cols)),
        shape = dtm.shape)

        return freq_mat


    def score(self, comments):
        """
        Returns the raw scores in sparse csr matrix, n documents x n queries
        Args:
            comments: Comments object containing batch of comments to score
        Prereqs:
            build() method has been run
        """

        if self.__queries is None:
            raise ValueError("""Must run bm25.build() before bm25.score()
            to score comments.""")

        #get weight components
        qtm = self.__queries.get_qtm()
        idf = self._compute_idf()
        tf = self._compute_tf(comments)

        #Get tfidf matrix
        trans_mat = tf@idf
        score_mat = trans_mat@qtm.transpose()

        #Normalize by term length
        query_lengths = qtm@np.ones(qtm.shape[1])
        for j in range(len(query_lengths)):
            score_mat[:, j] = score_mat[:, j]/query_lengths[j]

        return score_mat
