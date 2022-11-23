from joblib import Parallel, delayed
import math
import numpy as np
import os
import pandas as pd
import pickle
import random
from scipy import sparse
from search_utils import search, helpers
import sys

def distribute_bm25(dtm, index, bm25):
    """
    Distributes computational load of batch bm25 calculation
    Args:
    dtm: sparse csr document term matrix for batch
    index: search index corresponding to dtms
    bm25: initialized bm25 model
    """
    #instantiate comments
    comment_batch = search.Comments(dtm)
    comment_batch.set_ld(index['term_length'].values)

    #Score batch
    bm25_mat = bm25.score(comment_batch)

    return bm25_mat, index

def main():

    #===========================================================================
    #FCC queries
    #===========================================================================

    print('Reading in queries...')

    #Document term matrix for queries
    pickle_in = open("data/pickles/query_dtm.pickle","rb")
    sp_query_dtm = pickle.load(pickle_in)

    #Get query selector
    non_zero_entries = sp_query_dtm.nonzero()

    #Create boolean sparse mat
    query_boolean = sparse.csr_matrix(([1]*sp_query_dtm.getnnz(),
                                        non_zero_entries))

    #Calculate term length for queries
    query_lengths = query_boolean@np.ones(query_boolean.shape[1])

    #===========================================================================
    #Comment DTMs
    #===========================================================================
    print('Reading in comment dtms...')

    #Document term matrix for corpus
    pickle_in = open("data/pickles/search_dtms.pickle","rb")
    search_dtms_pdfs = pickle.load(pickle_in)

    #Dtms for express comments
    pickle_in = open("data/pickles/search_dtms_express.pickle","rb")
    search_dtms_express = pickle.load(pickle_in)

    #Stack
    search_dtms = sparse.vstack((search_dtms_pdfs, search_dtms_express))

    #===========================================================================
    #Index keeping track of comment ids
    #===========================================================================
    print('Reading in comment indices...')

    #Get search index
    pickle_in = open("data/pickles/search_index.pickle","rb")
    search_index_pdfs = pickle.load(pickle_in)

    #Keep needed columns
    search_index_pdfs = search_index_pdfs[['doc_id', 'term_length', 'page_no']]

    #Get comment ids for express comments
    pickle_in = open("data/pickles/search_index_express.pickle","rb")
    search_index_express = pickle.load(pickle_in)
    search_index_express.columns = ['doc_id', 'term_length']

    #Add dummy page number
    search_index_express['page_no'] = 0

    #Stack
    search_index = search_index_pdfs.append(search_index_express)

    #Reset index
    search_index = search_index.reset_index()

    print('Selecting comment corpus to score...')

    #Load in comment metadata
    pickle_in = open("data/pickles/comment_universe.pickle","rb")
    comment_meta = pickle.load(pickle_in)

    #Merge weighted count to search index
    search_index = search_index.merge(comment_meta, on = 'doc_id', how = 'left')

    #drop comments not in defined universe
    keep_comments = search_index.loc[~(search_index.sweight.isna()),].index

    #drop from search dtms and search index
    search_dtms = search_dtms[keep_comments, :]
    search_index = search_index.loc[keep_comments, ]

    #===========================================================================
    #Set up bm25 algorithm
    #===========================================================================
    print('Building bm25...')

    #Build comment corpus
    corpus = search.Comments(search_dtms, search_index['sweight'].values)
    corpus.set_ld(search_index['term_length'].values)
    print(corpus)

    #Build query corpus
    queries = search.Queries(query_boolean)
    print(queries)

    #Build algorithm with parameters
    bm25_model = search.bm25(k = 0.9, b = 0.4).build(corpus, queries)
    print(bm25_model)

    #===========================================================================
    #Parallelize
    #===========================================================================
    print('Preparing batch jobs...')

    #Define row indices of jobs to send to cores
    list_length = search_dtms.shape[0]
    inc = 45000
    it_seq = range(0, list_length+1, inc)

    #Create tuples for each batch
    tuple_seq = helpers.create_list_sequence(list_length, inc,
    include_last = True)

    print('%d job batches prepared.'%(len(tuple_seq)))

    #Send jobs to cores
    seq_result = Parallel(n_jobs=len(tuple_seq),
    backend="multiprocessing")(delayed(distribute_bm25)(
    dtm = search_dtms[this_seq[0]:this_seq[1], :],
    index = search_index.iloc[this_seq[0]:this_seq[1], :],
    bm25 = bm25_model) for this_seq in tuple_seq)

    #===========================================================================
    #Combine results
    #===========================================================================
    print('Combining results...')

    #Parse result
    mats = [x[0] for x in seq_result]
    indexes = [x[1] for x in seq_result]

    #Combine
    master_mat = mats[0]
    master_index = indexes[0]
    for j in range(1, len(mats)):
        master_mat = sparse.vstack((master_mat, mats[j]))
        master_index = master_index.append(indexes[j])

    #Combine
    bm25_df = pd.DataFrame(master_mat.todense())
    bm25_df['doc_id'] = master_index['doc_id'].values
    bm25_df['page_no'] = master_index['page_no'].values

    #===========================================================================
    #Page level scores
    #===========================================================================
    print('Creating page-level scores...')

    #Info columns
    info_columns = query_dtm['_query_no_'].values

    #Name columns based on original query order
    bm25_df.columns = np.append(info_columns, ['doc_id', 'page_no'])

    #Merge to scores
    page_merge = bm25_df.merge(search_index, on = ['doc_id', 'page_no'],
    how = 'left')

    #Max score per page
    page_merge['max_raw'] = page_merge[info_columns].max(axis = 1)

    #Get column index of max score
    page_merge['max_index'] = page_merge[info_columns].idxmax(axis = 1)

    #Write out page-level results
    with open('data/pickles/information_scores_page_level.pickle', 'wb') as pk:
        pickle.dump(page_merge,
        pk, protocol=pickle.HIGHEST_PROTOCOL)

    #===========================================================================
    #Comment-level scores
    #===========================================================================

    print('Creating comment-level scores...')

    #Collapse over pages
    bm25_sum = bm25_df.iloc[:,:80].groupby('doc_id').sum().reset_index()

    #Get page count
    bm25_count = bm25_df.groupby('doc_id').count().reset_index()
    page_df = bm25_count[['doc_id', '27.']].copy()
    page_df.rename(columns = {'27.': 'page_count'}, inplace = True)

    #Merge in citation info
    filer_merge = bm25_sum.merge(comment_meta, on = 'doc_id', how = 'inner')
    filer_merge = filer_merge.merge(page_df, on = 'doc_id', how = 'inner')

    #Add index
    filer_merge.set_index('doc_id', inplace = True)

    #Sum raw scores
    filer_merge['raw_score'] = filer_merge[info_columns].sum(axis = 1)

    #Max raw score
    filer_merge['max_raw'] = filer_merge[info_columns].max(axis = 1)

    #File to write out
    outfile = filer_merge.iloc[:, -8:]

    #Flag express comments
    outfile.reset_index(inplace = True)
    pdfs = [x for x in outfile['doc_id'].values if 'pdf' in x]
    outfile['express'] = 1
    outfile.loc[(outfile.doc_id.isin(pdfs)), 'express'] = 0

    #Log of raw score
    outfile['log_raw_score'] = [math.log(x + 1) for x in \
    outfile['raw_score'].values]

    #Identify identical comment content via identical scores
    score_dups = outfile.groupby('raw_score').agg({'sweight': 'sum',
                         'cited': 'max'}).reset_index()
    score_dups.columns = ['raw_score', 'score_dup_count', 'score_dup_cited']

    #Exclude scores of 0
    score_dups = score_dups.loc[(score_dups.raw_score > 0),]

    #Merge to rank df
    outfile = outfile.merge(score_dups, on = 'raw_score', how = 'left')
    outfile.loc[(outfile.score_dup_count.isna()), 'score_dup_count'] = 0
    outfile.loc[(outfile.score_dup_cited.isna()), 'score_dup_cited'] = 0

    #Tag one near-duplicate as cited
    outfile.loc[((outfile.cited == 0) & \
    (outfile.score_dup_cited == 1)), 'cited'] = 1

    #Write out
    with open('data/pickles/information_scores.pickle', 'wb') as pk:
        pickle.dump(outfile.iloc[:,:-1],
        pk, protocol=pickle.HIGHEST_PROTOCOL)

    print('Done.')

if __name__ == '__main__':
    main()
