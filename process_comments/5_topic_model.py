from collections import defaultdict
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from math import exp
import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

def main():

    #===========================================================================
    #Join metadata to document vectors
    #===========================================================================
    print('Joining document vectors to metadata...')

    #Read in model
    final_model = Doc2Vec.load('data/models/doc2vec_dbow_comments.model')

    #Compute vectors
    doc_vectors = final_model.docvecs.vectors_docs

    #Initialize matrix of doc vectors
    doc_mat = np.zeros((len(doc_vectors), len(doc_vectors[0])))
    for i, this_vec in enumerate(doc_vectors):
        doc_mat[i, :] = this_vec

    #File index lookup
    file_df = pd.read_csv('data/csvs/filename_lookup.csv')

    #Replace .txt
    file_df['comment_id'] = file_df['filename'].apply(lambda \
    row: row.replace('.txt', ''))

    #Rename column
    file_df.rename(columns = {'index': 'file_index'}, inplace = True)

    #Comment metadata
    comment_data = pd.read_csv('data/csvs/comment_segmentation.csv')

    #Add comment id to match file names
    comment_data['doc_name'] = comment_data['doc_id'].apply(lambda row: \
    row.replace('.pdf', ''))
    comment_data['comment_id'] = comment_data[['doc_name', 'page_no']].apply(lambda \
    row: '_'.join(row.values.astype(str)), axis=1)

    #Express comments don't need page numbers
    comment_data.loc[(comment_data.express == 1), 'comment_id'] = \
    comment_data.loc[(comment_data.express == 1), 'doc_name']

    #Merge index
    comment_data = comment_data.merge(file_df, left_on = 'comment_id',
    right_on = 'comment_id', how = 'left')

    #Get citation vector
    comment_data.loc[(comment_data.cited.isna()), 'cited'] = 0

    #sort by file index
    comment_data.sort_values('file_index', inplace = True)

    #Append vectors
    doc_vectors = pd.DataFrame(doc_mat)
    doc_vectors['comment_id'] = file_df['comment_id']
    comment_data = pd.merge(comment_data[['comment_id', 'interest_group',
    'business_group', 'cited']], doc_vectors, on = 'comment_id')

    #document vector data
    doc_mat = np.array(comment_data.iloc[:,4:])

    #===========================================================================
    #Topic detection
    #===========================================================================
    print('Running topic model...')

    #1. Run k-means on document vectors
    nclust = 5
    this_kmeans = KMeans(n_clusters = nclust, random_state = 123).fit(doc_mat)

    #Check convergence
    print('Converged in %s steps.'%(this_kmeans.n_iter_))

    #Get cluster centroids
    kcentroids = this_kmeans.cluster_centers_

    #2. Calculate dot product between each document and each cluster center.
    #Normalize dot products into proportions using softmax over all cluster centers.
    doc_centroids = pd.DataFrame(doc_mat@kcentroids.T)
    centroids = range(nclust)
    for c in centroids:
        doc_centroids['c_' + str(c)] = 0
        for i, row in doc_centroids.iterrows():
            d = np.sum([exp(row[x]) for x in centroids])
            p = exp(row[c]) / d
            doc_centroids.loc[i, 'c_' + str(c)] = p

    #Check sums
    print('%s documents scored.'%(doc_centroids.iloc[:,\
    nclust:].sum(axis = 1).sum()))

    #3. Calculate dot product between word output vector and each cluster center.
    #Normalize into probability using softmax over all word output vectors.
    word_vectors = final_model.trainables.syn1neg

    #Compute score vectors
    word_centroids = pd.DataFrame(word_vectors@kcentroids.T)
    for c in centroids:
        d = np.sum([exp(x) for x in word_centroids.iloc[:, c]])
        word_centroids['c_' + str(c)] = 0
        for i, row in word_centroids.iterrows():
            p = exp(row[c]) / d
            word_centroids.loc[i, 'c_' + str(c)] = p

    #Check sums
    word_centroids.iloc[:,nclust:].sum(axis = 0)

    #Add word labels
    word_centroids['word'] = final_model.wv.index2word

    #Look at top 10 words per cluster
    word_dict = defaultdict()
    for c in centroids:
        idx = word_centroids.sort_values('c_' + str(c), ascending = False).index[:19]
        word_dict['c_' + str(c)] = word_centroids.loc[idx]['word']

    #append to comment data
    comment_vectors = pd.concat((comment_data, doc_centroids), axis = 1)

    #Force unique cluster membership
    comment_vectors['cluster'] = comment_vectors[['c_0', 'c_1', 'c_2', 'c_3',
    'c_4']].idxmax(axis = 1)

    #Save results
    comment_vectors.to_csv('data/csvs/topic_vectors.csv', index = False)
    with open('data/pickles/topic_words.pickle', 'wb') as pk:
        pickle.dump(word_dict, pk, protocol=pickle.HIGHEST_PROTOCOL)

    print('Done.')

if __name__ == '__main__':
    main()
