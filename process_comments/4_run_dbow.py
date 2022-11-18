from collections import defaultdict
import gensim
from joblib import Parallel, delayed
from nltk.stem import PorterStemmer
import numpy as np
import os
import pandas as pd
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold
import smart_open

random.seed(123)

def read_corpus(file_path, file_list, stop_words):
    """
    Reads in list of files in file path and creates corpus
    Keyword args
    file_path = string with path to directory where files live
    file_list = list
    stop_words = list of stop words
    """
    porter_stemmer = PorterStemmer()

    for i, this_file in enumerate(file_list):
        f = open(os.path.join(os.path.expanduser(file_path), this_file), "r")
        comment_text = f.read()
        #Tokenize
        file_tokens = gensim.utils.simple_preprocess(comment_text)
        #Remove stop words
        file_tokens = [x for x in file_tokens if x not in stop_words]
        #Stem
        file_tokens = [porter_stemmer.stem(x) for x in file_tokens]
        yield gensim.models.doc2vec.TaggedDocument(file_tokens, [i])

def eval_model(doc_corpus, run_dict, comment_triplets):
    """
    Evaluates error rate given model and data sets passed through run_dict
    Keyword args
    run_dict = dictionary with keys: hyperparameters, train_idx, test_idx, test_names
    doc_corpus = list of gensim tagged documents
    comment_triplets = pandas dataframe of comment triplets for evaluation

    Returns
    error rate
    """
    #Build train corpus
    train_corpus = [doc_corpus[i] for i in run_dict['train_idx']]

    if run_dict['negative'] == 0:
        #Initialize model
        this_model = gensim.models.doc2vec.Doc2Vec(dm = 0,
        workers = 1, seed = 123, dbow_words = 1,
        vector_size = run_dict['size'], epochs = 40,
        min_count = run_dict['min_count'],
        window = run_dict['window'],
        negative = run_dict['negative'],
        hs = 1)
    else:
        #Initialize model
        this_model = gensim.models.doc2vec.Doc2Vec(dm = 0,
        workers = 1, seed = 123, dbow_words = 1,
        vector_size = run_dict['size'], epochs = 40,
        min_count = run_dict['min_count'],
        negative = run_dict['negative'],
        window = run_dict['window'])

    #Build vocab
    this_model.build_vocab(train_corpus)

    #Train model
    this_model.train(train_corpus, total_examples = this_model.corpus_count,
    epochs = this_model.epochs)

    #Compute doc vectors
    doc_vectors = defaultdict()
    for i in range(len(run_dict['test_idx'])):
        this_vector = this_model.infer_vector(doc_corpus[run_dict['test_idx'][i]].words)
        doc_vectors[run_dict['test_names'][i]] = this_vector

    #Grab test triplets
    test_triplets = comment_triplets.iloc[run_dict['test_triplet_idx']]

    #Compute similarity between comment triplets
    triplet_results = []
    for index, row in test_triplets.iterrows():
        triplet_docs = row.values
        pw_sim = cosine_similarity((doc_vectors[triplet_docs[0]],
        doc_vectors[triplet_docs[1]]))[0][1]
        pw_diff = cosine_similarity((doc_vectors[triplet_docs[0]],
        doc_vectors[triplet_docs[2]]))[0][1]
        triplet_results.append((pw_sim, pw_diff))

    #Convert to df
    triplet_df = pd.DataFrame(triplet_results, columns = ['score_sim',
    'score_diff'])

    #Count errors
    triplet_df['error'] = triplet_df[['score_sim', 'score_diff']].apply(lambda \
    row: int(row[0] < row[1]), axis=1)

    #Error rate
    error_rate = triplet_df['error'].mean()

    return (run_dict['min_count'], run_dict['negative'], run_dict['window'],
    run_dict['size'], error_rate)

def main():

    print('Preparing comment corpus for modeling...')

    #Read in comment pairs for cross-validation
    comment_triplets = pd.read_csv('data/csvs/comment_triplets.csv')

    #Read in stop words
    stop_words = pd.read_csv('data/csvs/stop_words.csv')
    stop_words = list(stop_words['word'].values)
    stop_words = stop_words + ['title', 'section', 'i', 'ii', 'iii', 'iv',
    'paragraph', 'net', 'neutrality']

    #Read entire corpus into memory in a random order
    file_path = 'data/comment_segmentation/'
    file_list = os.listdir(os.path.expanduser(file_path))
    random.shuffle(file_list)
    doc_corpus = list(read_corpus(file_path, file_list, stop_words))

    #Write out file index lookup
    file_df = pd.DataFrame(list(zip(file_list, range(len(file_list)))),
    columns = ['filename', 'file_index'])
    file_df.to_csv('data/csvs/filename_lookup.csv', index = False)

    #Cross-validation splits
    nsplits = 5
    kf = KFold(n_splits = nsplits, shuffle = True, random_state = 123)
    kf_split = kf.split(comment_triplets)

    #Parameters to search over
    min_counts = [0, 5, 10, 15, 20]
    negative = [0, 5, 10, 15, 20]
    windows = [5, 8, 10, 12]
    sizes = [50, 100]

    #Dictionary for model runs
    model_runs = dict.fromkeys(list(range(0, nsplits*len(min_counts))))

    print('Training and validating models...')

    #Build sets and hyperparameters for modeling
    k = 0
    #K folds to cross-validate over
    for train_index, test_index in kf_split:
        #Get triplets in fold
        X_train, X_test = comment_triplets.iloc[train_index], \
        comment_triplets.iloc[test_index]

        #Get individual docs from triplets
        train_docs = X_train.melt()['value'].values
        test_docs = X_test.melt()['value'].values

        #Extract doc file indices
        train_idx = [i for i in range(len(file_list)) if \
        file_list[i].replace('.txt', '') in train_docs]
        test_idx = [i for i in range(len(file_list)) if \
        file_list[i].replace('.txt', '') in test_docs]
        test_names = [file_list[i].replace('.txt', '')
        for i in range(len(file_list))
        if file_list[i].replace('.txt', '') in test_docs]

        for this_size in sizes:
            for this_window in windows:
                for this_count in min_counts:
                    for this_negative in negative:
                        #Add to model run dict
                        model_runs[k] = {
                        'size' : this_size,
                        'window': this_window,
                        'min_count': this_count,
                        'negative': this_negative,
                        'train_idx': train_idx,
                        'test_idx': test_idx,
                        'test_names': test_names,
                        'test_triplet_idx': test_index}
                        k+=1

    #Send runs to cores
    eval_results = Parallel(n_jobs=np.min([80, len(model_runs)]),
    backend="multiprocessing")(delayed(eval_model)(
    doc_corpus = doc_corpus,
    run_dict = model_runs[i],
    comment_triplets = comment_triplets) for i in range(len(model_runs)))

    #Collate results
    eval_df = pd.DataFrame(eval_results,
    columns = ['min_count', 'negative', 'window', 'size', 'error_rate'])
    average_result = eval_df.groupby(['min_count', 'negative',
    'window', 'size']).mean().reset_index()

    #Write out results
    average_result.to_csv('data/csvs/error_rates_dbow.csv')

    #Pick window with lowest cross-validated error
    min_index = average_result['error_rate'].idxmin(axis = 0)
    final_count = average_result.iloc[min_index]['min_count']
    final_negative = average_result.iloc[min_index]['negative']
    final_window = average_result.iloc[min_index]['window']
    final_size = average_result.iloc[min_index]['size']

    print('Training final model...')

    #Train final model on all data with chosen count parameter
    if final_negative == 0:
        #Initialize model
        final_model = gensim.models.doc2vec.Doc2Vec(dm = 0,
        workers = 1, seed = 123, dbow_words = 1,
        vector_size = final_size, epochs = 40,
        min_count = final_count,
        window = final_window,
        negative = final_negative,
        hs = 1)
    else:
        #Initialize model
        final_model = gensim.models.doc2vec.Doc2Vec(dm = 0,
        workers = 1, seed = 123, dbow_words = 1,
        vector_size = final_size, epochs = 40,
        min_count = final_count,
        negative = final_negative,
        window = final_window)

    #Build vocab
    final_model.build_vocab(doc_corpus)

    #Train model
    final_model.train(doc_corpus, total_examples = final_model.corpus_count,
    epochs = final_model.epochs)

    #Save model
    final_model.save('data/models/doc2vec_dbow_comments.model')

    print('Done.')

if __name__ == '__main__':
    main()
