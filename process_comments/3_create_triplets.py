import pandas as pd
import random
from sklearn.metrics.pairwise import cosine_similarity

def main():

    #===========================================================================
    #Read in metadata
    #===========================================================================
    comment_data = pd.read_csv('data/csvs/comment_segmentation.csv')

    #Add comment id to match file names
    comment_data['doc_name'] = comment_data['doc_id'].apply(lambda row: \
    row.replace('.pdf', ''))
    comment_data['comment_id'] = comment_data[['doc_name', 'page_no']].apply(lambda \
    row: '_'.join(row.values.astype(str)), axis=1)

    #Express comments don't need page numbers
    comment_data.loc[(comment_data.express == 1), 'comment_id'] = \
    comment_data.loc[(comment_data.express == 1), 'doc_name']

    #===========================================================================
    #Create comment triplets for validation
    #===========================================================================
    print('Creating comment triplets for validation...')

    #Compute similarty between pages on information scores
    doc_sim = pd.DataFrame(cosine_similarity(comment_data[comment_data.columns[:79]]),
    columns = [str(x) for x in range(len(comment_data))])

    #Reset index
    doc_sim.reset_index(inplace = True)

    #Initialize
    random.seed(123)
    doc_triplets = []

    while len(doc_sim) > 2:

        #Get random doc
        this_doc = random.choice(doc_sim['index'].values)

        #Get most similar doc
        most_sim = doc_sim.loc[(doc_sim['index'] == this_doc),
        ~doc_sim.columns.isin(['index', str(this_doc)])].idxmax(axis = 1)

        #Get least similar doc
        least_sim = doc_sim.loc[(doc_sim['index'] == this_doc),
        ~doc_sim.columns.isin(['index', str(this_doc)])].idxmin(axis = 1)

        #Selected list
        selected_docs = [this_doc, int(most_sim), int(least_sim)]
        selected_ids = comment_data.iloc[selected_docs]['comment_id'].values

        #Add triplet to list
        doc_triplets.append(selected_ids)

        #Remove docs from pool
        doc_sim = doc_sim.loc[~(doc_sim['index'].isin(selected_docs)),]
        doc_sim.drop(columns = [str(x) for x in selected_docs], inplace = True)

    #Convert to dataframe
    doc_df = pd.DataFrame(doc_triplets, columns = ['sim1', 'sim2', 'diff'])

    #Write out doc triplets
    doc_df.to_csv('data/csvs/comment_triplets.csv', index = False)

    print('Done.')

if __name__ == '__main__':
    main()
