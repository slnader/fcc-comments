from io import StringIO
from joblib import Parallel, delayed
import numpy as np
import os
import pandas as pd
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import pickle
import psycopg2
import re
import regex as re2
from search_utils import helpers
import subprocess

def set_interpreter():
    """
    Sets up the pdf interpreter to read in pdf as text
    """
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    return { 'retstr': retstr, 'device': device, 'interpreter': interpreter }

def process_pdf(get_files, metadata):
    """
    Returns pdf pages in plain text format
    Args:
    get_files: list of filenames to retrieve
    metadata: pd dataframe of metatdata including page numbers to retrieve
    """
    #Footnote cite - punctuation followed by a number no whitespace
    footnote_cite = re2.compile('(?<=[^\P{P}-(]|[a-z])\d{1,}')

    #Identity query numbers from FCC's proposal
    query_number = re2.compile('(?<=[\n]|[a-z])\d{1,}(?=\.)')

    #Set points for pdf set_interpreter
    password = ""
    maxpages = 0
    caching = True
    pagenos=set()
    page_counter = 0
    k = 0
    errors = []

    #Directory where pdfs are stored
    doc_dir = 'data/docs/'

    for this_doc in get_files:

        #Open pdf
        try:
            fp = open(os.path.join(doc_dir, this_doc), 'rb')
        except:
            errors.append((this_name, 'pdf open'))
            continue

        #Initialize interpreter
        si = set_interpreter()
        retstr = si['retstr']
        device = si['device']
        interpreter = si['interpreter']

        #Initialize page list
        page_list = []

        #Get page generator
        try:
            page_generator = PDFPage.get_pages(fp, pagenos, maxpages=maxpages,
            password=password,caching=caching, check_extractable=True)
        except:
            #Cleanup
            fp.close()
            device.close()
            retstr.close()
            errors.append((this_name, 'page generator'))
            continue

        try:
            #Loop through pages and process
            for pageNumber, page in enumerate(page_generator):

                #Process page
                try:
                    interpreter.process_page(page)
                    page_list.append(retstr.getvalue())

                    #Reset interpreter
                    si = set_interpreter()
                    retstr = si['retstr']
                    device = si['device']
                    interpreter = si['interpreter']

                except:
                    #Reset interpreter
                    si = set_interpreter()
                    retstr = si['retstr']
                    device = si['device']
                    interpreter = si['interpreter']

                    continue
        except:
            #Cleanup
            fp.close()
            device.close()
            retstr.close()
            errors.append((this_name, 'page generator iterate'))
            continue

        if len(page_list) > 0:

            #Identify informative pages
            pages = metadata.loc[(metadata.doc_id == this_doc),
            'page_no'].values

            for this_page in pages:
                if this_page > len(page_list):
                    continue
                else:
                    #Extract page text
                    this_text = page_list[this_page]

                    #Get citations
                    this_cite = footnote_cite.findall(this_text)
                    this_queries = query_number.findall(this_text)

                    #Get punctuation
                    cite_punct = [re.sub('\d', '', x) for x in this_cite]
                    query_punct = [re.sub('\d', '', x) for x in this_queries]

                    #Remove citations from text
                    for j in range(len(this_cite)):
                        this_text = re.sub(re.escape(this_cite[j]),
                        re.escape(cite_punct[j]), this_text)
                    for j in range(len(this_queries)):
                        this_text = re.sub(re.escape(this_queries[j]),
                        re.escape(query_punct[j]), this_text)

                    #Remove line breaks
                    this_text = re.sub('\n', ' ', this_text)

                    #Replace fancy quotes with plain text quotes
                    this_text = re.sub('“|”', '"', this_text)

                    #Replace fancy quotes with plain text quotes
                    this_text = re.sub('’', "'", this_text)

                    #Write to file
                    f = open('data/comment_segmentation/' + \
                    re.sub('.pdf', '', this_doc) \
                    + '_' + str(this_page) + '.txt', 'a')
                    f.write(this_text)
                    f.close()

    return errors

def main():

    with open('../.pw', 'r') as my_file:
        pw_string = my_file.read()

    #Get credentials
    pw_string = pw_string.replace('\n','').split(',')
    db_user = pw_string[0]
    db_pw = pw_string[1]

    #Establish connection
    conn = psycopg2.connect(dbname='fcc',
                            user=db_user,
                            password =db_pw,
                            host = 'localhost',
                            port = 5432)

    #Set autocommit to true
    conn.autocommit = True

    #Create cursor
    cur = conn.cursor()

    #Page-level score data
    pickle_in = open("data/pickles/information_scores_page_level.pickle","rb")
    page_df = pickle.load(pickle_in)

    #Info columns
    info_columns = page_df.columns[:79]

    print('Choosing most informative comment pages...')

    #===========================================================================
    #Subset to most informative pages (max-max)
    #===========================================================================
    #Get 75th percentile per page in cited docs
    score_cutoff =  np.percentile(page_df.loc[(page_df['cited'] == 1),
    'max_raw'].values, 75)

    #Get number of words per page in cited docs
    page_cutoff =  np.percentile(page_df.loc[(page_df['cited'] == 1),
    'term_length'].values, 50)

    #===========================================================================
    #Subset to informative comments
    #===========================================================================

    #Subset to comments above the minimum score cited
    info_comments = page_df.loc[((page_df.max_raw >= score_cutoff) &
    (page_df.term_length >= page_cutoff)), ].copy()

    #Number of unique pages
    len(info_comments)

    #Tag pdfs
    pdfs = [x for x in info_comments['doc_id'].values if 'pdf' in x]
    info_comments['express'] = 1
    info_comments.loc[(info_comments.doc_id.isin(pdfs)), 'express'] = 0
    info_comments.groupby(['express']).count()

    #Number of unique comments
    unique_comments = info_comments.drop_duplicates('doc_id')
    len(unique_comments.loc[(unique_comments.cited == 1),])/len(unique_comments)

    #Write out
    info_comments.to_csv('data/csvs/comment_segmentation.csv', index = False)

    #===========================================================================
    #Get express comment text
    #===========================================================================
    print('Saving express comment text...')

    #Retrieve express comments
    comment_string = ','.join(["'" + x + "'" for x in
    info_comments.loc[(info_comments.express == 1), 'doc_id'].values])

    #Retrieve text
    comment_query = """select comment_id, comment_text from comments
    where comment_id in (""" + comment_string + """);"""
    cur.execute(comment_query)
    comment_data = cur.fetchall()

    #Clear directory
    comment_path = 'data/comment_segmentation/'
    subprocess.run(['rm', '-r', comment_path])
    subprocess.run(['mkdir', comment_path])

    #Add files
    for this_comment in comment_data:
        f = open(comment_path + this_comment[0] + '.txt', 'a')
        f.write(this_comment[1])
        f.close()

    #===========================================================================
    #Get pdf comment text
    #===========================================================================
    print('Saving standard comment text...')

    #Get pdf filenames
    get_files = info_comments.loc[(info_comments.express == 0),
    'doc_id'].drop_duplicates().values

    #Define row indices of jobs to send to cores
    list_length = len(get_files)
    inc = 10
    it_seq= range(0, list_length, inc)

    #Create tuples for each batch
    tuple_seq = helpers.create_list_sequence(list_length, inc)

    #Send jobs to cores
    seq_result = Parallel(n_jobs=len(tuple_seq))(delayed(process_pdf)(
    get_files = get_files[range(this_seq[0], this_seq[1])],
    metadata = info_comments) for this_seq in tuple_seq)

    #Get errors
    errors = sum(seq_result, [])

    if len(errors) > 0:
        #Write out errors
        error_df = pd.DataFrame(list(errors),
        columns = ['filename', 'error'])

        error_df.to_csv('data/qa/comment_segmentation_errors.csv',
        index = False)

    print('Done.')

if __name__ == '__main__':
    main()
