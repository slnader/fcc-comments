from joblib import Parallel, delayed
import os
import pandas as pd
import pickle
import psycopg2

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

    print('Querying FCC database...')

    #Get doc filer information
    cite_query = """
    with cites as(
      select
      submission_id || '_' || document_id || '.pdf' as doc_id,
      1 as cited
      from docs_cited
      group by 1,2
    ),
    cited as(
      select distinct doc_id, cited
      from(
          select
          duplicate_document_id as doc_id, cited
          from cites a
          inner join near_duplicates b
          on a.doc_id = b.target_document_id
          union all
          select
          target_document_id as doc_id, cited
          from cites a
          inner join  b
          on a.doc_id = b.duplicate_document_id
          union all
          select doc_id, cited
          from cites
          ) base
    ),
    exparte_filers as(
          select
          filer_name
          from in_person_exparte a
          inner join filers b
          on a.submission_id = b.submission_id
          group by 1
    ),
    interest_group_filers as(
          select
          filer_name
          from interest_groups a
          inner join filers b
          on a.submission_id = b.submission_id
          group by 1
    )
    select
    coalesce(g.target_document_id,
    a.submission_id || '_' || a.document_id || '.pdf') as doc_id,
    max(coalesce(c.cited, 0)) cited,
    max(case when e.filer_name is not null then 1
    when d.submission_id in (select submission_id from in_person_exparte)
    then 1 else 0 end) in_person,
    max(case when f.submission_id is not null then 1 else 0 end) interest_group,
    max(coalesce(f.business, 0)) business_group,
    count(distinct a.submission_id || '_' || a.document_id || '.pdf') as sweight
    from documents a
    left join filers b
    on a.submission_id = b.submission_id
    left join cited c
    on a.submission_id || '_' || a.document_id || '.pdf' = c.doc_id
    inner join submissions d
    on a.submission_id = d.submission_id
    left join exparte_filers e
    on b.filer_name = e.filer_name
    left join interest_groups f
    on a.submission_id = f.submission_id
    left join exact_duplicates g
    on a.submission_id || '_' || a.document_id || '.pdf' = g.duplicate_document_id
    where a.file_extension in ('pdf', 'txt', 'rtf', 'text', 'docx', 'doc')
    and a.download_status = 200
    and substring(a.submission_id,1,1) != '0'
    and a.document_name not in ('DOC-347927A1.pdf', 'FCC-17-166A6.pdf',
    'FCC-15-24A1_Rcd.pdf', 'FCC-17-166A3.pdf',
    'FCC-17-166A1_Rcd.pdf', 'FCC-17-166A1.pdf', 'FCC-17-60A1.pdf',
    'FCC-17-60A2.pdf',
    'FCC-17-60A4.pdf', 'FCC-17-166A4.pdf', 'FCC-17-166A2.pdf',
    'FCC-17-60A1_Rcd.pdf',
    'FCC-17-60A3.pdf', 'FCC-17-166A5.pdf', 'DOC-344614A1.pdf')
    and coalesce(b.filer_name, '') not in ('Wireless Telecommunications Bureau',
    'Wireline Competition Bureau') and
    a.submission_id || '_' || a.document_id || '.pdf' not in
    ('10511158172956_349167aae58b2a9f8d8c2a7234154cae740c5de1d879da62fc8f79fdf853873d.pdf',
     '107171518608774_418c1975fecbb513341ab950c9a70564b6538f1036644c01b3e7c0ef7b550b1d.pdf',
     '10830277415067_2467bf2d96704d95949bd2f7a7f6e4dee3646836061d82721657c05b338863ed.pdf')
    and d.date_received <= '2017-12-14'
    group by 1
    union all
    select
    a.comment_id,
    max(coalesce(cited, 0)) as cited,
    max(case when e.filer_name is not null
    and e.filer_name not in (select filer_name from interest_group_filers)
    then 1
    when b.submission_id in (select submission_id from in_person_exparte)
    then 1 else 0 end) in_person,
    max(case when f.submission_id is not null then 1 else 0 end) interest_group,
    max(coalesce(f.business, 0)) business_group,
    count(distinct b.submission_id) as sweight
    from comments a
    inner join submissions b
    on a.comment_id = b.comment_id
    left join cited c
    on a.comment_id = c.doc_id
    left join filers d
    on b.submission_id = d.submission_id
    left join exparte_filers e
    on d.filer_name = e.filer_name
    left join interest_groups f
    on b.submission_id = f.submission_id
    where a.comment_text != '' and a.comment_text is not null
    and b.date_received <= '2017-12-14'
    group by 1;
    """

    cur.execute(cite_query)
    cite_data = cur.fetchall()
    doc_cited_df = pd.DataFrame(cite_data,
                             columns = ['doc_id', 'cited', 'in_person',
                             'interest_group', 'business_group', 'sweight'])
    #save
    with open('data/pickles/comment_universe.pickle', 'wb') as pk:
      pickle.dump(doc_cited_df,
      pk, protocol=pickle.HIGHEST_PROTOCOL)

    #Close connection
    conn.close()

    print('Done.')

if __name__ == '__main__':
    main()
