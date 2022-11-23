## FCC relational database

The core components of the database include a table for submission metadata, a table for attachment metadata, a table for filer metadata, and a table that contains comment text if submitted in express format. In addition to these core tables, there are several derived tables specific to the analyses in the paper, including which submissions and attachments were cited in the final order. The keys fit together as shown in the diagram below.

<img src="db_diagram.jpg" alt="drawing" width="600"/>

### comments
plain text comments associated with submissions

| column      | type | description |
| ----------- | ----------- | ----------- |
| comment_id      | character varying(64)       | unique id for plain text comment |
comment_text | text | raw text of plain text comment
row_id | integer | row sequence for plain text comments

### submissions
metadata for submissions

| column      | type | description |
| ----------- | ----------- | ----------- |
submission_id   | character varying(20)  | unique id for submission
submission_type | character varying(100) | type of submission (e.g., comment, reply, statement)
express_comment | numeric                | 1 if express comment
date_received   | date                   | date submission was received
contact_email   | character varying(255) | submitter email address
city            | character varying(255) | submitter city
address_line_1  | character varying(255) | submitter address line 1
address_line_2  | character varying(255) | submitter address line 2
state           | character varying(255) | submitter state
zip_code        | character varying(50)  | submitter zip
comment_id      | character varying(64)  | unique id for plain text comment

### filers
names of filers associated with submissions

| column      | type | description |
| ----------- | ----------- | ----------- |
submission_id | character varying(20)  | unique id for submission
filer_name    | character varying(250) | name of filer associated with submission

### documents
attachments associated with submissions

| column      | type | description |
| ----------- | ----------- | ----------- |
submission_id   | character varying(20) | unique id for submission
document_name   | text                  | filename of attachment
download_status | numeric               | status of attachment download
document_id     | character varying(64) | unique id for attachment
file_extension  | character varying(4)  | file extension for attachment

### filers_cited
citations from final order

| column      | type | description |
| ----------- | ----------- | ----------- |
point           | numeric                | paragraph number in final order
filer_name      | character varying(250) | name of cited filer
submission_type | character varying(12)  | type of submission as indicated in final order
page_numbers    | text[]                 | cited page numbers
cite_id         | integer                | unique id for citation
filer_id        | character varying(250) | id for cited filer

### docs_cited
attachments associated with cited submissions

| column      | type | description |
| ----------- | ----------- | ----------- |
cite_id       | numeric               | unique id for citation
submission_id | character varying(20) | unique id for submission
document_id   | character varying(64) | unique id for attachment


### near_duplicates
lookup table for comment near-duplicates

| column      | type | description |
| ----------- | ----------- | ----------- |
target_document_id    | unique id for target document
duplicate_document_id | unique id for duplicate of target document

### exact_duplicates
lookup table for comment exact duplicates

| column      | type | description |
| ----------- | ----------- | ----------- |
target_document_id    | character varying(100) | unique id for target document
duplicate_document_id | character varying(100) | unique id for duplicate of target document   

### in_person_exparte
submissions associated with ex parte meeting

| column      | type | description |
| ----------- | ----------- | ----------- |
submission_id   | character varying(20) | unique id for submission

### interest_groups
submissions associated with interest groups

| column      | type | description |
| ----------- | ----------- | ----------- |
submission_id | character varying(20) | unique id for submission
business      | numeric | 1 if business group, 0 otherwise
