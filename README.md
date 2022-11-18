## Replication Code for "Do fake online comments pose a threat to regulatory policymaking?"

The code in this repository replicates the analyses in the main body of "Do fake online comments pose a threat to regulatory policymaking?"

## Requirements

The python code requires the dependencies in the environment.yml file.
To replicate this environment, install the
[open-source anaconda distribution](https://www.anaconda.com/distribution/),
navigate to the top level of the replication directory, and run
```
conda env create --name fcc -f=environments/fcc_environment.yml
```

## Directory structure
`process_comments`: contains the code to select the comment corpus, compute the relevance ranking for each comment, select relevant comments according to the ranking, and run doc2vec on the relevant comments. Note: `run_dbow` will produce slightly different document vectors with each run since doc2vec is [not deterministic](https://github.com/RaRe-Technologies/gensim/wiki/Recipes-&-FAQ#q11-ive-trained-my-word2vec--doc2vec--etc-model-repeatedly-using-the-exact-same-text-corpus-but-the-vectors-are-different-each-time-is-there-a-bug-or-have-i-made-a-mistake-2vec-training-non-determinism).

`main_body`: contains the code to produce the tables and figures in the main body of the paper.

`search_utils`: contains the software to execute bm25 with weighting.

`tests`: unit tests for `search_utils`.
