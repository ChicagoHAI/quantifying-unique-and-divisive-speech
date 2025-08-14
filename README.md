# Quantifying the Uniqueness and Divisiveness of Presidential Discourse

Data and code for the paper, [Quantifying the Uniqueness and Divisiveness of Presidential Discourse](https://academic.oup.com/pnasnexus/article/3/10/pgae431/7814873) ([preprint](https://arxiv.org/abs/2401.01405)).



## Citations
```
@article{10.1093/pnasnexus/pgae431,
    author = {Zhou, Karen and Meitus, Alexander A and Chase, Milo and Wang, Grace and Mykland, Anne and Howell, William and Tan, Chenhao},
    title = {Quantifying the uniqueness and divisiveness of presidential discourse},
    journal = {PNAS Nexus},
    volume = {3},
    number = {10},
    pages = {pgae431},
    year = {2024},
    month = {10},
    issn = {2752-6542},
    doi = {10.1093/pnasnexus/pgae431},
    url = {https://doi.org/10.1093/pnasnexus/pgae431},
    eprint = {https://academic.oup.com/pnasnexus/article-pdf/3/10/pgae431/59961584/pgae431.pdf},
}

```


If you use the data please also cite [the American Presidency Project](https://www.presidency.ucsb.edu/documents):
```
@article{woolley1999american,
  title={The American presidency project},
  author={Woolley, [dataset] John T and Peters, Gerhard},
  journal={Santa Barbara, CA. Available from World Wide Web: http://www. presidency. ucsb. edu/ws},
  year={1999}
}
```

## Data
We share our sentence-delimited corpora of presidential debates, State of the Union addresses, and campaign speeches that we scrape and process from [the American Presidency Project](https://www.presidency.ucsb.edu/documents).

Please see [data/README.md](data/README.md) for more details.

## Divisive Word Lexicon
We define language as "divisive" if it intends to impugn and delegitimize the speaker's target, e.g., by attacking their intelligence, integrity, or intentions. Such labels are expressly designed to put the target on defense and accentuate differences and distance between parties.

The methodology of lexicon construction is described in our paper. In this repo, we provide two files:
- `majority-words.txt` - contains the 178 terms that ≥3 out of 4 annotators agree are divisive. This set is used for analysis in the paper.
- `unanimous-words.txt` - contains the 123 terms that all 4 annotators agree are divisive. 

## Methods & Analysis

### Uniqueness score

Model training: see [models/README.md](models/README.md)

Get scores from model: run `analysis/score_uniqueness.py`. See `analysis/run_score_uniqueness.sh` for an example of how to run this script.

Plot scores: run `analysis/plot_uniqueness.py`. See `analysis/run_plot_uniqueness.sh` for an example of how to run this script.

### Divisive lexicon usage
Run `analysis/plot_divisive_lexicon.py`. See `analysis/run_plot_lexicon.sh` for an example of how to run this script.


### FW-overlap analysis
Run `analysis/plot_fw_overlap.py`. See `analysis/run_fw_overlap.sh` for an example of how to run this script.



