# Data

| Corpus | # Speeches | # Sentences | Date range |
| -------- | -------- | -------- | -------- |
| Debates   | 35   | 35,096   | 1960-2020  |
| SOTU   | 246   | 69,630  | 1790-2022   |
| Campaign  | 640   | 83,038   | 1932-2020   |


For each data type in ["debates", "sotu", "campaign"], the corresponding directory contains the following files:
- `train.csv` - 
    - Columns:
        ```
        - speech_id: corresponding to `speech_metadata.jsonl`
        - sent_id: 0-indexed sentences in each speech
        - speaker
        - text: "<speaker name>: <text_orig>"
        - text_orig: original sentence as scraped from APP, without masking or speaker prefix
        - text_orig_masked:  original sentence as scraped from APP, without speaker prefix but replacing named entities with "<ENT>" token 
        - mentions_opponent: Y, N, or # (maybe)
        - party 
        - year
        ```
- `scored_data_<model_name>.csv` - train.csv with additional columns corresponding to uniqueness scores from <model_name>
    - Key additional columns:
        ```
        - sent_len: # tokens (words) in sentence
        - sent_uniq_bpc: the LLM-based uniqueness score
        - bpc: bits-per-character (bits-per-byte) of the sentence
        - loss: loss array as obtained from <model_name>
        - mean_loss: mean of loss array
        # for each alternate speaker, used for calculating sent_uniq_bpc:
        - alt_<spk>_mean_loss
        - alt_<spk>_bpc
        ```
- `speakers.json`, `speakers_metadata.json` - utility files with information on speakers
- `speech_metadata.jsonl` - each line corresponds to metadata of the speech as obtained from APP. Match from `train.csv` with the `speech_id` field.


## Re-scraping APP
If you want to scrape [the American Presidency Project](https://www.presidency.ucsb.edu/documents) for additional data not included in `train.csv`, `speech_metadata.jsonl` contains additional URLs we filtered out (e.g., Dem/Rep candidate debates, duplicate campaign stump speeches, etc.). Please see the paper for the full details of our data filtering steps.


## Citations

If you use this data please cite both of the following:
```
@misc{zhou2024quantifying,
      title={Quantifying the Uniqueness of Donald Trump in Presidential Discourse}, 
      author={Karen Zhou and Alexander A. Meitus and Milo Chase and Grace Wang and Anne Mykland and William Howell and Chenhao Tan},
      year={2024},
      eprint={2401.01405},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

and 
```
@article{woolley1999american,
  title={The American presidency project},
  author={Woolley, [dataset] John T and Peters, Gerhard},
  journal={Santa Barbara, CA. Available from World Wide Web: http://www. presidency. ucsb. edu/ws},
  year={1999}
}
```