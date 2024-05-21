import json
import os
import pandas as pd
from tqdm import tqdm
import string

def read_progress(filename):
    if not os.path.exists(filename):
        print(f"File {filename} does not exist.")
        return None
    data = {}
    with open(filename, "r") as fin:
        for line in fin:
            line = json.loads(line)
            data.update(line)
    return data


def read_jsonlist(filename):
    if not os.path.exists(filename):
        print(f"File {filename} does not exist.")
        return None
    data = []
    with open(filename, "r") as fin:
        for line in fin:
            line = json.loads(line)
            data.append(line)
    return data


def load_data_and_speakers(data_dir, data_type, model_type=None, load_metadata=False):

    with open(f"{data_dir}/speakers.json", "r") as f:
        speakers = json.load(f)
        del speakers["Other"]

    if data_type == "sotu": # or DATA_TYPE == "campaign":
        with open(f"{data_dir}/speakers_metadata.json", "r") as f:
            metadata = json.load(f)
            speakers = {v["clean"]: v["lm_id"] for _, v in metadata.items()}

    if data_type == "campaign":
        with open(f"{data_dir}/speakers_metadata.json", "r") as f:
            metadata = json.load(f)
            speakers = {v["clean"]: v["lm_id"] for _, v in metadata.items()
                        if len(v["year"]) >0 and int(v["year"][-1]) >= 2008}

    spk_data = None
    if load_metadata:
        with open(f"{data_dir}/speakers_metadata.json", "r") as f:
            if data_type == "sotu" or data_type == "campaign":
                metadata = json.load(f)
                spk_data = {v["clean"]: v for _, v in metadata.items()}
            else:
                spk_data = json.load(f)

    if model_type:
        in_file = f"{data_dir}/scored_data_{model_type}.csv"
    else:
        in_file = f"{data_dir}/train.csv"
    data = pd.read_csv(in_file)


    return data, speakers, spk_data


def get_uniqueness_w_speaker_subset(df, subset):
    data_dict = df.to_dict(orient="records")
    speaker_dict = df.groupby("speech_id")["speaker"].unique().apply(set).to_dict()

    for _, row in enumerate(tqdm(data_dict)):
        speaker = row["speaker"]
        speech_id = row["speech_id"]
        debate_speakers = speaker_dict[speech_id]

        if speaker not in subset: continue 
        alt_speakers = set(subset.keys()) - debate_speakers

        alt_acc = 0.0
        alt_bpc = 0.0
        num_others = 0
        for alt_spk in alt_speakers:
        
            alt_temp = alt_spk.translate(str.maketrans('', '', string.punctuation))
            alt_name =  "_".join(alt_temp.lower().split())

            alt_acc += row[f"alt_{alt_name}_mean_loss"]
            alt_bpc += row[f"alt_{alt_name}_bpc"]
            num_others +=1

        if num_others > 0:
            row["sent_uniq_bpc"] = (alt_bpc / num_others) - row["bpc"]


    new_df = pd.DataFrame(data_dict)
    return new_df


def data_with_intersected_speakers(lm_datas, data_speakers, metadatas, spk_dfs, use_intersect=False):
    '''
    Update stored data and metadata depending on whether we want to use the intersection of speakers
    '''
    if use_intersect:
        tmp = None  # get intersection of speakers for all datatypes
        for dt, speakers in data_speakers.items():
            if tmp is None:
                tmp = set(speakers.keys())
            else:
                if dt != "campaign":
                    tmp = tmp.intersection(set(speakers.keys()))
    else:
        tmp = None  # get UNION of speakers for all datatypes
        for _, speakers in data_speakers.items():
            if tmp is None:
                tmp = set(speakers.keys())
            else:
                tmp = tmp.union(set(speakers.keys()))


    spk_intersect = {}
    metadata_intersect = {}
    for dt, speakers in data_speakers.items():
        for s in speakers:
            if s in tmp:
                spk_intersect[s] = speakers[s]
                metadata_intersect[s] = metadatas[dt][s]


    new_datas = {}
    new_speakers = {}
    new_metadatas = {}
    new_spk_dfs = {}    
    if use_intersect:
        for dt, lm_data in lm_datas.items():
            if dt != "campaign":
                new_datas[dt] = get_uniqueness_w_speaker_subset(lm_data, spk_intersect)
                new_spk_dfs[dt] = lm_data[lm_data["speaker"].isin(spk_intersect.keys())]
                new_speakers[dt] = spk_intersect
                new_metadatas[dt] = metadata_intersect
            elif dt == "campaign":
                new_datas[dt] = get_uniqueness_w_speaker_subset(lm_data, data_speakers[dt])
                new_spk_dfs[dt] = lm_data[lm_data["speaker"].isin(data_speakers[dt].keys())]
                new_speakers[dt] = data_speakers[dt]
                new_metadatas[dt] = metadatas[dt]
    else:
        for dt, lm_data in lm_datas.items():
            new_datas[dt] = lm_data
            new_spk_dfs[dt] = lm_data[lm_data["speaker"].isin(data_speakers[dt].keys())]
            new_speakers[dt] = data_speakers[dt]
            new_metadatas[dt] = metadatas[dt]

    lm_data = pd.concat(lm_datas.values())
    
    return new_datas, new_speakers, new_metadatas, new_spk_dfs
    

def assign_normalized_sent_ids(lm_data, speakers, use_full=True):
    
    grouped = lm_data.groupby('speech_id').size()
    speech_lengths =  dict(grouped)
    
    if use_full:
        lm_dict = lm_data.to_dict("records")
    # # add sentence_id for each debate
    else:
        ppl_df = lm_data[lm_data["speaker"].isin(speakers)]
        lm_dict = ppl_df.to_dict("records")
        
    spch = 0
    sent_id = 0
    prev_spch = -1
    spch_len = -1

    for row in lm_dict:
        spch = row["speech_id"]
        if spch == prev_spch:
            sent_id += 1
            row["sent_id"] = sent_id / spch_len * 100
        else:
            prev_spch = spch
            spch_len = speech_lengths[spch]
            sent_id = 0
            row["sent_id"] = sent_id / spch_len * 100

    lm_data = pd.DataFrame(lm_dict)

    ppl_df = lm_data[lm_data["speaker"].isin(speakers)]
    
    return ppl_df