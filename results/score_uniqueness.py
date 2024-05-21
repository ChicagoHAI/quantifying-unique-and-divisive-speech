from utils import read_progress, load_data_and_speakers
import argparse
import os
import pandas as pd
import json
import torch
from torch import nn
from tqdm import tqdm
import yaml
from pathlib import Path
import string
import numpy as np

from torch.utils.data import DataLoader


import sys

try:  ## FOR GPT-2
    import pytorch_lightning as pl
    sys.path.append(
        'quantifying-unique-and-divisive-speech/models/gpt2') # TODO: Change to your own path to repo base dir
    from model import Transformer_PL  
    pl.seed_everything(42)
except:
    # ## FOR GEMMA2B
    sys.path.append(
        '<path to LLaMA-Factory>') # Change to your own path to repo base dir
    from llmtuner.model.loader import load_model, load_config, load_tokenizer
    from llmtuner.hparams import  get_train_args





def get_data_for_speaker(data, speaker):
    '''
    Return DataFrames with speeches ft speaker only
    '''
    spk_data = data[data["speaker"] == speaker]
    spk_temp = spk_data["speech_id"].unique()
    spk_df = data[data["speech_id"].isin(spk_temp)]

    return spk_df



def get_speaker_sent_tokens_with_context(
    spk_df, 
    speaker, 
    tokenizer, 
    orig_text_field,
    max_length=512,
    batch_size=8,
    ):
    '''
    Return list of dictionaries with tokenized sentences for speaker with context
    '''

    res = []

    speaker_lines = len(spk_df[spk_df["speaker"] == speaker])

    for speech_id in spk_df["speech_id"].unique():  # for each speech
        spk_speech_df = spk_df[spk_df["speech_id"]==speech_id]
        spk_data = spk_speech_df.to_dict("records")

        sent_tokens = [
            tokenizer(row["text"], return_tensors="pt") for row in spk_data
        ]
        text_tokens = [
            tokenizer(row[orig_text_field], return_tensors="pt") for row in spk_data
        ]

        for i, row in enumerate(spk_data):  # check if each row is speaker's sentence
            if row["speaker"] == speaker:

                orig_text = row[orig_text_field]
                text = row["text"]
                toks = sent_tokens[i]
                orig_len = toks.input_ids.size(1)
                trg_len = text_tokens[i].input_ids.size(1)

                ct_id = i - 1
                size_acc = toks.input_ids.size(1)

                # if batch_size > 1 and  longer than max_length, truncate the end
                if batch_size > 1 and size_acc > max_length:
                    for key in toks.keys():
                        toks[key] = toks[key][:, :max_length]
                    size_acc = max_length
                    assert size_acc == toks.input_ids.size(1)

                # if shorter than max_length, add context
                while size_acc < max_length and ct_id >= 0:
                    prev = sent_tokens[ct_id]
                    prev_len = prev.input_ids.size(1)
                    if size_acc + prev_len <= max_length:
                        for key in toks.keys():
                            toks[key] = torch.cat([prev[key], toks[key]] , 1)
                        size_acc += prev_len
                        assert size_acc == toks.input_ids.size(1)
                    else:
                        fragment_size = max_length - size_acc
                        for key in toks.keys():
                            toks[key] = torch.cat([prev[key][:, -fragment_size:], toks[key]] , 1)
                        size_acc += fragment_size
                        assert size_acc == toks.input_ids.size(1)

                    ct_id -= 1
                
                # if less than max_length, pad on left
                if size_acc < max_length:
                    pad_size = max_length - size_acc
                    for key in toks.keys():
                        toks[key] = torch.cat([torch.zeros(toks[key].size(0), pad_size, dtype=toks[key].dtype), toks[key]], 1)

                
                assert toks.input_ids.size(1) == max_length if batch_size > 1 else True

                target_ids = toks.input_ids.clone()
                target_ids[:, :-trg_len] = -100  # set speaker prompt and context to -100, to ignore

                ment_key = row["mentions_opponent"]
                res.append(
                    {"orig_text": orig_text,
                    "text": text,
                    "toks" : toks, 
                    "target_ids": target_ids, 
                    "target_len": trg_len,
                    "ment_key": ment_key}
                )

    assert len(res) == speaker_lines, f"{len(res)} != {speaker_lines}"

    return res 


def get_mean_loss(losses):
    mean_loss = np.nanmean([np.nanmean(l) for l in losses])
    return mean_loss


def hf_bpc_context_sent_batched(
        model, 
        model_type,
        tokenizer,
        sent_batch, 
        device, 
        batch_size=8,
        add_start_token=True,
        skip_len=1):
    '''
    use with `get_speaker_sent_tokens_with_context` -- gets sent losses w/ context for sentence
    '''

    losses = []


    loss_fct = nn.CrossEntropyLoss(reduction="none")

    for batch_start in tqdm(range(0, len(sent_batch), batch_size)):
        batch_end = min(batch_start + batch_size, len(sent_batch))
        batch = sent_batch[batch_start:batch_end]

        encoded_batch = torch.cat([b["toks"].input_ids for b in batch], dim=0).to(device)
        attn_mask = torch.cat([b["toks"].attention_mask for b in batch], dim=0).to(device)
        trg_ids = torch.cat([b["target_ids"] for b in batch], dim=0).to(device)
        trg_lens = [b["target_len"] for b in batch]
        sent_text = [b["text"] for b in batch]

        if add_start_token:
            bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(0)).to(device)

            encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
            trg_ids = torch.cat([bos_tokens_tensor, trg_ids], dim=1)
            attn_mask = torch.cat(
                [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
            )

        with torch.no_grad():
            if model_type == "gpt2":
                out_logits = model.model(encoded_batch, attention_mask=attn_mask).logits
                shift_logits = out_logits[..., :-1, :].contiguous()
                shift_labels = trg_ids[..., 1:].contiguous()
                shift_attention_mask_batch = attn_mask[..., 1:].contiguous()
            else:
                with torch.cuda.amp.autocast():
                    out_logits = model(input_ids=encoded_batch, attention_mask=attn_mask).logits
                    shift_logits = out_logits[..., :-1, :].contiguous()
                    shift_labels = trg_ids[..., 1:].contiguous()
                    shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        batch_loss = loss_fct(shift_logits.transpose(1,2), shift_labels)

        for i, loss in enumerate(batch_loss):
            trg_len = trg_lens[i]
            if trg_len <= skip_len:
                losses.append([])
                continue
            if trg_len > len(loss):
                trg_len = len(loss)
                loss = loss[:trg_len].tolist()
            else:
                loss = loss[-trg_len:].tolist()
            assert len(loss) == trg_len, f"{len(loss)} != {trg_len}, {sent_text[i]}"
            losses.append(loss)
   


    assert len(losses) == len(sent_batch)
    return {
        "losses": losses,
        "mean_loss": get_mean_loss(losses),
        }


def generate_all_losses_sent_context(
        model, 
        model_type, 
        tokenizer, 
        data, 
        speakers, 
        output_dir, 
        orig_text_field, 
        device,
        batch_size=8,
        window_size=512,
        add_start_token=True,
        skip_len=1
    ):

    output_file = os.path.join(output_dir, "all_spk_losses.jsonl")
    done_spks = read_progress(output_file)
    done = set(done_spks.keys()) if done_spks else set()
    print(f"Done: {done}")
    
    
    for speaker in speakers:
        if speaker in done:
            continue
        print(speaker)
        spk_df = get_data_for_speaker(data, speaker)
        # spk_data = spk_df[spk_df["speaker"]==speaker].to_dict("records")
        sent_batches = get_speaker_sent_tokens_with_context(spk_df, speaker, tokenizer, orig_text_field, max_length=window_size)

        d = {speaker : {
            "all": {},
            # "Y": {},
            # "N": {}
        }}

        loss_res = hf_bpc_context_sent_batched(model, model_type, tokenizer, sent_batches, device, batch_size, add_start_token, skip_len)
        d[speaker]["all"]["mean_loss"] = loss_res["mean_loss"]
        d[speaker]["all"]["losses"] = loss_res["losses"]


        with open(output_file, "a") as fout:
            fout.write(json.dumps(d) + "\n")


def replace_spk_losses_sent_context(
        model, 
        model_type,
        tokenizer, 
        data, 
        speakers, 
        output_dir, 
        orig_text_field, 
        device,
        batch_size=8,
        window_size=512,
        add_start_token=True,
        skip_len=1
        ):
    ### REPLACING {SPEAKER}
    all_spk_losses = read_progress(os.path.join(output_dir, "all_spk_losses.jsonl"))

    # rep_losses = {speaker : {} for speaker in speakers}

    for speaker in speakers:
        spk_temp = speaker.translate(str.maketrans('', '', string.punctuation))
        spk_name = "_".join(spk_temp.lower().split())
        print(spk_name)
        output_file = os.path.join(output_dir, f"{spk_name}_rep_losses.jsonl")

        done_spks = read_progress(output_file)
        done_spks = set(done_spks.keys()) if done_spks else set()

        alt_speakers = set(speakers.keys())
        alt_speakers -= done_spks
        print(f"Done: {done_spks}")

        for alt_spk in alt_speakers:
            print(f"{speaker} --> {alt_spk}")

            def _repl_spk_in_text(row):
                if row["speaker"] == speaker:
                    return alt_spk + ": " + row[orig_text_field]
                return row["text"]

            d = {alt_spk : {
                "all": {},
                }
            }
            if alt_spk == speaker:
                d[speaker] = all_spk_losses[speaker]
                with open(output_file, "a") as fout:
                    fout.write(json.dumps(d) + "\n")
                continue

            spk_df = get_data_for_speaker(data, speaker)
            
            # replace speaker prompt with alt speaker
            spk_df.loc[:, "text"] = spk_df.apply(_repl_spk_in_text, axis=1)

            sent_batches = get_speaker_sent_tokens_with_context(
                spk_df, speaker, tokenizer, orig_text_field, max_length=window_size)

            loss_res = hf_bpc_context_sent_batched(model, model_type, tokenizer, sent_batches, device, batch_size, add_start_token, skip_len)

            d[alt_spk]["all"]["mean_loss"] = loss_res["mean_loss"]
            d[alt_spk]["all"]["losses"] = loss_res["losses"]

            with open(output_file, "a") as fout:
                fout.write(json.dumps(d) + "\n")


def _get_bits_per_char(losses, orig_sent):
    '''
    Compute bits per character for a sentence
    '''
    sent_len = len(orig_sent)
    bits_per_char = sum(losses) / sent_len

    return bits_per_char


def match_losses_to_csv(model_type, data, speakers, data_dir, output_dir, orig_text_field):
    '''
    Match sentence losses from jsonl files to sentence csv rows and calculate bpc and uniqueness
    '''
    print("creating csv")
    all_spk_losses = read_progress(os.path.join(output_dir, "all_spk_losses.jsonl"))
    ment_key = "all"

    data_dict = data.to_dict("index")

    for speaker in speakers:
        print(speaker)
        spk_df = get_data_for_speaker(data, speaker)
        spk_data = spk_df[spk_df["speaker"]==speaker].to_dict("index")

        assert len(spk_data) == len(all_spk_losses[speaker][ment_key]["losses"])

        # alt speaker lossess
        spk_temp = speaker.translate(str.maketrans('', '', string.punctuation))
        spk_name = "_".join(spk_temp.lower().split())
        in_file = os.path.join(output_dir, f"{spk_name}_rep_losses.jsonl")
        rep_losses = read_progress(in_file)

        for sent_idx, (df_idx, row) in tqdm(enumerate(spk_data.items())):

            row["loss"] = all_spk_losses[speaker][ment_key]["losses"][sent_idx]
            row["mean_loss"] = np.nanmean(all_spk_losses[speaker][ment_key]["losses"][sent_idx])
            row["sent_len"] = len(row["loss"])
            row["bpc"] = _get_bits_per_char(row["loss"], row[orig_text_field])

            if np.isnan(row["mean_loss"]) or row["sent_len"] == 0:
                print(f"{speaker}: nan loss for {sent_idx}")
                data_dict[df_idx] = row
                continue

            alt_acc = 0.0
            alt_bpc = 0.0
            num_others = 0
            for alt_spk in rep_losses.keys():
                if alt_spk == speaker: continue

                alt_temp = alt_spk.translate(str.maketrans('', '', string.punctuation))
                alt_name =  "_".join(alt_temp.lower().split())
                row[f"alt_{alt_name}_loss"] = rep_losses[alt_spk][ment_key]["losses"][sent_idx]
                row[f"alt_{alt_name}_mean_loss"] = np.nanmean(rep_losses[alt_spk][ment_key]["losses"][sent_idx])
                if np.isnan(row[f"alt_{alt_name}_mean_loss"]):
                    print(f"{speaker}: nan loss for {alt_name}")
                    continue
                alt_acc += row[f"alt_{alt_name}_mean_loss"]
                row[f"alt_{alt_name}_bpc"] = _get_bits_per_char(row[f"alt_{alt_name}_loss"], row[orig_text_field])
                alt_bpc += row[f"alt_{alt_name}_bpc"]
                num_others +=1

            if num_others > 0:
                row["sent_uniq_bpc"] = (alt_bpc / num_others) - row["bpc"]

            data_dict[df_idx] = row

    lm_data = pd.DataFrame.from_dict(data_dict, orient="index")

    with open(os.path.join(data_dir, f"scored_data_{model_type}.csv"), "w") as f:
        lm_data.to_csv(f, index=False)
        print("saved csv")




def load_from_llama_factory(config_path, data_type, model_type):
    args = yaml.safe_load(Path(f"{config_path}/{model_type}_lora_{data_type}.yaml").read_text())
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)
    return model, tokenizer


def main():
    code_dir = os.path.dirname(os.path.abspath(__file__))
    working_dir = os.path.dirname(code_dir)

    parser = argparse.ArgumentParser(description="Calculate uniqueness after finetuning your model")
    parser.add_argument('--model_type', 
                        type=str, 
                        choices=["gpt2", "gemma2b", "phi1-5b"], 
                        required=True,
                        help="gpt2 (original) or gemma2b (validation)")
    parser.add_argument('--model_checkpoint', 
                        type=str, 
                        required=True,
                        help="path to model checkpoint") 
    parser.add_argument('--data', 
                        type=str, 
                        choices=["debates", "sotu", "campaign"], 
                        required=True,
                        help="debates | sotu | campaign") 
    parser.add_argument('--data_dir', 
                        type=str, 
                        default=os.path.join(working_dir, "data"),
                        help="directory of data") 
    parser.add_argument('--output_dir', 
                        type=str, 
                        default=os.path.join(working_dir, "results"),
                        help="directory to save results")
    parser.add_argument('--cache_dir',
                        type=str,
                        default=None,  # TODO: change, indigo: "/data/transformers". bingo "/data/karen/transformers"
                        help="directory to save cache files")
    parser.add_argument('--lf_config',
                        type=str,
                        default=None,
                        help='path to llama factory config file')
    parser.add_argument('--mask_ents', 
                        type=int, 
                        default=1,
                        help="mask named entities in text, true by default")
    parser.add_argument('--device', 
                        type=str, 
                        default="cuda",
                        help="device to use for model inference")
    parser.add_argument('--window_size', 
                        type=int, 
                        default=512,
                        help="context window size for calculating loss")
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=8,
                        help="batch size for calculating loss")
    parser.add_argument('--add_start_token', 
                        type=int, 
                        default=0,
                        help="whether to add start token")
    

    args = parser.parse_args()

    data_dir = os.path.join(args.data_dir, args.data)

    data, speakers, metadata = load_data_and_speakers(data_dir, args.data, model_type=None, load_metadata=True)
    device = args.device

    if args.model_type == "gpt2":
        # device = "cuda"
        model = Transformer_PL.load_from_checkpoint(
            args.model_checkpoint,
            cache_dir=args.cache_dir)
        model.to(device)
        model.eval()
        model.freeze()
        if device != "cpu":
            model.half()
        tokenizer = model.tokenizer
    elif args.model_type == "gemma2b" or args.model_type == "phi1-5b":
        model, tokenizer = load_from_llama_factory(args.lf_config, args.data, args.model_type)
        tokenizer.add_bos_token = False  # very important!!
    else:
        raise ValueError("Model type not recognized")
    
    orig_text_field = "text_orig_masked" if args.mask_ents else "text_orig"

    skip_len = 1 if args.add_start_token == 1 else 2
    

    # get losses for original speaker
    generate_all_losses_sent_context(
        model, 
        args.model_type,
        tokenizer, 
        data, 
        speakers, 
        args.output_dir, 
        orig_text_field, 
        args.device,
        args.batch_size,
        args.window_size,
        args.add_start_token,
        skip_len
        )

    # get losses for alternate speakers
    replace_spk_losses_sent_context(
        model, 
        args.model_type,
        tokenizer, 
        data, 
        speakers, 
        args.output_dir, 
        orig_text_field, 
        args.device,
        args.batch_size,
        args.window_size,
        args.add_start_token,
        skip_len
        )
    
    # create new csv with scores
    match_losses_to_csv(args.model_type, data, speakers, data_dir, args.output_dir, orig_text_field)



if __name__ == "__main__":
    main()

