import numpy as np
import contractions
import pandas as pd
from nltk.stem.snowball import SnowballStemmer

import string

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm
import argparse
import os
import spacy

from utils import load_data_and_speakers


sns.set_theme(font_scale=2.0, rc={
#    "lines.linewidth": 5,
   "lines.markersize":20,
   "ps.useafm": True,
   "font.sans-serif": ["Helvetica"],
   "pdf.use14corefonts" : True,
   })
sns.set_style("ticks")

nlp = spacy.load('en_core_web_sm')


def load_lexicon(file, use_stemmer=False):
    with open(file, "r") as f:
        lexicon = f.readlines()
    
    if use_stemmer:
        stemmer = SnowballStemmer("english")
        lexicon = [stemmer.stem(contractions.fix(word)) for word in lexicon]

    return lexicon


def get_words_per_speaker(df, speakers, lexicon, orig_text_field, use_stemmer=False):
    if use_stemmer: 
        stemmer = SnowballStemmer("english")
    res = {spk: {} for spk in speakers}
    all_words = {spk: 0 for spk in speakers}
    
    data = df.to_dict("records")
    for _, row in enumerate(tqdm(data)):
        speaker = row["speaker"]
        text = row[orig_text_field]
        # words = text.split()
        doc = nlp(text)
        for _, word in enumerate(doc):
            word = word.text
            if use_stemmer:
                word = stemmer.stem(contractions.fix(word.translate(str.maketrans('', '', string.punctuation))))
            else:
                word = contractions.fix(word.translate(str.maketrans('', '', string.punctuation)))

            all_words[speaker] += 1
            if word in lexicon:
                if word not in res[speaker]:
                    res[speaker][word] = 0
                res[speaker][word] += 1
    
    return res, all_words



def determine_term(row, data_type):
    elec_years = [1932, 1948, 1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020]


    if data_type == "debates" or data_type == "campaign":
        prev_term = -1
        for i, year in enumerate(elec_years):
            if row['year'] < year:
                return int(prev_term)
            prev_term = year

    else:
        prev_term = -1
        for i, year in enumerate(elec_years):
            if row['year'] <= year:
                return int(prev_term)
            prev_term = year

    return year


### score by speaker
def overall_score(outfile, spk_df, data_types, field="div_freq", title="Divisive Word Frequency", 
                  use_pct=False,
                  ment_key="all", use_intersect=False, sharey=True, overwrite=True):
    print(f"Overall {field}")
    c = sns.color_palette("pastel").as_hex()
    p = {'D': c[0], 'R':c[3]}

    font = { 'weight': 'normal',
    'size'   : 16}
    matplotlib.rc('font', **font)


    h = 8
    fig, axes = plt.subplots(1, len(data_types), figsize=(len(data_types) * h, h), sharey=sharey)

    for i, dt in enumerate(data_types):


        ppl_df = spk_df[spk_df["data_type"] == dt]

        if ment_key != "all":
            ppl_df = ppl_df[ppl_df["type"] == ment_key]

        if use_pct:
            ppl_df[field] = ppl_df[field] * 100

        grouping = ppl_df.groupby(["speaker"])[field].aggregate(np.mean).reset_index().sort_values(field)

        sns.barplot( data=ppl_df, x ="speaker", y = field,
            hue='party',
                    dodge=False, 
                    palette=p,
                    order=grouping["speaker"],
                    ax=axes[i]
                ) 
        axes[i].tick_params("x", labelrotation=90) 
        axes[i].set(title=f"[{dt.upper()}]")
        ylab = f"{title}{f' (%)' if use_pct else ''}"
        axes[i].set(ylabel=ylab if i == 0 else None, xlabel=None)


        if i != len(data_types) - 1: 
            axes[i].get_legend().remove()
        else:
            axes[i].get_legend().set_title("party")
            sns.move_legend(axes[i], "upper left", bbox_to_anchor=(1.1, 0.75), frameon=False)

        fig.tight_layout(rect=(0.025,0,1,1))
        fig.supxlabel("Speaker")

    if overwrite:
        title = "_".join(title.lower().split())
        plt.savefig(outfile, bbox_inches="tight")  
        plt.close(fig)
        print("saved overall score")



def plot_over_time(outfile, spk_df, data_types, x_field="year", y_field="div_freq", title="Divisive Word Frequency", 
                    use_pct=False,
                   ment_key="all", use_intersect=False, sharey=True, overwrite=True):
    c = sns.color_palette("pastel").as_hex()
    p = {'D': c[0], 'R':c[3]}

    font = { 'weight': 'normal',
    'size'   : 16}
    matplotlib.rc('font', **font)


    h = 6
    fig, axes = plt.subplots(1, len(data_types), figsize=(len(data_types) * h, h * 3/4), sharey=True)


    for i, dt in enumerate(data_types):


        spk_df['term'] = spk_df.apply(lambda x: determine_term(x, dt), axis=1)
        spk_df['term'] = spk_df['term'].astype(int)
        spk_df = spk_df[spk_df["term"] != -100]

        ppl_df = spk_df[spk_df["data_type"] == dt]

        if ment_key != "all":
            ppl_df = ppl_df[ppl_df["type"] == ment_key]

        temp = ppl_df.groupby([x_field, "party"]).sum()[
            ["num_div", "num_stems"]]
        temp[y_field] = temp["num_div"] / temp["num_stems"]
        temp = temp.reset_index()
        ppl_df = temp

        if use_pct:
            ppl_df[y_field] = ppl_df[y_field] * 100
        

        sns.lineplot(data=ppl_df, 
                     x=x_field, 
                     y=y_field, 
                     hue="party", 
                     style="party", 
                     style_order=p.keys(), 
                    #  markers=True,
                    #  markersize=10,
                    linewidth=2,
                     palette=p, 
                     ax=axes[i])
        axes[i].set(title=f"[{dt.upper()}]")
        ylab = ylab = f"{title}{f' (%)' if use_pct else ''}" if i == 0 else None
        axes[i].set(ylabel=ylab, xlabel=None)
        axes[i].legend(loc='upper left', title="party")
        plot_range = range(2008, 2020+2, 4) if dt == "campaign" else range(1960, 2020+2, 4) 
        axes[i].set_xticks(plot_range)
        axes[i].tick_params("x", labelrotation=90) 

        if i != len(data_types) - 1: 
            axes[i].get_legend().remove()
        else:
            axes[i].get_legend().set_title("party")
            sns.move_legend(axes[i], "upper left", bbox_to_anchor=(1.1, 0.75), frameon=False)

        fig.tight_layout(rect=(0.025,0,1,1))
        fig.supxlabel("Year")

    if overwrite:
        plt.savefig(outfile, bbox_inches="tight")  
        plt.close(fig)


def get_heat_map(ppl_df, words_dict, lexicon, delimiter=" ", normalize=False):
    # candidates = []
    m_words_dict = {}
    speakers = list(words_dict.keys())
    ppl_df = ppl_df[ppl_df["speaker"].isin(speakers)]
    grouping = ppl_df.groupby(["speaker"])["sent_uniq_bpc"].aggregate(np.mean).reset_index().sort_values("sent_uniq_bpc", ascending=False)
    speakers = list(grouping["speaker"])
    print(speakers)

    all_words = ppl_df.groupby('speaker')['num_stems'].sum().to_dict()
    
    # lexicon = sorted(lexicon)
    for i, word in enumerate(lexicon):
        m_words_dict[word] = i
    heatmap = np.zeros((len(speakers), len(lexicon)))
    
    for i, speaker in enumerate(speakers):
        candidate = speaker
        # if candidate in filter_speakers: # or " ".join(candidate.split(delimiter)[:-1]) in filter_speakers:
        if normalize:
            total_words = all_words[candidate]
        for word in words_dict[candidate]:
            heatmap[i][m_words_dict[word]] = words_dict[candidate][word]/total_words if normalize else words_dict[candidate][word]
    
    df = pd.DataFrame(heatmap, columns = lexicon, index = speakers)
    df = df.loc[:, (df != 0).any(axis=0)]
    df = df[~(df == 0).all(axis=1)]
    return df


def plot_heatmap(outfile, spk_df, data_speakers, data_types, lexicon, use_stemmer, normalize=False, use_intersect=True, overwrite=True):
    
    for dt in data_types:
        df = spk_df[spk_df["data_type"] == dt]
        speakers = data_speakers[dt]

        words_dict, _ = get_words_per_speaker(df, speakers, lexicon, use_stemmer=use_stemmer)
        div_df = get_heat_map(df, words_dict, lexicon, speakers, normalize=normalize)

        h = len(div_df)
        w = len(div_df.columns)

        if not use_stemmer:
            sns.set(font_scale = 3.5)
        else:
            sns.set(font_scale = 2.0)

        fig, ax = plt.subplots(figsize=(w, h))
        sns.heatmap(div_df, xticklabels=True, yticklabels=True, cmap = "rocket_r")
        plt.ylabel("Speaker")
        plt.xlabel("Divisive Word")

        if overwrite:
            plt.savefig(outfile, bbox_inches="tight")
            plt.close(fig)


def main():

    code_dir = os.path.dirname(os.path.abspath(__file__))
    working_dir = os.path.dirname(code_dir)

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_types', 
                        type=str, 
                        default=["debates,sotu,campaign"], 
                        help="comma-separated data types to use: debates, sotu, campaign") 
    parser.add_argument('--data_dir', 
                        type=str, 
                        default=os.path.join(working_dir, "data"),
                        help="directory of data") 
    parser.add_argument('--figure_dir', 
                        type=str, 
                        default=os.path.join(working_dir, "figures"),
                        help="directory to save figures")
    parser.add_argument('--mask_ents', 
                        type=bool, 
                        default=True,
                        help="mask named entities in text, true by default")
    parser.add_argument('--stem',
                        default=0,
                        type=int,
                        help="1 if use stemmer, 0 if not")
    parser.add_argument('--lexicon', 
                        type=str,
                        default="majority",
                        options=["majority", "unanimous"],
                        help="version of lexicon to use")
    
    args = parser.parse_args()

    data_types = [dt.strip() for dt in args.data_types.split(",")]
    orig_text_field = "text_orig_masked" if args.mask_ents else "text_orig"
    fig_dir = args.fig_dir

    lex_path = os.path.join(working_dir, "divisive-word-lexicon", f"{args.lexicon}_words.txt")

    lexicon = load_lexicon(lex_path, args.stem)


    lm_datas = {}
    data_speakers = {}
    metadatas = {}
    spk_dfs = {}
    div_rates = {}

    # load data and calculate usage rates
    for data_type in data_types:
        data_dir = os.path.join(args.data_dir, data_type)
        data, speakers, metadata = load_data_and_speakers(data_dir, data_type, load_metadata=True)
 
        data["data_type"] = data_type
        data["type"] = data["mentions_opponent"]
        data.rename(columns={"speaker_clean": "speaker"}, inplace=True)
        data["is_trump"] = data["speaker"] == "Donald Trump"  
        data["num_words"] = data[orig_text_field].apply(lambda x: len(x.split()))
        temp_df = data[data["speaker"].isin(data_speakers[data_type].keys())]

        lm_datas[data_type] = data
        data_speakers[data_type] = speakers
        metadatas[data_type] = metadata
        spk_dfs[data_type] = temp_df

        temp = temp_df.groupby("speaker").sum()[
            ["num_div", "num_stems"]]
        temp["div_freq"] = temp["num_div"] / temp["num_stems"]
        temp = temp.reset_index()
        temp["data_type"] = data_type

        div_rates[data_type] = temp

    spk_df = pd.concat(spk_dfs.values())
    div_rates_df = pd.concat(div_rates.values())


    # call plotting functions
    for use_pct in [False, True]:
        outfile = os.path.join(fig_dir, f"overall_div_freq_{args.lexicon}{'_stem' if args.stem else ''}.pdf")
        overall_score(outfile, div_rates_df, data_types, "div_freq", "Divisive Word Frequency", 
                      use_pct=use_pct)
        
        outfile = os.path.join(fig_dir, f"overall_div_freq_{args.lexicon}{'_stem' if args.stem else ''}_pct.pdf")
        plot_over_time(outfile, spk_df, data_types, "year", "div_freq", "Divisive Word Frequency", 
                        use_pct=use_pct)


    outfile = os.path.join(fig_dir, f"heatmap_{args.lexicon}{'_stem' if args.stem else ''}.pdf")
    plot_heatmap(outfile, spk_df, data_speakers, data_types, lexicon, use_stemmer=args.stem, normalize=False)
    
    outfile = os.path.join(fig_dir, f"heatmap_{args.lexicon}{'_stem' if args.stem else ''}_norm.pdf")
    plot_heatmap(outfile, spk_df, data_speakers, data_types, lexicon, use_stemmer=args.stem, normalize=True)


if __name__ == "__main__":
    main()

