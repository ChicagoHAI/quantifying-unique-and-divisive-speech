import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import argparse
import os

from fw_utils import calc_overlap_metrics, calc_fightin_words_by_speaker
from utils import load_data_and_speakers



sns.set_theme(font_scale=2.0, rc={
#    "lines.linewidth": 5,
   "lines.markersize":20,
   "ps.useafm": True,
   "font.sans-serif": ["Helvetica"],
   "pdf.use14corefonts" : True,
   })

sns.set_style("ticks")
c = sns.color_palette("pastel").as_hex()


def fw_plot(fig_dir, data_types, dt_fw, metadatas, ment_key="all", errbar=('ci', 95), hue="spk_party", title="FW", pos_tags=["ALL"], overwrite=True):

    h = 8
    sharey=False
    fig, axes = plt.subplots(1, len(data_types), figsize=(len(data_types) * h , h), sharey=sharey)

    for i, dt in enumerate(data_types):
        spk_data = metadatas[dt]
        
        fw_list = dt_fw[dt]

        
        overlap_mo = calc_overlap_metrics(fw_list, pos_tags=pos_tags)
        overlap_mo["party"] = overlap_mo["speaker"].apply(lambda x: spk_data[x]["party"])
        overlap_mo["spk_party"] = overlap_mo["speaker"].apply(lambda x: spk_data[x]["party"] 
                                                            if x != "Donald Trump" else "Trump")
        overlap_mo["spk_party"] = overlap_mo["spk_party"].apply(lambda x: "R (excl. Trump)" if x == "R" else x)
        
        if ment_key != "all":
            overlap_mo = overlap_mo[overlap_mo["mentions_opponent"] == ment_key]
        
        
        p = {'D': c[0], 'R (excl. Trump)':c[3], "Trump": c[1]}
        
        sns.lineplot(data= overlap_mo,
                    x="N", 
                    y="score", 
                    hue=hue, 
                    palette=p,
                    style=hue,
                    style_order=p.keys(),
                    markers=True,
                    markersize=20,
                    linewidth=5,
                    ax=axes[i],
                    errorbar=errbar)\
        .set_title(f"[{dt.upper()}]")\
        
        axes[i].set(ylabel=f"Overlap Score" if i == 0 else None, xlabel=None)

        if i != len(data_types) - 1: 
            axes[i].get_legend().remove()
        else:
            axes[i].get_legend().set_title("party")
            sns.move_legend(axes[i], "upper left", bbox_to_anchor=(1.1, 0.75), frameon=False)

    fig.tight_layout(rect=(0.025,0.05,1,1))
    supx = f"# Top Overlapping FW" if pos_tags == ["ALL"] else f"# Top Overlapping FW [{' & '.join(pos_tags)}]" 
    fig.supxlabel(supx if ment_key == "all" 
                  else f"{supx} | Mentions Opponent = {ment_key}")    
    
    if overwrite:
        sharedy = f"_unshared" if not sharey else ""
        title = "_".join(title.lower().split())
        err_suffix = "_ci" if errbar else ""
        pos_tag_suffix = "_" + "_".join(pos_tags)
        plt.savefig(f"{fig_dir}/fig_agg_{title.lower()}_overlap_{ment_key.lower()}{sharedy}{err_suffix}{pos_tag_suffix}.pdf", 
                    bbox_inches="tight")  
        plt.close(fig)
                                        

def fw_plot_by_data(fig_dir, data_type, fw_list, spk_data, errbar, ment_keys = ["Y", "N"], hue="spk_party", title="FW", pos_tags=["ALL"], sharey=True, overwrite=True):
    

    if len(ment_keys) >= 2:
        h = 8
        
        fig, axes = plt.subplots(1, len(ment_keys), figsize=(len(ment_keys) * h , h ), sharey=sharey)
    else:
        h = 8
        fig, axes = plt.subplots(1, 1, figsize=(len(ment_keys) * h , h ), sharey=False)
        
    
    overlap_mo = calc_overlap_metrics(fw_list, pos_tags=pos_tags)
    overlap_mo["party"] = overlap_mo["speaker"].apply(lambda x: spk_data[x]["party"])
    overlap_mo["spk_party"] = overlap_mo["speaker"].apply(lambda x: spk_data[x]["party"] 
                                                        if x != "Donald Trump" else "Trump")
    
    overlap_mo["spk_party"] = overlap_mo["spk_party"].apply(lambda x: "R (excl. Trump)" if x == "R" else x)
           
    for i, ment_key in enumerate(ment_keys):
        if ment_key != "all":
            ment_df = overlap_mo[overlap_mo["mentions_opponent"] == ment_key]
        else:
            ment_df = overlap_mo.copy()   
    
        # p = {'D': c[0], 'R':c[3], "Trump": c[1]}
        p = {'D': c[0], 'R (excl. Trump)':c[3], "Trump": c[1]}
        
        sns.lineplot(data= ment_df,
                    x="N", 
                    y="score", 
                    hue=hue, 
                    palette=p,
                    style=hue,
                    style_order=p.keys(),
                    markers=True,
                    markersize=20,
                    linewidth=5,
                    ax=axes[i] if len(ment_keys) > 1 else axes,
                    errorbar=errbar)\
        .set_title(f"Mentions Opponent: {ment_key}" if ment_key != "all" else "All FW Overlap")
        
        plot_range = range(5, 26, 5)
        if len(ment_keys) > 1:
            axes[i].set(ylabel=f"Overlap Score" if i == 0 else None, xlabel=None)
            axes[i].set_xticks(plot_range)
            # axes[i].get_legend().set_title(None)
            if i != len(ment_keys) - 1: 
                axes[i].get_legend().remove()
            else:
                axes[i].get_legend().set_title("party")
                sns.move_legend(axes[i], "upper left", bbox_to_anchor=(1.1, 0.75), frameon=False)
        else:
            axes.set_xticks(plot_range)
            axes.set(ylabel=f"Overlap Score", 
                     xlabel="# Top Overlapping FW" if pos_tags == ["ALL"] 
                  else f"# Top Overlapping FW [{pos_tags[0]}]" if len(pos_tags) == 1
                  else f"# Top Overlapping FW [{' & '.join(pos_tags)}]")
            axes.get_legend().set_title("party")
            sns.move_legend(axes, "best", frameon=True)

    if len(ment_keys) > 1:
        fig.tight_layout(rect=(0.025,0.05,1,1))
        fig.supxlabel("# Top Overlapping FW" if pos_tags == ["ALL"] 
                  else f"# Top Overlapping FW [{pos_tags[0]}]" if len(pos_tags) == 1
                  else f"# Top Overlapping FW [{' & '.join(pos_tags)}]")   
    
        
    if overwrite:
        sharedy = f"_unshared" if not sharey else ""
        title = "_".join(title.lower().split())
        err_suffix = "_ci" if errbar else ""
        pos_tag_suffix = "_" + "_".join(pos_tags)
        plt.savefig(f"{fig_dir}/fig_{title.lower()}_overlap_{len(ment_keys)}_side{sharedy}{err_suffix}{pos_tag_suffix}.pdf", 
                    bbox_inches="tight")  
        plt.close(fig)


def fw_bar_plot_by_data(fig_dir, n, data_type, fw_list, spk_data, ment_keys = ["Y", "N"], hue="party", title="FW", pos_tags=["ALL"], sharey=True, overwrite=True):
    

    if len(ment_keys) == 2:
        h = 8
        fig, axes = plt.subplots(1, len(ment_keys), figsize=(len(ment_keys) * h , h ), sharey=sharey)
    elif len(ment_keys) == 3:
        h = 6
        fig, axes = plt.subplots(1, len(ment_keys), figsize=(len(ment_keys) * h , h ), sharey=sharey)
        font = { 'weight': 'normal',
            'size'   : 18}
        matplotlib.rc('font', **font)
    else:
        h = 8
        fig, axes = plt.subplots(1, 1, figsize=(len(ment_keys) * h , h ), sharey=False)
        font = { 'weight': 'normal',
            'size'   : 18}
        matplotlib.rc('font', **font)


    overlap_mo = calc_overlap_metrics(fw_list, pos_tags=pos_tags)
    
    overlap_mo = overlap_mo[overlap_mo["N"] == n]

    overlap_mo["party"] = overlap_mo["speaker"].apply(lambda x: spk_data[x]["party"])
    overlap_mo["spk_party"] = overlap_mo["speaker"].apply(lambda x: spk_data[x]["party"] 
                                                        if x != "Donald Trump" else "Trump")
    
    overlap_mo["spk_party"] = overlap_mo["spk_party"].apply(lambda x: "R (excl. Trump)" if x == "R" else x)
           
    for i, ment_key in enumerate(ment_keys):
        if ment_key != "all":
            ment_df = overlap_mo[overlap_mo["mentions_opponent"] == ment_key]
        else:
            ment_df = overlap_mo.copy()   
    
        p = {'D': c[0], 'R':c[3]}
        # p = {'D': c[0], 'R (excl. Trump)':c[3], "Trump": c[1]}
        
        grouping = ment_df.groupby(["speaker"])["score"].aggregate(
            np.mean).reset_index().sort_values("score")

        subtitle = f"Mentions Opponent: {ment_key}" if ment_key != "all" else "All FW Overlap"
        subtitle = f"[{' & '.join(pos_tags)}] " + subtitle if pos_tags != ["ALL"] else subtitle
        sns.barplot(data= ment_df,
                    x="speaker", 
                    y="score", 
                    hue=hue, 
                    palette=p,
                    dodge=False,
                    order=grouping["speaker"],
                    ax=axes[i] if len(ment_keys) > 1 else axes,
                    )\
        .set_title(subtitle)
        
        if len(ment_keys) > 1:
            axes[i].tick_params("x", labelrotation=90)
            axes[i].set(ylabel=f"Top-{n} FW Overlap Score" if i == 0 else None, 
                        xlabel=None)
            if i != len(ment_keys) - 1: 
                axes[i].get_legend().remove()
            else:
                axes[i].get_legend().set_title("party")
                sns.move_legend(axes[i], "upper left", bbox_to_anchor=(1.1, 0.75), frameon=False)
        else:
            axes.tick_params("x", labelrotation=90)
            axes.set(ylabel=f"Top-{n} FW Overlap Score", 
                        xlabel="Speaker")
            axes.get_legend().set_title("party")
            sns.move_legend(axes, "upper left", bbox_to_anchor=(1.1, 0.75), frameon=False)

    if len(ment_keys) > 1:
        fig.tight_layout(rect=(0.025,0.05,1,1))
        fig.supxlabel(f"Speaker")    
    
        
    if overwrite:
        sharedy = f"_unshared" if not sharey else ""
        title = "_".join(title.lower().split())
        pos_tag_suffix = "_" + "_".join(pos_tags)
        plt.savefig(f"{fig_dir}/fig_{title.lower()}_bar{n}_overlap_{len(ment_keys)}_side{sharedy}{pos_tag_suffix}.pdf", 
                    bbox_inches="tight")  
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

    
    args = parser.parse_args()

    data_types = [dt.strip() for dt in args.data_types.split(",")]
    orig_text_field = "text_orig_masked" if args.mask_ents else "text_orig"
    fig_dir = args.fig_dir

    lm_datas = {}
    data_speakers = {}
    metadatas = {}
    spk_dfs = {}
    
    dt_fw = {}
    dt_ctr = {}

    # load data and calculate FW rates
    for data_type in data_types:
        data_dir = os.path.join(args.data_dir, data_type)
        data, speakers, metadata = load_data_and_speakers(data_dir, data_type, load_metadata=True)
 
        data["data_type"] = data_type
        data["type"] = data["mentions_opponent"]
        data.rename(columns={"speaker_clean": "speaker"}, inplace=True)
        data["is_trump"] = data["speaker"] == "Donald Trump"  
        data["num_words"] = data[orig_text_field].apply(lambda x: len(x.split()))
        spk_df = data[data["speaker"].isin(data_speakers[data_type].keys())]

        lm_datas[data_type] = data
        data_speakers[data_type] = speakers
        metadatas[data_type] = metadata
        spk_dfs[data_type] = spk_df
   
        
        fw_list, counter_dict = calc_fightin_words_by_speaker(spk_df, speakers, orig_text=orig_text_field, pos_suffix=True)
        dt_fw[dt] = fw_list
        dt_ctr[dt] = counter_dict

    for pos_tags in [["ADJ"]]: #[["ALL"], ["ADJ"], ["ADV"], ["ADJ", "ADV"]]:
        print(pos_tags)
        for ment_key in ["all", "Y", "N"]:
            for errbar in [('ci', 95), None]:
                fw_plot(fig_dir, data_types, dt_fw, metadatas, ment_key=ment_key, errbar=errbar, pos_tags=pos_tags)

        for dt in data_types:
            dt_fig_dir = os.path.join(fig_dir, dt)
        
            for ment_keys in [["all", "Y", "N"], ["Y", "N"], ["Y"]]:
                for n in [5, 10, 15, 25]:
                    fw_bar_plot_by_data(dt_fig_dir, n, dt, dt_fw[dt], metadatas[dt], ment_keys=ment_keys, pos_tags=pos_tags)

                for errbar in [('ci', 95), None]:
                    fw_plot_by_data(dt_fig_dir, dt, dt_fw[dt], metadatas[dt], errbar=errbar, ment_keys=ment_keys, pos_tags=pos_tags)



if __name__ == "__main__":
    main()

