import argparse
import os
from pathlib import Path

import numpy as np
from utils import load_data_and_speakers, data_with_intersected_speakers, assign_normalized_sent_ids
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib


SNS_STYLE = "ticks"
sns.set_theme(font_scale=2.0, 
            style=SNS_STYLE, 
              rc={
#    "lines.linewidth": 5,
   "lines.markersize":20,
   "ps.useafm": True,
   "font.sans-serif": ["Helvetica"],
   "pdf.use14corefonts" : True,
    "axes.unicode_minus": False,
   })


def aggregate_plots(
    fig_dir,
    data_types, 
    datas, 
    data_speakers, 
    metadatas, 
    spk_dfs, 
    use_intersect=True,
    use_bpc=True
    ):
    lm_datas, spk_intersect, metadata_intersect, spk_dfs = data_with_intersected_speakers(datas, data_speakers, metadatas, spk_dfs, use_intersect)
    spk_df = pd.concat(spk_dfs.values())


    intersect_suffix ="_union" if not use_intersect else ""

    ## additional filtering
    spk_df = spk_df[(spk_df["data_type"] != "sotu") | (spk_df["sent_id"] != 0)]  # remove first sentence if sotu


    ### score by party/is_trump
    def score_by_party(spk_df, field="sent_uniq_bpc", title="Uniqueness", with_n=False, sharey=True, overwrite=True):
        print(f"Plotting {title} by sentence length")
        c = sns.color_palette("pastel").as_hex()
        p = {'D': c[0], 'R':c[3], 'Trump': c[1]}
 
        legend_labels = ["Dem", "Rep\n(excl. Trump)", "Trump"]
        
        h = 6
        fig, axes = plt.subplots(1, len(data_types), figsize=(len(data_types) * h * 3/4, h), sharey=sharey)

        for i, dt in enumerate(data_types):
            # if not use_intersect:
            #     spk_df = spk_dfs[dt]
            ppl_df = spk_df[spk_df["data_type"] == dt]
            ppl_df["party"] = ppl_df.apply(lambda row: "Trump" if row["is_trump"] else row["party"], axis=1)
                
            all_df = ppl_df[["party", field, "type"]]
            all_df["type"] = "all"

            df = pd.concat([all_df, ppl_df])

            g = sns.barplot( data=ppl_df, x ="party", 
                        y = field,
                        # hue='type',
                        dodge=False, 
                        palette=p,
                        order= ["D", "R", "Trump"],
                        ax=axes[i]
                    ) #\
                # .set(title=f'[{data_type.upper()}] Average Sentence {title} by Party')

            axes[i].set(ylabel=f"Sentence {title}" if i == 0 else None, xlabel=None)
            axes[i].set_xticklabels(["Dem", "Rep\n(excl.\nTrump)", "Trump"])
            axes[i].set(title=f"[{dt.upper()}]")

            if with_n:
                party_counts = {}
                for party, group in ppl_df.groupby("party"):
                    # unique_speakers = group["speaker"].nunique()
                    party_counts[party] = len(group)
                
                new_labels = []
                for label, count in zip(legend_labels, party_counts.values()):
                    label = f"n={count}"
                    new_labels.append(label)
        
                # h, _ = g.get_legend_handles_labels()
                # g.legend(handles=h, labels=new_labels, loc="best")
                axes[i].bar_label(axes[i].containers[0], new_labels, label_type="edge", fontsize=14, padding=-30)

        
        fig.tight_layout(rect=(0.025,0,1,1))
        # fig.supxlabel("Party")

        if overwrite:
            sharedy = f"_unshared" if not sharey else ""
            legend_suffix = "_with_n" if with_n else ""
            plt.savefig(os.path.join(fig_dir, f"fig_agg_{title.lower()}_by_party{sharedy}{intersect_suffix}{legend_suffix}.pdf"), 
                    bbox_inches="tight")
            plt.close(fig)

    ### score by sentence length
    def score_by_sent_len(spk_df, field="sent_uniq_bpc", title="Uniqueness", sharey=True, overwrite=True):
        print(f"Plotting {title} by sentence length")
        c = sns.color_palette("pastel").as_hex()
        p = {'D': c[0], 'R':c[3]}
        
        for xlims in [(0, 150), (0, 50), (10,50)]:
            h = 6
            fig, axes = plt.subplots(1, len(data_types), figsize=(len(data_types) * h, h * 3/4), sharey=sharey)

            for i, dt in enumerate(data_types):
                ppl_df = spk_df[spk_df["data_type"] == dt]
            
                sns.lineplot(data=ppl_df, x="num_words", y=field,
                            ax=axes[i], label="All", color=c[4], linestyle="dotted")
                sns.lineplot(data=ppl_df[ppl_df["party"] == "D"], x="num_words", y=field,
                             ax=axes[i], label="Dem", color=c[0],  linestyle="dashed")
                sns.lineplot(data=ppl_df[(ppl_df["party"] == "R") & (ppl_df["is_trump"] == False)],
                            x="num_words", y=field, ax=axes[i], label="Rep (excl. Trump)", color=c[3],
                            linestyle="dashdot")
                sns.lineplot(data=ppl_df[ppl_df["is_trump"] == True], x="num_words", y=field, 
                            ax=axes[i], label="Trump", color=c[1], linestyle="solid")
        
                axes[i].set(title=f"[{dt.upper()}]")
                axes[i].set(ylabel=f"Sentence {title}" if i == 0 else None, xlabel=None)

                if i != len(data_types) - 1: 
                    axes[i].get_legend().remove()
                else:
                    axes[i].get_legend().set_title("party")
                    sns.move_legend(axes[i], "upper left", bbox_to_anchor=(1.1, 0.75), frameon=False)

            for ax in axes:
                ax.set(xlim=xlims)

            fig.tight_layout(rect=(0.025,0.025,1,1))
            fig.supxlabel("Sentence Length")
                    

            if overwrite:
                xlims = f"{xlims[0]}_{xlims[1]}"
                sharedy = f"unshared_" if not sharey else ""
                plt.savefig(os.path.join(fig_dir, f"fig_agg_{title.lower()}_over_sent_len_{sharedy}{xlims}{intersect_suffix}.pdf"), 
                        bbox_inches="tight")
                plt.close(fig)

    ### score by speaker
    def overall_score(spk_df, field="sent_uniq_bpc", title="Uniqueness", ment_key="all", sharey=True, overwrite=True):
        print(f"Overall {field}")
        c = sns.color_palette("pastel").as_hex()
        p = {'D': c[0], 'R':c[3]}


        h = 8
        fig, axes = plt.subplots(1, len(data_types), figsize=(len(data_types) * h, h), sharey=sharey)

        for i, dt in enumerate(data_types):
            # if not use_intersect:
            #     spk_df = spk_dfs[dt]

            ppl_df = spk_df[spk_df["data_type"] == dt]

            if ment_key != "all":
                ppl_df = ppl_df[ppl_df["type"] == ment_key]

            grouping = ppl_df.groupby(["speaker"])[field].aggregate(np.mean).reset_index().sort_values(field)

            sns.barplot( data=ppl_df, x ="speaker", y = field,
                hue='party',
                        dodge=False, 
                        palette=p,
                        order=grouping["speaker"],
                        ax=axes[i]
                    ) #\
                # .set(title=f'[{data_type.upper()}] Average Sentence {title} by Candidate')
            axes[i].tick_params("x", labelrotation=90) 
            axes[i].set(title=f"[{dt.upper()}]")
            ylab = f"# Words in Sentence" if field == "num_words" else f"Sentence {title}"
            axes[i].set(ylabel=ylab if i == 0 else None, xlabel=None)
            # for tick in axes[i].get_xticklabels():
            #     tick.set_rotation("x", 90)

            if i != len(data_types) - 1: 
                axes[i].get_legend().remove()
            else:
                axes[i].get_legend().set_title("party")
                sns.move_legend(axes[i], "upper left", bbox_to_anchor=(1.1, 0.75), frameon=False)

            fig.tight_layout(rect=(0.025,0,1,1))
            fig.supxlabel("Speaker")

        if overwrite:
            sharedy = f"_unshared" if not sharey else ""
            plt.savefig(os.path.join(fig_dir, f"fig_agg_overall_{title.lower()}_{ment_key.lower()}{sharedy}{intersect_suffix}.pdf"), 
                        bbox_inches="tight")  
            plt.close(fig)


    ### rate of opp ments
    def rate_opp_ment(spk_df, data_types, field="opp_mention_rate", sharey=True, overwrite=True):
        c = sns.color_palette("pastel").as_hex()
        p = {'D': c[0], 'R':c[3]}

        h = 8
        fig, axes = plt.subplots(1, 
                                len(data_types), 
                                figsize=(len(data_types) * h, h), 
                                sharey=sharey)
        
        def calculate_opp_mention_rate(df):
            # Group by "speaker" and calculate the percentage of "mentions_opponent" = Y
            grouped_df = df.groupby(["speaker"])["mentions_opponent"].value_counts(normalize=True).unstack().fillna(0)
            percent_sents_mention = (grouped_df["Y"] * 100).reset_index()
            percent_sents_mention.rename(columns={'Y': field}, inplace=True)

            return percent_sents_mention
        
        # spk_df = lm_data[lm_data["speaker"].isin(spk_intersect.keys())]

        for i, dt in enumerate(data_types):
            # if not use_intersect:
            #     ppl_df = spk_dfs[dt]
            # else:
            ppl_df = spk_df[spk_df["data_type"] == dt]
            ppl_df = calculate_opp_mention_rate(ppl_df)
            spk_data = metadatas[dt]
            ppl_df["party"] = ppl_df["speaker"].apply(lambda x: spk_data[x]["party"] if x in spk_data else None)

            # spk_df["num_words"] = spk_df["text"].apply(lambda x: len(x.split()))
            # ppl_df = result_df[result_df["data_type"] == dt]
            grouping = ppl_df.groupby(
                ["speaker"])[field].aggregate(
                np.mean).reset_index().sort_values(
                field)
            sns.barplot( data=ppl_df, 
                        x ="speaker", y = field,
                        hue='party',
                                dodge=False, 
                                palette=p,
                                order=grouping["speaker"],
                                ax=axes[i]
                            ) #\
                        # .set(title=f'[{data_type.upper()}] Average Sentence {title} by Candidate')
            axes[i].tick_params("x", labelrotation=90) 
            axes[i].set(title=f"[{dt.upper()}]")
            axes[i].set(ylabel=f"Rate (%) of Opponent Mentions" if i == 0 else None, xlabel=None)
            # for tick in axes[i].get_xticklabels():
            #     tick.set_rotation("x", 90)

            if i != len(data_types) - 1: 
                axes[i].get_legend().remove()
            else:
                axes[i].get_legend().set_title("party")
                sns.move_legend(axes[i], "upper left", bbox_to_anchor=(1.1, 0.75), frameon=False)

            fig.tight_layout(rect=(0.025,0,1,1))
            fig.supxlabel("Speaker")

        if overwrite:
            sharedy = f"_unshared" if not sharey else ""
            plt.savefig(os.path.join(fig_dir, f"fig_agg_rate_ments{sharedy}{intersect_suffix}.pdf"), 
                                bbox_inches="tight")
            plt.close(fig)


    def score_over_time(spk_df, field="sent_uniq_bpc", title="Uniqueness", sharey=True, overwrite=True):
        c = sns.color_palette("pastel").as_hex()
        p = {'all': "lightgrey", 'D': c[0], 'R':c[3]}

        h = 6
        fig, axes = plt.subplots(1, len(data_types), figsize=(len(data_types) * h, h), sharey=sharey)

        for i, dt in enumerate(data_types):
            # if not use_intersect:
            #     spk_df = spk_dfs[dt]

            ppl_df = spk_df[spk_df["data_type"] == dt]

            all_df = ppl_df[["party", "year", field]]
            all_df["party"] = "all"

            df = pd.concat([all_df, ppl_df])
            df = df.reset_index()

            sns.lineplot(data=df, x="year", y=field, hue="party", palette=p, 
                         style="party", 
                     style_order=p.keys(), 
                    linewidth=2,
                    ax=axes[i])
            axes[i].set(title=f"[{dt.upper()}]")
            ylab = f"Sentence {title}" if i == 0 else None
            axes[i].set(ylabel=ylab, xlabel=None)
            # for tick in axes[i].get_xticklabels():
            #     tick.set_rotation("x", 90)
            plot_range = range(2008, 2020+2, 4) if dt == "campaign" else range(1960, 2020+2, 20) 
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
            sharedy = f"_unshared" if not sharey else ""
            plt.savefig(os.path.join(fig_dir, f"fig_agg_{title.lower()}_over_time{sharedy}{intersect_suffix}.pdf"), 
                        bbox_inches="tight")  
            plt.close(fig)



  
    score_over_time(spk_df, "sent_uniq_bpc", "Uniqueness")
    score_over_time(spk_df, "sent_uniq_bpc", "Uniqueness", sharey=False)
    score_over_time(spk_df, "bpc", "BPC")
    score_over_time(spk_df, "bpc", "BPC", sharey=False)
    
    score_by_sent_len(spk_df, "sent_uniq_bpc", "Uniqueness")

    ## generate plots
    score_by_party(spk_df, "sent_uniq_bpc", "Uniqueness")
    score_by_party(spk_df, "sent_uniq_bpc", "Uniqueness", with_n=True)

    score_by_sent_len(spk_df, "sent_uniq_bpc", "Uniqueness", sharey=False)

    score_by_party(spk_df, "sent_uniq_bpc", "Uniqueness", sharey=False)
    score_by_party(spk_df, "sent_uniq_bpc", "Uniqueness", with_n=True, sharey=False)

    rate_opp_ment(spk_df, data_types)
    rate_opp_ment(spk_df, data_types, sharey=False)

    for ment_key in ["all", "Y", "N"]:
        overall_score(spk_df, "sent_uniq_bpc", "Uniqueness", ment_key)

        overall_score(spk_df, "sent_uniq_bpc", "Uniqueness", ment_key, sharey=False)

        overall_score(spk_df, "num_words", "Length", ment_key)
        overall_score(spk_df, "num_words", "Length", ment_key, sharey=False)

        overall_score(spk_df, "bpc", "BPC", ment_key)
        overall_score(spk_df, "bpc", "BPC", ment_key, sharey=False)

    score_by_sent_len(spk_df, "bpc", "BPC")
    score_by_party(spk_df, "bpc", "BPC")
    score_by_party(spk_df, "bpc", "BPC", with_n=True)
    score_by_sent_len(spk_df, "bpc", "BPC", sharey=False)
    score_by_party(spk_df, "bpc", "BPC", sharey=False)
    score_by_party(spk_df, "bpc", "BPC", with_n=True, sharey=False)


def robustness_plots(fig_dir, data_type, spk_df, speakers, spk_data, use_intersect=True, thresh=0):
    if thresh > 0:
        fig_dir = f"{fig_dir}/threshold_{thresh}"

    
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    t_suffix = f" (sent_len ≥ {thresh})" if thresh > 0 else ""


    ## additional filtering
    if data_type == "sotu":
        spk_df = spk_df[(spk_df["sent_id"] != 0)]
    #     data = data[(data["sent_id"] != 0)]  # remove first sentence if sotu

    # lm_data = get_uniqueness_w_speaker_subset(lm_data, speakers)

    elec_years = [1932, 1948, 1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020]

    terms_map = {}
    for spk, v in spk_data.items():
        # name = v["clean"]
        spk_terms = [spk+"|"+str(x) for x in v["year"]]
        for i, spk_term in enumerate(spk_terms):
            terms_map[spk_term] = f"{spk} [{i+1}]"


    def determine_term(row, data_type):
    
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


    def overall_score(
            spk_df, 
            field="sent_uniq_bpc", 
            title="Uniqueness", 
            x="spk_term",
            xlabel="Speaker|Term",
            ment_key="all", 
            overwrite=True):
        print(f"Overall {field}")
        c = sns.color_palette("pastel").as_hex()
        p = {'D': c[0], 'R':c[3]}
        
        # spk_df = df[df["speaker"].isin(speakers)]

        # Add the "term" column to the DataFrame
        spk_df['term'] = spk_df.apply(lambda x: determine_term(x, data_type), axis=1)
        spk_df['term'] = spk_df['term'].astype(int)
        spk_df = spk_df[spk_df["term"] != -100]
        
        if x == "spk_yr":
            spk_df = spk_df[spk_df["year"] >= 2000] # only plot from 2000 onwards
        spk_df["spk_term"] = spk_df["speaker"] + "|" + spk_df["term"].astype(str)
        spk_df["spk_term_order"] = spk_df["spk_term"].apply(lambda x: terms_map.get(x, None))
        if x == "spk_term_order":
            spk_df = spk_df[spk_df["spk_term_order"].notnull()]
        spk_df["spk_yr"] = spk_df["speaker"] + "|" + spk_df["year"].astype(str)
        spk_df["party"] = spk_df["speaker"].apply(lambda x: spk_data[x]["party"])    

        if ment_key != "all":
            spk_df = spk_df[spk_df["type"] == ment_key]

        grouping = spk_df.groupby([x])[field].aggregate(np.mean).reset_index().sort_values(field)

        fig, ax = plt.subplots()
        sns.barplot( data=spk_df, x =x, y = field,
            hue='party',
                    dodge=False, 
                    palette=p,
                    order=grouping[x],
                    ax=ax
                ) \
            .set(title=f'[{data_type.upper()}]',
            ylabel=f"Sentence {title}",
            xlabel=xlabel)
        l = plt.xticks(rotation=90) 

        if overwrite:
            intersect_suffix ="_union" if not use_intersect else ""
            plt.savefig(os.path.join(fig_dir, f"fig_{data_type}_{x}_overall_{title.lower()}{intersect_suffix}.pdf"), 
                                bbox_inches="tight")
            plt.close(fig)


    def score_by_mention_shared(spk_df, field="sent_uniq_bpc", title="Uniqueness", sharey=True, overwrite=True):
        print(f"{title} by mention, shared plot")
        c = sns.color_palette("pastel").as_hex()
        # print(c)
        p = {'D': c[0], 'R':c[3]}
        # print(p)

        h = 8
        fig, axes = plt.subplots(1, 2, figsize=(2 * h, h), sharey=sharey)

        ment_keys = ["Y", "N"]
        
        # print(spk_df)
    
        for i, ment_key in enumerate(ment_keys):
            # spk_df = lm_data[lm_data["speaker"].isin(speakers)]
            tmp_df = spk_df[spk_df["type"] == ment_key]
            print(ment_key, len(spk_df))

            grouping = tmp_df.groupby(["speaker"])[field].aggregate(np.mean).reset_index().sort_values(field)

            sns.barplot( data=tmp_df, x ="speaker", y = field,
                hue='party',
                        dodge=False, 
                        palette=p,
                        order=grouping["speaker"],
                        ax=axes[i]
                    ) #\
                # .set(title=f'[{data_type.upper()}] Average Sentence {title} by Candidate')
            axes[i].tick_params("x", labelrotation=90) 
            axes[i].set(title=f"Mentions Opponent: {ment_key}")
            ylab = f"Sentence {title}"
            axes[i].set(ylabel=ylab if i == 0 else None, xlabel=None)
            # for tick in axes[i].get_xticklabels():
            #     tick.set_rotation("x", 90)

            if i != len(ment_keys) - 1: 
                axes[i].get_legend().remove()
            else:
                axes[i].get_legend().set_title("party")
                sns.move_legend(axes[i], "upper left", bbox_to_anchor=(1.1, 0.75), frameon=False)

            fig.tight_layout(rect=(0.025,0,1,1))
            fig.supxlabel("Speaker")

        if overwrite:
            sharedy = f"_unshared" if not sharey else ""
            intersect_suffix ="_union" if not use_intersect else ""
            plt.savefig(os.path.join(fig_dir, f"fig_side_mentions_{title.lower()}{sharedy}{intersect_suffix}.pdf"), 
                        bbox_inches="tight")  
            plt.close(fig)


    ### Function Calls ###
    score_by_mention_shared(spk_df, "sent_uniq_bpc", "Uniqueness")
    score_by_mention_shared(spk_df, "sent_uniq_bpc", "Uniqueness", sharey=False)
    
    overall_score(spk_df, field="sent_uniq_bpc", title="Uniqueness", x="spk_term", xlabel="Speaker|Term", ment_key="all")
    overall_score(spk_df, field="sent_uniq_bpc", title="Uniqueness", x="spk_term_order", xlabel="Speaker [Term #]", ment_key="all")
    if data_type == "sotu":
        overall_score(spk_df, field="sent_uniq_bpc", title="Uniqueness", x="spk_yr", xlabel="Speaker|Year", ment_key="all")


def generate_plots_from_df(fig_dir, data_type, data, speakers, spk_data, use_bpc=True, thresh=0):
    '''
    `thresh` is the threshold for sentence length
    '''
    if thresh > 0:
        fig_dir = f"{fig_dir}/threshold_{thresh}"

    
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    t_suffix = f" (sent_len ≥ {thresh})" if thresh > 0 else ""


    data["type"] = data["mentions_opponent"]
    data.rename(columns={"speaker_clean": "speaker"}, inplace=True)
    data["is_trump"] = data["speaker"] == "Donald Trump"  

    ## additional filtering
    if data_type == "sotu":
        data = data[(data["sent_id"] != 0)]  # remove first sentence if sotu

    spk_df = data[data["speaker"].isin(speakers)]
    spk_df["party"] = spk_df["speaker"].apply(lambda x: spk_data[x]["party"])


    def score_over_sent_len(spk_df, field="sent_uniq_bpc", title="Uniqueness", overwrite=True):
        print(f"{title} Over Sent Len")
        c = sns.color_palette("pastel").as_hex()

        for xlims in [(0, 150, 50), (0, 50, 10), (10,50, 10)]:
            fig, ax = plt.subplots(figsize=(6,6))

            sns.lineplot(data=spk_df, x="num_words", y=field,
                            ax=ax, label="All", color=c[4], linestyle="dotted", 
                            # marker="0",
                            # markersize=10
                 )
            sns.lineplot(data=spk_df[spk_df["party"] == "D"], x="num_words", y=field,
                            ax=ax, label="Dem", color=c[0],  linestyle="dashed", 
                            # marker="v",
                            # markersize=10
                 )
            sns.lineplot(data=spk_df[(spk_df["party"] == "R") & (spk_df["is_trump"] == False)],
                        x="num_words", y=field, ax=ax, label="Rep (excl. Trump)", color=c[3],
                        linestyle="dashdot", 
                        # marker="s",
                        # markersize=10
                 )
            sns.lineplot(data=spk_df[spk_df["is_trump"] == True], x="num_words", y=field, 
                            ax=ax, label="Trump", color=c[1], linestyle="solid", 
                            # marker="D",
                            # markersize=10
                 )


            # for ax in axes:
            ax.set(xlim=(xlims[0], xlims[1]))
            ax.set_xticks(range(xlims[0], xlims[1]+1, xlims[2]))
            # ax.set(ylim=(1.5, 4))
            ax.set(xlabel="Sentence Length", ylabel=f"Sentence {title}", title=f"[{data_type.upper()}]")

            if overwrite:
                xlims = f"{xlims[0]}_{xlims[1]}"
                plt.savefig(os.path.join(fig_dir, f"fig_{title.lower()}_sent_len_{xlims}.pdf"), 
                            bbox_inches="tight")
            
            plt.close(fig)

    ### AVG Uniqueness BY PARTY ###
    def avg_score_party(spk_df, field="sent_uniq_bpc", title="Uniqueness", overwrite=True):
        print(f"Avg {title} Party")
        c = sns.color_palette("pastel").as_hex()
        # p = {'D': c[0], 'R':c[3]}

        # print(ppl_df.info())

        all_df = spk_df[["party", field, "type"]]
        all_df["type"] = "all"

        fig, ax = plt.subplots()
        df = pd.concat([all_df, spk_df])
        sns.barplot( data=df, x ="party", 
                    y = field,
            hue='type',
                    # dodge=False, 
                    palette=c,
                    ax=ax
                ) \
            .set(title=f'[{data_type.upper()}]',
                 ylabel=f"Sentence {title}",
                 xlabel="Party")

        
        if overwrite:
            plt.savefig(os.path.join(fig_dir, f"fig_avg_{title.lower()}_party.pdf"), 
                        bbox_inches="tight")
            plt.close(fig)
        
    def overall_score(spk_df, field="sent_uniq_bpc", title="Uniqueness", ment_key="all", overwrite=True):
        print(f"Overall {field}")
        c = sns.color_palette("pastel").as_hex()
        p = {'D': c[0], 'R':c[3]}

        if ment_key != "all":
            spk_df = spk_df[spk_df["type"] == ment_key]

        grouping = spk_df.groupby(["speaker"])[field].aggregate(np.mean).reset_index().sort_values(field)

        fig, ax = plt.subplots()
        sns.barplot( data=spk_df, x ="speaker", y = field,
            hue='party',
                    dodge=False, 
                    palette=p,
                    order=grouping["speaker"],
                    ax=ax
                ) \
            .set(title=f'[{data_type.upper()}]',
            ylabel=f"Sentence {title}",
            xlabel="Speaker")
        l = plt.xticks(rotation=90) 

        if overwrite:
            plt.savefig(os.path.join(fig_dir, f"fig_overall_{title.lower()}_{ment_key.lower()}.pdf"), 
                        bbox_inches="tight")
            plt.close(fig)

    def x_over_time(data, field="sent_uniq_bpc", title="Uniqueness", overwrite=True):
        print("{} Over Time".format(title))
        c = sns.color_palette("pastel").as_hex()
        p = {'D': c[0], 'R':c[3]}

        spk_df = assign_normalized_sent_ids(data, speakers)
        spk_df["party"] = spk_df["speaker"].apply(lambda x: spk_data[x]["party"])

        sns.lmplot( data=spk_df[spk_df["num_words"] >= thresh], x ="sent_id", y=field, x_bins=10,
                ).set(title=f'[{data_type.upper()}] Average Sentence {title} Throughout Speech{t_suffix}',
                    xlabel=f"progress through speech")
        
        if overwrite:
            plt.savefig(os.path.join(fig_dir, f"fig_{title.lower()}_over_time.pdf"), 
                        bbox_inches="tight")

        sns.lmplot( data=spk_df[spk_df["num_words"] >= thresh], x="sent_id", y=field, x_bins=10,
                hue='party',
                palette=p,
            ).set(title=f'[{data_type.upper()}] Average Sentence {title} Throughout Speech{t_suffix}',
                    xlabel=f"progress through speech")
        
        if overwrite:
            plt.savefig(os.path.join(fig_dir, f"fig_{title.lower()}_over_time_by_party.pdf"), 
                        bbox_inches="tight")
        
        sns.lmplot( data=spk_df[spk_df["num_words"] >= thresh], x ="sent_id", y=field, x_bins=10,
                hue='is_trump',
            ).set(title=f'[{data_type.upper()}] Average Sentence {title} Throughout Speech{t_suffix})',
                xlabel=f"progress through speech",
                ylabel=f"Sentence {title}")
        
        if overwrite:
            plt.savefig(os.path.join(fig_dir, f"fig_{title.lower()}_over_time_by_trump.pdf"), 
                        bbox_inches="tight")


    def score_by_mention(spk_df, field="sent_uniq_bpc", title="Uniqueness",overwrite=True):
        print(f"{title} by mention")
        c = sns.color_palette("pastel").as_hex()
        # p = {'D': c[0], 'R':c[3]}

        fig, ax = plt.subplots()
        grouping = spk_df.groupby(["speaker"])[field].aggregate(np.mean).reset_index().sort_values(field)

        sns.barplot( data=spk_df[(spk_df["type"]=="N") | 
                                (spk_df["type"]=="Y")].sort_values(field), 
                    x ="speaker", y =field,
            hue='type',
                    dodge=False, 
                    order=grouping["speaker"],
                    palette=c
                    
                ) \
            .set(title=f'[{data_type.upper()}]',
                 ylabel=f"Sentence {title}",
                xlabel="Speaker")
        l = plt.xticks(rotation=90) 

        if overwrite:
            plt.savefig(os.path.join(fig_dir, f"fig_{title.lower()}_by_mention.pdf"),  
                        bbox_inches="tight")
            plt.close(fig)



    score_over_sent_len(spk_df, "sent_uniq_bpc", "Uniqueness")
    score_over_sent_len(spk_df, "bpc", "BPC")

    # ## generate plots
    
    avg_score_party(spk_df, "sent_uniq_bpc", "Uniqueness")
    avg_score_party(spk_df, "bpc", "BPC")

    for ment_key in ["all", "Y", "N"]:
        overall_score(spk_df, "sent_uniq_bpc", "Uniqueness", ment_key)
        overall_score(spk_df, "bpc", "BPC", ment_key)

    # # # uniqueness by party
    # # x_over_time(field="bpc", title="BPC")
    # # x_over_time(field="sent_uniq_bpc", title="Uniqueness")

    score_by_mention(spk_df, "sent_uniq_bpc", "Uniqueness")
    score_by_mention(spk_df, "bpc", "BPC")





def main():
    code_dir = os.path.dirname(os.path.abspath(__file__))
    working_dir = os.path.dirname(code_dir)

    parser = argparse.ArgumentParser(description="Calculate uniqueness after finetuning your model")
    parser.add_argument('--model_type', 
                        type=str, 
                        choices=["gpt2", "gemma2b", "phi1-5b"], 
                        required=True,
                        help="gpt2 (original) or gemma2b (validation)")
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
    agg_figure_dir = os.path.join(args.figure_dir, f"{args.model_type}_{len(data_types)}")

    orig_text_field = "text_orig_masked" if args.mask_ents else "text_orig"


    lm_datas = {}
    data_speakers = {}
    metadatas = {}
    spk_dfs = {}
    for data_type in data_types:
        data_dir = os.path.join(args.data_dir, data_type)
        data, speakers, metadata = load_data_and_speakers(data_dir, data_type, model_type=args.model_type, load_metadata=True)
 
        data["data_type"] = data_type
        data["type"] = data["mentions_opponent"]
        data.rename(columns={"speaker_clean": "speaker"}, inplace=True)
        data["is_trump"] = data["speaker"] == "Donald Trump"  
        data["num_words"] = data[orig_text_field].apply(lambda x: len(x.split()))

        lm_datas[data_type] = data
        data_speakers[data_type] = speakers
        metadatas[data_type] = metadata
        spk_dfs[data_type] = data[data["speaker"].isin(data_speakers[data_type].keys())]

    ## Aggregate plots (Data types side-by-side)
    aggregate_plots(agg_figure_dir, data_types, lm_datas, data_speakers, metadatas, spk_dfs, use_intersect=False)
    aggregate_plots(agg_figure_dir, data_types, lm_datas, data_speakers, metadatas, spk_dfs, use_intersect=True)

    ## individual plots
    intersect_datas, intersect_speakers, intersect_metadatas, intersect_spk_dfs = data_with_intersected_speakers(lm_datas, data_speakers, metadatas, spk_dfs, use_intersect=True)

    for data_type in data_types:
        data = lm_datas[data_type]
        speakers = data_speakers[data_type]
        spk_data = metadatas[data_type]
        spk_df = spk_dfs[data_type]

        fig_dir = os.path.join(args.figure_dir, f"{args.model_type}_{data_type}")

        robustness_plots(fig_dir, data_type, spk_df, speakers, spk_data, use_intersect=False, thresh=0)
        if data_type != "campaign":
            intersect_spk_df = intersect_spk_dfs[data_type]
            spk_intersect = intersect_speakers[data_type]
            metadata_intersect = intersect_metadatas[data_type]
            
            intersect_spk_df = intersect_spk_dfs[data_type]
            robustness_plots(fig_dir, data_type, intersect_spk_df, spk_intersect, metadata_intersect, use_intersect=True, thresh=0)

        for threshold in [0]: #[0, 5, 10]:
            generate_plots_from_df(fig_dir, data_type, data, speakers, spk_data, thresh=threshold)



if __name__ == "__main__":
    main()
