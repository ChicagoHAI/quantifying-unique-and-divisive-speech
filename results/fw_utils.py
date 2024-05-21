import numpy as np
import pandas as pd
from statistics import mean
import spacy
from collections import Counter

from operator import itemgetter
from sklearn.feature_extraction.text import CountVectorizer as CV
import string

from matplotlib import pyplot as plt

nlp = spacy.load('en_core_web_sm')
exclude = set(string.punctuation)


def basic_sanitize(in_string):
    '''Returns a very roughly sanitized version of the input string.'''  
    in_string = ''.join([ch for ch in in_string if ch not in exclude])
    in_string = in_string.lower()
    in_string = ' '.join(in_string.split())
    return in_string

def bayes_compare_language_from_counter(c1, c2, prior=.01, filter_short=True):
    '''
    Arguments:
    - c1, c2; counters from each sample
    - prior; either a float describing a uniform prior, or a dictionary describing a prior
    over vocabulary items. If you're using a predefined vocabulary, make sure to specify that
    when you make your CountVectorizer object.

    Returns:
    - A list of length |Vocab| where each entry is a (n-gram, zscore) tuple.'''
    vocab = set(c1.keys()) | set(c2.keys())
    if type(prior) is float:
        priors = {w: prior for w in vocab}
    else:
        priors = prior
    z_scores = {}
    a0 = sum(priors.values())
    n1 = sum(c1.values())
    n2 = sum(c2.values())
    print("Comparing language...")
    for w in vocab:
        #compute delta
        w1, w2, wp = c1.get(w, 0), c2.get(w, 0), priors[w]
        term1 = np.log((w1 + wp) / (n1 + a0 - w1 - wp))
        term2 = np.log((w2 + wp) / (n2 + a0 - w2 - wp))        
        delta = term1 - term2
        #compute variance on delta
        var = 1. / (w1 + wp) + 1. / (w2 + wp)
        #store final score
        z_scores[w] = delta / np.sqrt(var)
    return_list = [(w, z_scores[w]) for w in vocab]
    return_list.sort(key=itemgetter(1))
    if filter_short:
        return_list = [(w, s) for (w, s) in return_list if len(w) > 1]
    # print("words most associated with the first corpus")
    # for (w, s) in return_list[-20:][::-1]:
    #     print(w, s)
    # print("words most associated with the second corpus")
    # for (w, s) in return_list[:20]:
    #     print(w, s)
    return return_list


def plot_bayes_compare_language_from_counter(c1, c2, prior=.01, word_size=4, filter_short=True, sig_val=2.573, DATA_PATH=None):
    '''
    Plot fightin words, given COUNTERS of words in corpora

    Arguments:
    - c1, c2; counters from each sample
    - prior; either a float describing a uniform prior, or a dictionary describing a prior
    over vocabulary items. If you're using a predefined vocabulary, make sure to specify that
    when you make your CountVectorizer object.

    Returns:
    - A list of length |Vocab| where each entry is a (n-gram, zscore) tuple.'''
    vocab = set(c1.keys()) | set(c2.keys())
    if type(prior) is float:
        priors = {w: prior for w in vocab}
    else:
        priors = prior
    z_scores = {}
    a0 = sum(priors.values())
    n1 = sum(c1.values())
    n2 = sum(c2.values())
    print("Comparing language...")
    for w in vocab:
        #compute delta
        w1, w2, wp = c1.get(w, 0), c2.get(w, 0), priors[w]
        term1 = np.log((w1 + wp) / (n1 + a0 - w1 - wp))
        term2 = np.log((w2 + wp) / (n2 + a0 - w2 - wp))        
        delta = term1 - term2
        #compute variance on delta
        var = 1. / (w1 + wp) + 1. / (w2 + wp)
        #store final score
        z_scores[w] = delta / np.sqrt(var)
    return_list = [(w, z_scores[w]) for w in vocab]
    return_list.sort(key=itemgetter(1))
    if filter_short:
        return_list = [(w, s) for (w, s) in return_list if len(w) > 1]
    print("words most associated with the first corpus")
    for (w, s) in return_list[-20:][::-1]:
        print(w, s)
    print("words most associated with the second corpus")
    for (w, s) in return_list[:20]:
        print(w, s)
    
    x_vals = np.array([c1.get(w, 0) + c2.get(w, 0) for w in vocab])
    y_vals = np.array([z_scores[w] for w in z_scores])
    sizes = abs(y_vals) * word_size
    neg_color, pos_color, insig_color = ('orange', 'purple', 'grey')
    colors = []
    annots = []
    for w, y in zip(vocab, y_vals):
        if y > sig_val:
            colors.append(pos_color)
            annots.append(w)
        elif y < -sig_val:
            colors.append(neg_color)
            annots.append(w)
        else:
            colors.append(insig_color)
            annots.append(None)
    fig, ax = plt.subplots()
    ax.scatter(x_vals, y_vals, c=colors, s=sizes, linewidth=0)
    for i, annot in enumerate(annots):
        if annot is not None:
            ax.annotate(annot, (x_vals[i], y_vals[i]), color=colors[i], size=sizes[i])
    ax.set_xscale('log')
    plt.xlabel("Frequency of Word")
    plt.ylabel("Z-Score")
    
    plt.show() 
    if DATA_PATH:  
        plt.savefig(f'{DATA_PATH}/data/test.pdf')  # TODO define DATA_PATH
    # return return_list


def bayes_compare_language_from_counter_and_freq(c1, c2, prior=.01, filter_short=True):
    '''
    Prints z-score AND frequency

    Arguments:
    - c1, c2; counters from each sample
    - prior; either a float describing a uniform prior, or a dictionary describing a prior
    over vocabulary items. If you're using a predefined vocabulary, make sure to specify that
    when you make your CountVectorizer object.

    Returns:
    - A list of length |Vocab| where each entry is a (n-gram, zscore) tuple.'''
    vocab = set(c1.keys()) | set(c2.keys())
    if type(prior) is float:
        priors = {w: prior for w in vocab}
    else:
        priors = prior
    z_scores = {}
    a0 = sum(priors.values())
    n1 = sum(c1.values())
    n2 = sum(c2.values())
    print("Comparing language...")
    for w in vocab:
        #compute delta
        w1, w2, wp = c1.get(w, 0), c2.get(w, 0), priors[w]
        term1 = np.log((w1 + wp) / (n1 + a0 - w1 - wp))
        term2 = np.log((w2 + wp) / (n2 + a0 - w2 - wp))        
        delta = term1 - term2
        #compute variance on delta
        var = 1. / (w1 + wp) + 1. / (w2 + wp)
        #store final score
        z_scores[w] = delta / np.sqrt(var)
    return_list = [(w, z_scores[w], c1.get(w, 0), c2.get(w, 0)) for w in vocab]
    return_list.sort(key=itemgetter(1))
    if filter_short:
        return_list = [(w, s, cw1, cw2) for (w, s, cw1, cw2) in return_list if len(w) > 1]
    return return_list

def create_doc_counter(text, pos_suffix=True):
    '''
    Convert a pandas series of strings to a Counter of each word.
    pos_suffix: if True, append the part of speech to each word
    '''
    text_list =  text.to_list()
    res = Counter()
    for text in text_list:
        if pos_suffix:
            toks = [f'{token.text.lower()}_{token.pos_}' for token in nlp(text) if token.text not in exclude]
        else:
            toks = [token.text.lower() for token in nlp(text) if token.text not in exclude]
        toks_ctr = Counter(toks)
        res.update(toks_ctr)
    return res


def calc_fightin_words_by_speaker(df, speaker_list, orig_text="text", pos_suffix=True, test=False):
    '''
    Iterate over a list of speakers and calculate fighting word z-scores for
    each speaker comparing opponent mention sentences to non-opponent mention
    sentences.
    '''
    fw_list = []
    counter_dict = dict()
    for i, s in enumerate(speaker_list):
        print(f'{s} ({i+1} of {len(speaker_list)})')
        speaker_yes = df[(df.speaker == s) & (df.mentions_opponent == 'Y')]
        speaker_no = df[(df.speaker == s) & (df.mentions_opponent == 'N')]
        yes_counter = create_doc_counter(speaker_yes[f"{orig_text}"], pos_suffix=pos_suffix)
        no_counter = create_doc_counter(speaker_no[f"{orig_text}"], pos_suffix=pos_suffix)
        try:
            # bcl = bayes_compare_language(speaker_yes[f"{orig_text}"], speaker_no[f"{orig_text}"])
            bcl = bayes_compare_language_from_counter(yes_counter, no_counter)
        except:
            print(f'WARNING - bayes_compare_language failed for speaker {s}')
            continue
        fw = pd.DataFrame(bcl, columns=['word', 'z_score'])
        fw.insert(0, 'speaker', s)
        if pos_suffix:
            fw['pos'] = [w.split('_')[-1] for w in fw.word]
            fw['word'] = [w.split('_')[0] for w in fw.word]
        else:
            fw['pos'] = [nlp(w)[0].pos_ for w in fw.word]
        fw_list.append(fw)
        counter_dict[s] = {'yes': yes_counter, 'no': no_counter}
        if test:
            break
    return (fw_list, counter_dict)


def count_fw_frequency(fw_list, n=10, pos='ALL', opponent_words=True, removal_dict={},
        min_z_score=0):
    '''
    Count fighting word frequency for top n fighting words from each element in 
    list of candidates' fighting words.

    Inputs:
        fw_list: a list of pandas dataframes, where each dataframe has the
            variables ["speaker", "word", "z_score", "pos"] and is sorted by
            z_score in ascending order. 
        n (int): the number of top-n words to include for each candidate
        pos (str): the part of speech to include in the word counts, with the
            default including ALL parts of speech
        opponent_words (bool): Whether to take the top n words most associated 
            with opponent mentions (True) or the least associated with opponent
            mentions (False)
        removal_dict (dict): dictionary where keys are parts of speech and
            values are lists of words to be removed if they are present
        min_z_score (float): The minimum z_score required for a word to be 
            included for a candidate. 
    
    Outputs:
        A tuple including:
            - A Counter object with counts of all words that appear in
                candidates' top-n words
            - A dictionary where each key is a candidate and each value is a 
                list of the candidate's top-n words

    '''
    fw_counter = Counter()
    top_n_dict = dict()
    removal_words = removal_dict.get(pos)
    for fw in fw_list:
        fwc = fw.copy(deep=True)
        speaker = fwc.speaker.unique()[0]
        if not opponent_words:
            fwc.z_score = fwc.z_score * -1
        fwc = fwc.sort_values("z_score")
        if removal_words:
            fwc = fwc[~fwc.word.isin(removal_words)]
        fwc = fwc[fwc.z_score > min_z_score]
        if pos == 'ALL':
            top_n = fwc.word.tail(n)
        else:
            top_n = fwc[fwc.pos == pos].word.tail(n)
        fw_counter.update(top_n)
        top_n_dict[speaker] = top_n
    
    return (fw_counter, top_n_dict)


def count_fw_frequency_from_df(fw_df, n='ALL', pos='ALL', opponent_words=True,
        removal_dict={}, min_z_score=0):
    '''
    Count fighting word frequency for top n fighting words from each element in 
    list of candidates' fighting words.
    '''
    fw_counter = Counter()
    top_n_dict = dict()
    speakers = fw_df.speaker.unique()
    removal_words = removal_dict.get(pos)
    for speaker in speakers:
        fw_speaker = fw_df[fw_df.speaker == speaker]
        if removal_words:
            fw_speaker = fw_speaker[~fw_speaker.word.isin(removal_words)]
        if not opponent_words:
            fw_speaker.z_score = fw_speaker.z_score * -1
        fw_speaker_sorted = fw_speaker.sort_values("z_score")

        fw_speaker_sorted = fw_speaker_sorted[fw_speaker_sorted.z_score > min_z_score]
        if pos == 'ALL':
            if n == 'ALL':
                top_n = fw_speaker_sorted.word
            else:
                top_n = fw_speaker_sorted.word.tail(n)
        else:
            if n == 'ALL':
                top_n = fw_speaker_sorted[fw_speaker_sorted.pos == pos].word
            else:
                top_n = fw_speaker_sorted[fw_speaker_sorted.pos == pos].word.tail(n)
        fw_counter.update(top_n)
        top_n_dict[speaker] = top_n
    
    return (fw_counter, top_n_dict)


def calc_mean_degree(speaker_top_words, top_word_counter):
    '''
    Given a set of word nodes and a "graph" where nodes are speakers and words
    and edges are if the word is in the speaker's top/bottom N for ranked 
    fighting words, calculate the average degree of the set of word nodes.
    '''
    return mean([top_word_counter[word] for word in speaker_top_words])

def compile_mean_degrees(fw_counter, fw_dict):
    '''
    Iterate through speakers and calculate the mean degrees. 
    '''
    d = dict()
    for speaker, words in fw_dict.items():
        if len(words) > 0:
            d[speaker] = calc_mean_degree(words, fw_counter)
        else:
            d[speaker] = 0
    df = pd.DataFrame().from_dict(d, orient='index').reset_index() 
    #df.columns = ['speaker', 'mean_degree', 'N']     
    return df

def iterate_top_n(fw_list, pos='ALL', n_list=[5, 10, 15, 20, 25], 
    opponent_words=True, min_z_score=0):
    '''
    Calculate average top-n word degrees by speaker for different n's
    '''
    df_list = []
    for n in n_list:
        N = n
        fw_counter, fw_dict = count_fw_frequency(fw_list, pos=pos, n=N, 
            opponent_words=opponent_words, removal_dict={'ADJ': ['ukraine', 'blumenthal']}, 
            min_z_score=min_z_score)
        mean_deg_df = compile_mean_degrees(fw_counter, fw_dict)
        mean_deg_df['N'] = N
        mean_deg_df.columns = ['speaker', 'score', 'N']
        df_list.append(mean_deg_df)
    df = pd.concat(df_list)
    df.reset_index(inplace=True)
    return df

def calc_candidate_overlap(top_n_dict):
    '''
    Calculate mean overlap of top-n fightin words with other candidates. 
    '''
    overlap_tups = []
    for can_a, words_a in top_n_dict.items():
        for can_b, words_b in top_n_dict.items():
            ab_intersection = set(words_a) & set(words_b)
            overlap_tups.append((can_a, can_b, len(ab_intersection), ab_intersection))
    return pd.DataFrame(overlap_tups, 
        columns=['candidate_a', 'candidate_b', 'num_overlap', 'overlap_words '])


def calc_overlap_metrics(fw_list:list, pos_tags:list, min_z_score=0, n_list=[5, 10, 15, 20, 25]) -> pd.DataFrame:
    '''
    TODO
    '''
    df_list = []
    for tag in pos_tags:
        mo = iterate_top_n(fw_list, pos=tag, opponent_words=True, min_z_score=min_z_score, n_list=n_list) 
        mo['pos'] = tag
        mo['mentions_opponent'] = 'Y'   
        no_mo = iterate_top_n(fw_list, pos=tag, opponent_words=False, min_z_score=0) 
        no_mo['pos'] = tag
        no_mo['mentions_opponent'] = 'N' 

        df_list.append(mo)
        df_list.append(no_mo)

    return pd.concat(df_list)