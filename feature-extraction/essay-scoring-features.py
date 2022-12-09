# training set has 13k essays
# 6 essay sets
# divide training set into 9k training essays and 3k test essays
# divide training essays further into 1200 training essays and 500 test sets per essay set

# feature generation
# generating five categories of features

# VISUAL (for every essay)
# find proportion of words that are visual
# find proportion of UNIQUE words that are visual and average imagery scores for these words
# average imagery score of the essay


# BEAUTIFUL WORDS
# only words above 5 characters in length are considered
# two factors for individual word beauty:
# 1. HIGH PERPLEXITY LETTER MODEL: for a word, find the product of its character frequencies. The lower the product, the more complex the word
# 2. HIGH PERPLEXITY PHONEME MODEL: create a 4-gram for syllables to determine phoneme frequency. For every word in the essay, check if a pronounciation exists, and yes then find the likeliness of its 4-gram combination
# RESULT IS: average letter and phoneme frequencies per beautiful word and per essay

# EMOTIVE EFFECTIVENESS
# find proportion of sentiments and strength individually for every essay

# MATURTIY
# for every essay, find its average maturity, top mature tokens, and vocabulary maturity

# perform svm 
# using one feature 
# then make a comparison on which feature is better

# ESSAY SET 1
# Resolved score range: 2 - 12
# Scoring: Score1, Score2, ResolvedScore

# ESSAY SET 3
# Score range: 0 - 3

# ESSAY SET 4
# Score range: 0 - 3

# ESSAY SET 5
# Score range: 0 - 4

# ESSAY SET 6
# Score range: 0 - 4

# ESSAY SET 7:
# Score range: 0 - 30

import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import TweetTokenizer
import enchant
import string
import os
import re
from g2p_en import G2p

tknzr = TweetTokenizer()
d = enchant.Dict("en_US")
g2p = G2p()

stop_words = set(stopwords.words('english'))
personal_pronouns = ["i", "she", "he", "it", "we", "you", "they", "me", "her", "him", "them", "us", "myself"]
relative_pronouns = ["that", "who", "where", "which", "whose", "when", "why", "whom", "whom", "what"]
proper_pronouns = ['@ORGANIZATION', '@PEOPLE', '@LOCATION', '@DATE', '@CAPS', '@NUM', '@MONTH', '@YEAR', '@PERCENT', '@TIME', '@MONEY', '@QUANTITY', '@LANGUAGE']
# eng_l_freq = [['e', 12.02], ['t', 9.10], ['a', 8.12), ('o', 7.68), ('i', 7.31), ('n', 6.95), ('s', 6.28), ('r', 6.02), ('h',5.92), ('d', 4.32), ('l', 3.98),
                    #   ('u', 2.88), ('c', 2.71), ('m', 2.61), ('f', 2.30), ('y', 2.11), ('w', 2.09), ('g', 2.03), ('p', 1.82), ('b',1.49), ('v', 1.11), ('k', 0.69)
                    #   ('x', 0.17), ('q', 0.11), ('j', 0.10), ('z', 0.07)]

def load_file(filename):
    dataset = pd.read_table(filename, header=0, sep=",", encoding="unicode_escape")
    
    essay_set1 = dataset.loc[dataset['essay_set'] == 1]
    # essay_set2 = dataset.loc[dataset['essay_set'] == 3]
    # essay_set3 = dataset.loc[dataset['essay_set'] == 4]
    # essay_set4 = dataset.loc[dataset['essay_set'] == 5]
    # essay_set5 = dataset.loc[dataset['essay_set'] == 6]
    # essay_set6 = dataset.loc[dataset['essay_set'] == 7]

    e = essay_set1['essay']
    return e

def word_tokenization(essay_set):

    # for essay in essay_set:

    e_tokenize = tknzr.tokenize(essay_set[0])
    bag_of_words = []
    for x in e_tokenize:
        if (x[:len(x)-1] in proper_pronouns) or (x in string.punctuation):
            pass
        else:
            bag_of_words.append(x.lower())

    # print(bag_of_words)

    # word-based tokenization
    word_tokens = {}
    token = 1
    for x in bag_of_words:
        if x in word_tokens.keys():
            pass
        else:
            word_tokens[x] = token
            token += 1

    return word_tokens
    
def vocabulary_check(word_tokens):

    # check for vocabulary errors
    total_words = len(word_tokens)
    correctness = 0
    correct = 0
    for x in word_tokens.keys():
        if (d.check(x) == True):
            correct = correct + 1

    correctness = correct / total_words
    # print(correctness)
    return correctness

def extract_feature_set1(word_tokens):
    
    # SECTION 3.3 - BEAUTIFUL WORDS FEATURE SET

    # find beautiful words per essay, length >= 6
    beautiful_words = []
    for x, y in word_tokens.items():
        if len(x) >= 6:
            # check if word exists in dictionary
            if d.check(x) == True:
                beautiful_words.append(y)

    # average letter frequencies per essay
    total_letters = 0
    characters = {}
    for word in word_tokens.keys():
        for ch in word:
            total_letters = total_letters + 1
            if ch not in characters.keys():
                characters[ch] = 0
            else:
                characters[ch] += 1

    for key in characters.keys():
        characters[key] = characters[key]/total_letters

    # phenomes
    # get phenome frequencies per essay
    phoneme_dict = {}

    total_phoneme = 0
    texts = word_tokens.keys()
    for text in texts:
        out = g2p(text)
        for x in out:
            total_phoneme += 1
            if re.sub(r'\d+', '', x) in phoneme_dict.keys():
                phoneme_dict[re.sub(r'\d+', '', x)] = phoneme_dict[re.sub(r'\d+', '', x)] + 1
            else:
                phoneme_dict[re.sub(r'\d+', '', x)] = 1
    
    for key in phoneme_dict.keys():
        phoneme_dict[key] = phoneme_dict[key]/total_phoneme

    # print(phenome_dict)

    return beautiful_words, characters, phoneme_dict

def extract_feature_set2(word_tokens):
    
    # SECTION 3.4 - EMOTIVE EFFECTIVENESS FEATURE SET
    lexicon = {}
    with open('subjclueslen1-HLTEMNLP05.tff') as f:
        
        for line in f:
            content = f.readline()
            row = content.split()
            type = row[0][5:]
            words = row[2][6:]
            pos = row[3][5:]
            polarity = row[5][14:]
            
            lexicon[words] = (type, pos, polarity)

    # print(lexicon)

    strong_positive = 0
    strong_negative = 0
    strong_neutral = 0
    strong_both = 0

    weak_positive = 0
    weak_negative = 0
    weak_neutral = 0
    weak_both = 0

    for w in word_tokens.keys():
        if w in lexicon.keys():
            if lexicon[w][0] == "strongsubj":
                if lexicon[w][2] == "positive":
                    strong_positive += 1

                elif lexicon[w][2] == "negative":
                    strong_negative += 1

                elif lexicon[w][2] == "neutral":
                    strong_neutral += 1

                elif lexicon[w][2] == "both":
                    strong_both += 1 

            elif lexicon[w][0] == "weaksubj":
                if lexicon[w][2] == "positive":
                    weak_positive += 1

                elif lexicon[w][2] == "negative":
                    weak_negative += 1

                elif lexicon[w][2] == "neutral":
                    weak_neutral += 1

                elif lexicon[w][2] == "both":
                    weak_both += 1 
    
    return strong_positive/len(word_tokens), strong_negative/len(word_tokens), strong_neutral/len(word_tokens), strong_both/len(word_tokens), weak_positive/len(word_tokens), weak_negative/len(word_tokens), weak_neutral/len(word_tokens), weak_both/len(word_tokens)

def extract_feature_set3(word_tokens):
    
    # SECTION 3.5 - LEARNING MATURITY
    # for every essay, find its average maturity, top mature tokens, and vocabulary maturity
    maturity_tokens = {}
    avg_maturity = 0
    with open('AoA Ratings.csv') as f:
        next(f)
        for line in f:
            content = f.readline()
            content = content.split(",")
            if content[0] in word_tokens.keys():
                avg_maturity += float(content[4])
                maturity_tokens[content[0]] = float(content[4])

    avg_maturity = avg_maturity/len(maturity_tokens)
    sorted_maturity_tokens = sorted(maturity_tokens, key=lambda item : maturity_tokens[item], reverse=True)

    # get the top 5 tokens
    top_tokens = []
    for word in sorted_maturity_tokens[:5]:
        top_tokens.append(word_tokens[word])

    return avg_maturity, top_tokens



essay_set = load_file("training_set.csv")
word_tokens = word_tokenization(essay_set)
correctness = vocabulary_check(word_tokens)

beautiful_words, character_list, phoneme_dict = extract_feature_set1(word_tokens)
strong_positive, strong_negative, strong_neutral, strong_both, weak_positive, weak_negative, weak_neutral, weak_both = extract_feature_set2(word_tokens)
avg_maturity, top_tokens = extract_feature_set3(word_tokens)

# References:
# https://stackoverflow.com/questions/23317458/how-to-remove-punctuation
# https://stackoverflow.com/questions/33666557/get-phonemes-from-any-word-in-python-nltk-or-other-modules
