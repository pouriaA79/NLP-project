# imports
import string

import nltk
from nltk import FreqDist, ngrams

pos_file = open('rt-polarity.pos', encoding='utf8').read()
neg_file = open('rt-polarity.neg', encoding='utf8').read()

# write the removal characters such as : panctuation
string.punctuation = string.punctuation + '"' + '"' + '-' + '''+''' + '—'
# -----------positive
pos_file_nl_removed = ""
for line in pos_file:
    line_nl_removed = line.replace("\n", " ")  # removes newlines
    pos_file_nl_removed += line_nl_removed
pos_file_p = "".join([char for char in pos_file_nl_removed if char not in string.punctuation])
pos_sents = nltk.sent_tokenize(pos_file_p)
# ----------negative
neg_file_nl_removed = ""
for line in neg_file:
    line_nl_removed = line.replace("\n", " ")  # removes newlines
    neg_file_nl_removed += line_nl_removed
neg_file_p = "".join([char for char in neg_file_nl_removed if char not in string.punctuation])
neg_sents = nltk.sent_tokenize(neg_file_p)
# ---------------------------------------------------generate unigrams for positives
pos_unigram = []
pos_tokenized_text = []
for line in pos_sents:
    sentence = line.lower()
    sequence = nltk.word_tokenize(sentence)
    # pos_unigram.append('<s>')
    for word in sequence:
        if word == '.':
            sequence.remove(word)
        else:
            pos_unigram.append(word)
    # pos_unigram.append('</s>')
    pos_tokenized_text.append(sequence)
# --------------------------------------------------------generate unigrams for negatives
neg_unigram = []
neg_tokenized_text = []
for line in neg_sents:
    sentence = line.lower()
    sequence = nltk.word_tokenize(sentence)
    # neg_unigram.append('<s>')
    for word in sequence:
        if word == '.':
            sequence.remove(word)
        else:
            neg_unigram.append(word)
    # neg_unigram.append('</s>')
    neg_tokenized_text.append(sequence)

# -------------------------------remove 8 most common and word freq les than 2
# 8 because does not consider <s> </s>
pos_freq_un = FreqDist(pos_unigram)
pos_freq_un_mostCommon = pos_freq_un.most_common(8)
print(pos_freq_un_mostCommon)

neg_freq_un = FreqDist(neg_unigram)
neg_freq_un_mostCommon = neg_freq_un.most_common(8)

for item in pos_freq_un.items():  # ('the', 2300)
    if item in pos_freq_un_mostCommon:
        for t in range(0, item[1]):
            pos_unigram.remove(item[0])
            # -------------------------------correct it
        if item[0] in neg_unigram:
            for tt in range(0, neg_freq_un.get(item[0])):
                neg_unigram.remove(item[0])
    elif item[1] < 2:
        pos_unigram.remove(item[0])
        if item[0] in neg_unigram:
            for tt in range(0, neg_freq_un.get(item[0])):
                neg_unigram.remove(item[0])

neg_freq_un = FreqDist(neg_unigram)
neg_un_len = len(neg_freq_un)
for item in neg_freq_un.items():
    if item in neg_freq_un_mostCommon:
        for t in range(0, item[1]):
            neg_unigram.remove(item[0])
            # -------------------------------correct it
        if item[0] in pos_unigram:
            for tt in range(0, pos_freq_un.get(item[0])):
                pos_unigram.remove(item[0])
    elif item[1] < 2:
        neg_unigram.remove(item[0])
        if item[0] in pos_unigram:
            if item[0] in pos_unigram:
                for tt in range(0, pos_freq_un.get(item[0])):
                    pos_unigram.remove(item[0])

# ---------------------------------------------------generate bigrams
pos_bigram = []
pos_bigram.extend(list(ngrams(pos_unigram, 2)))
pos_freq_bi = nltk.FreqDist(pos_bigram)
neg_bigram = []
neg_bigram.extend(list(ngrams(neg_unigram, 2)))
neg_freq_bi = nltk.FreqDist(neg_bigram)
# --------------------------------------------------recreate unigarms after deletion
pos_freq_un = FreqDist(pos_unigram)
pos_un_len = len(pos_freq_un)
neg_freq_un = FreqDist(neg_unigram)
neg_un_len = len(neg_freq_un)
# ---------------------------------------------------calculate Possibilities
pos_bi_prob = {}  #
for item in pos_freq_bi.items():
    pos_bi_prob[item[0]] = item[1] / pos_freq_un.get(item[0][0])

pos_un_prob = {}
for item in pos_freq_un.items():
    pos_un_prob[item[0]] = pos_freq_un.get(item[0]) / pos_un_len

neg_bi_prob = {}
for item in neg_freq_bi.items():
    neg_bi_prob[item[0]] = item[1] / neg_freq_un.get(item[0][0])

neg_un_prob = {}
for item in neg_freq_un.items():
    neg_un_prob[item[0]] = neg_freq_un.get(item[0]) / neg_un_len
# -------------------------------------------------------get input from user
q_text = "!q"
while True:
    input_sentence = input("enter sentence:")
    if input_sentence == q_text:
        break
    input_sentence = input_sentence.replace("\n", " ")
    input_sentence = "".join([char for char in input_sentence if char not in string.punctuation])
    token_1 = nltk.word_tokenize(input_sentence)
    token_1.insert(0, '<s>')
    token_1.append('</s>')
    input_unigram = token_1
    input_bigram = []
    input_bigram.extend(list(ngrams(token_1, 2)))

    # ----------------- variables
    lambda1_bi = 0.1
    lambda2_bi = 0.2
    lambda3_bi = 0.7
    lambda1_un = 0.3
    lambda2_un = 0.7
    epsilon_bi = 0.84
    epsilon_un = 0.84

    # --------------------------- calculate input Possibilities
    # ---- pos bigram
    input_bi_pos_prob = {}
    for element in input_bigram:
        if element in pos_bi_prob.keys():
            pos_bi_ele_prob = pos_bi_prob[element]
        else:
            pos_bi_ele_prob = 0
        if element[1] in pos_un_prob.keys():
            pos_un_ele_prob = pos_un_prob[element[1]]
        else:
            pos_un_ele_prob = 0
        input_bi_pos_prob[
            element] = lambda3_bi * pos_bi_ele_prob + lambda2_bi * pos_un_ele_prob + lambda1_bi * epsilon_bi
    # ---- pos unigram
    input_un_pos_prob = {}
    for element in input_unigram:
        if element in pos_un_prob.keys():
            pos_un_ele_prob = pos_un_prob[element]
        else:
            pos_un_ele_prob = 0
        input_un_pos_prob[element] = lambda2_un * pos_un_ele_prob + lambda1_un * epsilon_un

    # ------ neg bigram
    input_bi_neg_prob = {}
    for element in input_bigram:
        if element in neg_bi_prob.keys():
            neg_bi_ele_prob = neg_bi_prob[element]
        else:
            neg_bi_ele_prob = 0
        if element[1] in neg_un_prob.keys():
            neg_un_ele_prob = neg_un_prob[element[1]]
        else:
            neg_un_ele_prob = 0
        input_bi_neg_prob[
            element] = lambda3_bi * neg_bi_ele_prob + lambda2_bi * neg_un_ele_prob + lambda1_bi * epsilon_bi
    # ----- neg unigram
    input_un_neg_prob = {}
    for element in input_unigram:
        if element in neg_un_prob.keys():
            neg_un_ele_prob = neg_un_prob[element]
        else:
            neg_un_ele_prob = 0
        input_un_neg_prob[element] = lambda2_un * neg_un_ele_prob + lambda1_un * epsilon_un
    # --------------------------- multiply bigrams Possibilities
    input_pos_bi_ans = 1
    for x in input_bi_pos_prob.keys():
        input_pos_bi_ans = input_pos_bi_ans * input_bi_pos_prob[x]
    input_neg_bi_ans = 1
    for x in input_bi_neg_prob.keys():
        input_neg_bi_ans = input_neg_bi_ans * input_bi_neg_prob[x]
    # ---------------------------- multiply unigrams Possibilities
    input_pos_un_ans = 1
    for x in input_un_pos_prob.keys():
        input_pos_un_ans = input_pos_un_ans * input_un_pos_prob[x]
    input_neg_un_ans = 1
    for x in input_un_neg_prob.keys():
        input_neg_un_ans = input_neg_un_ans * input_un_neg_prob[x]

    print("input_neg_bi_ans", input_neg_bi_ans)
    print("input_pos_bi_ans", input_pos_bi_ans)
    if input_neg_bi_ans > input_pos_bi_ans:
        print("bigram : filter this")
    else:
        print("bigram : not filter this")

    print("input_neg_un_ans", input_neg_un_ans)
    print("input_pos_un_ans", input_pos_un_ans)
    if input_neg_un_ans > input_pos_un_ans:
        print("unigram : filter this")
    else:
        print("unigram : not filter this")
