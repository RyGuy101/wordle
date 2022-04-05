#!/usr/bin/env python3

import numpy as np
import pickle

guesses = []
with open("guesses.txt", 'r') as f:
    for line in f:
        guesses.append(line.rstrip())

answers = []
with open("answers.txt", 'r') as f:
    for line in f:
        answers.append(line.rstrip())

WordT  = np.dtype([('positions', np.uint32, 5), ('counts', np.uint8, 26)])
StateT = np.dtype([('positions', np.uint32, 5), ('counts', np.uint8, 26), ('upper_bounded', np.bool_, 26)])

blankState = np.array([(
    np.full(5, 0xffffffff, dtype=np.uint32),
    np.zeros(26, dtype=np.uint8),
    np.zeros(26, dtype=np.bool_)
)], dtype=StateT)[0]

def encodeWord(word):
    encodedWord = np.zeros(1, dtype=WordT)[0]
    for j in range(5):
        l = ord(word[j]) - ord('a')
        encodedWord['positions'][j] = 1 << l
        encodedWord['counts'][l] += 1
    return encodedWord

# def decodeWord(encodedWord):
#     word = ""
#     for j in range(5):
#         c = int(np.log2(encodedWord['positions'][j])) + ord('a')
#         word += chr(c)
#     return word

encodedAnswers = np.zeros(len(answers), dtype=WordT)
for i in range(len(answers)):
    encodedAnswers[i] = encodeWord(answers[i])

encodedGuesses = np.zeros(len(guesses), dtype=WordT)
for i in range(len(guesses)):
    encodedGuesses[i] = encodeWord(guesses[i])


def filter(state, answers_list):
    indices = []
    for i, answer in enumerate(answers_list):
        not_upper_bounded = ~state['upper_bounded']
        if (
            np.all(answer['positions'] & state['positions']) and 
            np.all(answer['counts'][not_upper_bounded]      >= state['counts'][not_upper_bounded]) and 
            np.all(answer['counts'][state['upper_bounded']] == state['counts'][state['upper_bounded']])
        ):
            indices.append(i)

    return np.array(indices, dtype=np.uint16)


def nextState(state, guess, truth):
    state = state.copy()
    for j in range(5):
        if guess['positions'][j] == truth['positions'][j]:
            state['positions'][j] = truth['positions'][j]
        else:
            state['positions'][j] &= ~guess['positions'][j]
    for l in range(26):
        if guess['counts'][l] <= truth['counts'][l]:
            if guess['counts'][l] > state['counts'][l]:
                state['counts'][l] = guess['counts'][l]
        else:
            state['counts'][l] = truth['counts'][l]
            state['upper_bounded'][l] = True
    return state


def hintHash(guess, truth):
    hash = 0
    unused_letters = np.ones(5, dtype=np.bool_)
    for j in range(5):
        if guess['positions'][j] == truth['positions'][j]:
            hash += 2 * 3**j
            unused_letters[j] = False
    for j in np.arange(5)[unused_letters]:
        for k in np.arange(5)[unused_letters]:
            if guess['positions'][j] == truth['positions'][k]:
                hash += 1 * 3**j
                unused_letters[k] = False
                break
    return hash


# hintHashes = np.zeros((encodedGuesses.shape[0], encodedAnswers.shape[0]), dtype=np.uint8)
# filters = np.full((encodedGuesses.shape[0], 3**5), None, dtype=object)
# for i in range(encodedGuesses.shape[0]):
#     for j in range(encodedAnswers.shape[0]):
#         hintHashes[i][j] = hintHash(encodedGuesses[i], encodedAnswers[j])
#         if filters[i][hintHashes[i][j]] is None:
#             filters[i][hintHashes[i][j]] = filter(nextState(blankState, encodedGuesses[i], encodedAnswers[j]), encodedAnswers)
#     print("Preprocessed {}/{} guess words".format(i+1, encodedGuesses.shape[0]))
# with open("hintHashes.pickle", 'wb') as f:
#     pickle.dump(hintHashes, f)
# with open("filters.pickle", 'wb') as f:
#     pickle.dump(filters, f)

# with open("hintHashes.pickle", 'rb') as f:
#     hintHashes = pickle.load(f)
with open("filters.pickle", 'rb') as f:
    filters = pickle.load(f)


def concatFilters(filter1, filter2):
    combined_filter = []
    i1 = 0
    i2 = 0
    while i1 < filter1.shape[0] and i2 < filter2.shape[0]:
        if filter1[i1] < filter2[i2]:
            i1 += 1
        elif filter1[i1] > filter2[i2]:
            i2 += 1
        else:
            combined_filter.append(filter1[i1])
            i1 += 1
            i2 += 1
    return combined_filter


def best1StepStartingWord():
    entropies = np.zeros(encodedGuesses.shape[0])
    for i in range(encodedGuesses.shape[0]):
        info_gains = []
        info_gain_weights = []
        for j in range(3**5):
            if filters[i][j] is not None:
                info_gains.append(np.log2(encodedAnswers.shape[0] / filters[i][j].shape[0]))
                info_gain_weights.append(filters[i][j].shape[0])
        entropies[i] = np.average(info_gains, weights=info_gain_weights)
        print("Evaluated {}/{} words".format(i+1, encodedGuesses.shape[0]))
    print()

    print("Sorted first guess entropies for Wordle game with {} possible solutions:".format(encodedAnswers.shape[0]))
    sorting = entropies.argsort()
    entropies = entropies[sorting]
    sorted_guesses = np.array(guesses)[sorting]
    N = sorted_guesses.shape[0]
    for i in range(min(100, N)):
        print("    {} | E[I] = {:.3f}".format(sorted_guesses[N-1-i], entropies[N-1-i]))


best1StepStartingWord()
