#!/usr/bin/env python3

import numpy as np
import pickle
import bisect
from dataclasses import dataclass

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

allAnswerIndices = np.array(range(encodedAnswers.shape[0]), dtype=np.uint16)
allGuessIndices = np.array(range(encodedGuesses.shape[0]), dtype=np.uint16)
answerToGuessIndex = np.where(np.isin(encodedGuesses, encodedAnswers))[0]

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

with open("hintHashes.pickle", 'rb') as f:
    hintHashes = pickle.load(f)
with open("filters.pickle", 'rb') as f:
    filters = pickle.load(f)


def concatFilters(filter1, filter2):
    # https://stackoverflow.com/a/53896643
    idx = np.searchsorted(filter1, filter2)
    idx[idx==filter1.shape[0]] = 0
    return filter2[filter1[idx] == filter2]


@dataclass
class HintNode:
    words: list
    num_possible_words: np.uint16
    hint_hash: np.uint8

@dataclass
class WordNode:
    hints: list
    expected_guesses: np.float32
    guess_index: np.uint16


def bestWords(init_filter, n_options):
    best_words = []
    best_entropies_neg = []
    possible_filters = []
    hint_hashes = []
    filtered_index_to_check = 0
    for i in allGuessIndices:
        info_gains = []
        info_gain_weights = []
        possible_filters_temp = []
        hint_hashes_temp = []
        for j in range(3**5):
            if filters[i][j] is not None:
                combined_filter = concatFilters(init_filter, filters[i][j])
                if len(combined_filter) > 0:
                    info_gains.append(np.log2(init_filter.shape[0] / combined_filter.shape[0]))
                    info_gain_weights.append(combined_filter.shape[0])
                    possible_filters_temp.append(combined_filter)
                    hint_hashes_temp.append(j)
        entropy = np.average(info_gains, weights=info_gain_weights)
        if filtered_index_to_check < len(init_filter) and i == answerToGuessIndex[init_filter[filtered_index_to_check]]:
            index = bisect.bisect_left(best_entropies_neg, -entropy)
            filtered_index_to_check += 1
        else:
            index = bisect.bisect_right(best_entropies_neg, -entropy)
        if entropy > 0:
            best_words.insert(index, i)
            best_entropies_neg.insert(index, -entropy)
            possible_filters.insert(index, possible_filters_temp)
            hint_hashes.insert(index, hint_hashes_temp)
            if len(best_words) > n_options:
                best_words.pop()
                best_entropies_neg.pop()
                possible_filters.pop()
                hint_hashes.pop()
    return best_words, best_entropies_neg, possible_filters, hint_hashes


def best1StepStartingWords():
    N = 100
    best_words, best_entropies_neg, possible_filters, hint_hashes = bestWords(allAnswerIndices, N)
    print("Sorted first guess entropies for Wordle game with {} possible solutions:".format(encodedAnswers.shape[0]))
    for i in range(len(best_words)):
        print("    {} | E[I] = {:.3f}".format(guesses[best_words[i]], -best_entropies_neg[i]))


def buildWordleTree(filter, hint_node, prefix=""):
    N = 10 # int(np.ceil(8 * 2**(-depth)))
    best_words, best_entropies_neg, possible_filters, hint_hashes = bestWords(filter, N)
    for i in range(len(best_words)):
        hint_node.words.append(WordNode([], 0, best_words[i]))
        for j in range(len(possible_filters[i])):
            new_prefix = "{}  {}.{}".format(prefix, i, j)
            print(new_prefix)
            hint_node.words[i].hints.append(HintNode([], possible_filters[i][j].shape[0], hint_hashes[i][j]))
            if possible_filters[i][j].shape[0] > 2:
                buildWordleTree(possible_filters[i][j], hint_node.words[i].hints[j], new_prefix)
            else:
                last_guesses = answerToGuessIndex[possible_filters[i][j]]
                for guess in last_guesses:
                    hint_node.words[i].hints[j].words.append(WordNode([], 0, guess))


def optimizeWordleTree(hint_node):
    if hint_node.num_possible_words > 2:
        expected_guesses = []
        for word_node in hint_node.words:
            expected_guesses_for_hints = []
            hint_weights = []
            for next_hint_node in word_node.hints:
                if next_hint_node.hint_hash == 3**5 - 1:
                    expected_guesses_for_hints.append(0)
                else:
                    optimizeWordleTree(next_hint_node)
                    expected_guesses_for_hints.append(next_hint_node.words[0].expected_guesses)
                hint_weights.append(next_hint_node.num_possible_words)
            expected_guesses.append(np.average(expected_guesses_for_hints, weights=hint_weights))
        best_word_index = np.argmin(expected_guesses)
        hint_node.words = [hint_node.words[best_word_index]]
        hint_node.words[0].expected_guesses = 1 + expected_guesses[best_word_index]
    else:
        if hint_node.num_possible_words == 2:
            expected_guesses = 1.5
        elif hint_node.num_possible_words == 1:
            expected_guesses = 1
        else:
            assert(False)
        for word_node in hint_node.words:
            word_node.expected_guesses = expected_guesses


def verifyWordlePerformance(wordle_tree):
    avg_guesses = 0

    for j in range(encodedAnswers.shape[0]):
        n_guesses = 0
        hint_node = wordle_tree

        while (hint_node.num_possible_words > 2) and (hint_node.hint_hash != 3**5 - 1):
            next_hint_hash = hintHashes[hint_node.words[0].guess_index][j]
            for next_hint_node in hint_node.words[0].hints:
                if next_hint_node.hint_hash == next_hint_hash:
                    break
            assert(next_hint_node.hint_hash == next_hint_hash)
            hint_node = next_hint_node
            n_guesses += 1
    
        if hint_node.hint_hash != 3**5 - 1:
            for word_node in hint_node.words:
                n_guesses += 1
                if word_node.guess_index == answerToGuessIndex[j]:
                    break
            assert(word_node.guess_index == answerToGuessIndex[j])

        avg_guesses += n_guesses

    avg_guesses /= encodedAnswers.shape[0]
    return avg_guesses


if __name__ == "__main__":
    # PROGRAM 1: Show best starting words measured by 1-step expected information gain
    # best1StepStartingWords()


    # PROGRAM 2: Create tree of Wordle games using a limited set of "good" guesses at each step
    # wordleTree = HintNode([], encodedAnswers.shape[0], 0)
    # buildWordleTree(allAnswerIndices, wordleTree)
    # with open("wordleTree.pickle", 'wb') as f:
    #     pickle.dump(wordleTree, f)


    # PROGRAM 3: Optimize the Wordle tree to minimize the average number of guesses
    # with open("wordleTree.pickle", 'rb') as f:
    #     wordleTree = pickle.load(f)
    # optimizeWordleTree(wordleTree)
    # print("Expected number of guesses: {}".format(wordleTree.words[0].expected_guesses))
    # print("Starting word: {}".format(guesses[wordleTree.words[0].guess_index]))
    # print("Second word if all blanks: {}".format(guesses[wordleTree.words[0].hints[0].words[0].guess_index]))
    # with open("wordleTreeOptimized.pickle", 'wb') as f:
    #     pickle.dump(wordleTree, f)


    # PROGRAM 4: Empirically compute the average number of guesses by testing over all possible answer words
    with open("wordleTreeOptimized.pickle", 'rb') as f:
        wordleTree = pickle.load(f)
    print("Average number of guesses: {}".format(verifyWordlePerformance(wordleTree)))
