from collections import defaultdict
import pandas as pd
from convokit import Corpus, download


def find_comment_id_by_shortened_id(corpus, shortened_comment_id):
    for key in corpus.utterances.keys():
        if key[3:] == shortened_comment_id:
            return key
    raise Exception(f"Shortened comment id: {shortened_comment_id} was not found in corpus")


def find_ids_of_related_to_conversation_utterances(corpus, conversation_id, direct_reply_only=True):
    related_utterances = set()

    if direct_reply_only:
        for shortened_comment_id in corpus.utterances[conversation_id].meta['replies']:
            full_comment_id = find_comment_id_by_shortened_id(corpus, shortened_comment_id)
            related_utterances.add(full_comment_id)

    else:
        for key, val in corpus.utterances.items():
            if val.conversation_id == conversation_id and key != val.conversation_id:
                related_utterances.add(key)

    return related_utterances


def divide_to_positive_and_negative_lists(related_utterances):
    positive_labeled_utterances = []
    negative_labeled_utterances = []

    for comment_id in related_utterances:
        success = corpus.utterances[comment_id].meta['success']
        if success == 1:
            positive_labeled_utterances.append(comment_id)
        elif success == 0:
            negative_labeled_utterances.append(comment_id)

    return positive_labeled_utterances, negative_labeled_utterances


def balance_pos_and_neg_labels(positive_labeled, negative_labeled):
    pos_len = len(positive_labeled)
    neg_len = len(negative_labeled)

    min_len = min(pos_len, neg_len)

    return positive_labeled[:min_len], negative_labeled[:min_len]


def generate_data_points_from_related_utterances(corpus, orig_utter, pos_utter, neg_utter):
    op_comment = [corpus.utterances[orig_utter].text] * 2 * len(pos_utter)
    label = [1] * len(pos_utter) + [0] * len(neg_utter)
    reply = []

    for utter in pos_utter:
        reply.append(corpus.utterances[utter].text)
    for utter in neg_utter:
        reply.append(corpus.utterances[utter].text)

    return op_comment, reply, label


def preprocess_corpus(corpus, direct_reply_only=False):
    conversations_id = set(corpus.conversations.keys())
    op_comment, reply, label = [], [], []

    for conversation_id in conversations_id:
        related_utterances = find_ids_of_related_to_conversation_utterances(corpus, conversation_id, direct_reply_only=direct_reply_only)
        pos_list, neg_list = divide_to_positive_and_negative_lists(related_utterances)
        balanced_pos, balanced_neg = balance_pos_and_neg_labels(pos_list, neg_list)
        op, rep, lab = generate_data_points_from_related_utterances(corpus, conversation_id, balanced_pos, balanced_neg)
        op_comment += op
        reply += rep
        label += lab

    corpus_df = pd.DataFrame({'op_comment':op_comment, 'reply':reply, 'label':label})
    return corpus_df


if __name__ == "__main__":
    corpus = Corpus(filename=download("winning-args-corpus"))
    corpus_df = preprocess_corpus(corpus)


