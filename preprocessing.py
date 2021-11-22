import os.path
import pandas as pd
import transformers
from convokit import Corpus, download

# Tokenization constants
BERT_BASE_CASED = "bert-base-cased"

# CMV Dataset
CONVOKIT_DATASET_NAME = 'winning-args-corpus'
CMV_DATASET_NAME = 'cmv_delta_prediction.json'
SUCCESS = 'success'
REPLIES = 'replies'

# Pandas column names
OP_COMMENT = 'op_comment'
REPLY = 'reply'
LABEL = 'label'


def find_comment_id_by_shortened_id(cmv_corpus, shortened_comment_id):
    for key in cmv_corpus.utterances.keys():
        if key[3:] == shortened_comment_id:
            return key
    raise Exception(f"Shortened comment id: {shortened_comment_id} was not found in corpus")


def find_ids_of_related_to_conversation_utterances(cmv_corpus, conversation_id, direct_reply_only=True):
    related_utterances = set()

    if direct_reply_only:
        for shortened_comment_id in cmv_corpus.utterances[conversation_id].meta[REPLIES]:
            full_comment_id = find_comment_id_by_shortened_id(cmv_corpus, shortened_comment_id)
            related_utterances.add(full_comment_id)

    else:
        for key, val in cmv_corpus.utterances.items():
            if val.conversation_id == conversation_id and key != val.conversation_id:
                related_utterances.add(key)

    return related_utterances


def divide_to_positive_and_negative_lists(related_utterances):
    positive_labeled_utterances = []
    negative_labeled_utterances = []

    for comment_id in related_utterances:
        success = cmv_corpus.utterances[comment_id].meta[SUCCESS]
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


def generate_data_points_from_related_utterances(cmv_corpus, orig_utter, pos_utter, neg_utter):
    op_comment = [cmv_corpus.utterances[orig_utter].text] * 2 * len(pos_utter)
    label = [1] * len(pos_utter) + [0] * len(neg_utter)
    reply = []

    for utter in pos_utter:
        reply.append(cmv_corpus.utterances[utter].text)
    for utter in neg_utter:
        reply.append(cmv_corpus.utterances[utter].text)

    return op_comment, reply, label


def preprocess_corpus(cmv_corpus, direct_reply_only=False):
    conversations_id = set(cmv_corpus.conversations.keys())
    op_comment, reply, label = [], [], []

    for conversation_id in conversations_id:
        related_utterances = find_ids_of_related_to_conversation_utterances(cmv_corpus,
                                                                            conversation_id,
                                                                            direct_reply_only=direct_reply_only)
        pos_list, neg_list = divide_to_positive_and_negative_lists(related_utterances)
        balanced_pos, balanced_neg = balance_pos_and_neg_labels(pos_list, neg_list)
        op, rep, lab = generate_data_points_from_related_utterances(cmv_corpus,
                                                                    conversation_id,
                                                                    balanced_pos,
                                                                    balanced_neg)
        op_comment += op
        reply += rep
        label += lab

    corpus_df = pd.DataFrame({OP_COMMENT: op_comment, REPLY: reply, LABEL: label})
    return corpus_df


def tokenize(corpus_df):
    """

    :param corpus_df: A pandas DataFrame instance with the columns {OP_COMMENT, REPLY, LABEL}. The OP_COMMENT
        and REPLY columns consist of text entries while LABEL is {0, 1}.
    :return: A dataframe with the same columns as the original corpus_df parameter, but with OP_COMMENT and
        REPLY tokenized.
    """
    tokenized_corpus = corpus_df.copy(deep=True)
    tokenizer = transformers.BertTokenizer.from_pretrained(BERT_BASE_CASED)
    tokenized_op_comments = tokenized_corpus[OP_COMMENT].apply(lambda s: tokenizer.tokenize(s))
    tokenized_reply = tokenized_corpus[REPLY].apply(lambda s: tokenizer.tokenize(s))
    tokenized_corpus[OP_COMMENT] = tokenized_op_comments
    tokenized_corpus[REPLY] = tokenized_reply
    return tokenized_corpus


if __name__ == "__main__":
    dataset_path = os.path.join(os.getcwd(), CMV_DATASET_NAME)
    if not (CMV_DATASET_NAME in os.listdir(os.getcwd())):
        cmv_corpus = Corpus(filename=download(CONVOKIT_DATASET_NAME))
        corpus_df = preprocess_corpus(cmv_corpus)
        corpus_df.to_json(dataset_path)
    else:
        corpus_df = pd.read_json(dataset_path)
    tokenized_dataset = tokenize(corpus_df)
    print(tokenized_dataset)
