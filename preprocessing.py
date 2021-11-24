import os.path
import pandas as pd
import numpy as np

from datasets import Dataset
import transformers
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

import torch
from convokit import Corpus, download


TRAIN = "train"
TEST = "test"

# Tokenization constants
BERT_BASE_CASED = "bert-base-cased"

# BERT constants
INPUT_IDS = 'input_ids'
TOKEN_TYPE_IDS = 'token_type_ids'
ATTENTION_MASK = 'attention_mask'

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
    """

    :param cmv_corpus: A convokit corpus instance as described in https://convokit.cornell.edu/documentation/model.html.
    :param shortened_comment_id: A string
    :return:
    """
    for key in cmv_corpus.utterances.keys():
        if key[3:] == shortened_comment_id:
            return key
    raise Exception(f"Shortened comment id: {shortened_comment_id} was not found in corpus")


def find_ids_of_related_to_conversation_utterances(cmv_corpus, conversation_id, direct_reply_only=True):
    """

    :param cmv_corpus: A convokit corpus instance as described in https://convokit.cornell.edu/documentation/model.html.
    :param conversation_id: A string ID used to map to a conversation within the cmv_corpus.
    :param direct_reply_only: Boolean representing whether or not we should consider only direct replies to OP.
        If False, our dataset will also consist of utterances which are not direct replies to OP.
    :return: A set of utterance IDs representing utterances within a thread which respond to OP's initial opinion.
    """
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
    """

    :param related_utterances: A set of utterance IDs representing utterances within a thread which respond to OP's
        initial opinion.
    :return: A tuple containing two sets -- (positive_labeled_utterances, negative_labeled_utterances).
        * positive_labeled_utterances -- IDs of utterances which have earned a delta from OP (successful arguments).
        * negative_labeled_utterances -- IDs of utterances which did not earn a delta from OP.
    """
    positive_labeled_utterances = set()
    negative_labeled_utterances = set()

    for comment_id in related_utterances:
        success = cmv_corpus.utterances[comment_id].meta[SUCCESS]
        if success == 1:
            positive_labeled_utterances.add(comment_id)
        elif success == 0:
            negative_labeled_utterances.add(comment_id)

    return positive_labeled_utterances, negative_labeled_utterances


def balance_pos_and_neg_labels(positive_labeled, negative_labeled):
    """

    :param positive_labeled: IDs of utterances which have earned a delta from OP (successful arguments).
    :param negative_labeled: IDs of utterances which did not earn a delta from OP.
    :return: A tuple containing two sets -- (positive_labeled_utterances, negative_labeled_utterances) as above.
    """
    pos_len = len(positive_labeled)
    neg_len = len(negative_labeled)
    min_len = min(pos_len, neg_len)
    sampled_positive_labals = np.random.choice(positive_labeled, min_len)
    sampled_negative_labels = np.random.choice(negative_labeled, min_len)
    return sampled_positive_labals, sampled_negative_labels


def generate_data_points_from_related_utterances(cmv_corpus, orig_utter, pos_utter, neg_utter):
    """

    :param cmv_corpus: A convokit corpus instance as described in https://convokit.cornell.edu/documentation/model.html.
    :param orig_utter: The original utterance generated by OP.
    :param pos_utter: IDs of utterances which have earned a delta from OP (successful arguments).
    :param neg_utter: IDs of utterances which did not earn a delta from OP.
    :return: A 3-tuple consisting of (op_comment, reply, label).
        * op_comment -- A string containing the original comment written by OP.
        * reply -- A string containing a comment within the thread started by op_comment.
    """
    op_comment = [cmv_corpus.utterances[orig_utter].text] * 2 * len(pos_utter)
    label = [1] * len(pos_utter) + [0] * len(neg_utter)
    reply = []

    for utter in pos_utter:
        reply.append(cmv_corpus.utterances[utter].text)
    for utter in neg_utter:
        reply.append(cmv_corpus.utterances[utter].text)

    return op_comment, reply, label


def preprocess_corpus(cmv_corpus, direct_reply_only=False):
    """

    :param cmv_corpus: A convokit corpus instance as described in https://convokit.cornell.edu/documentation/model.html.
    :param direct_reply_only: Boolean representing whether or not we should consider only direct replies to OP.
        If False, our dataset will also consist of utterances which are not direct replies to OP.
    :return: A pandas DataFrame instance with the columns {OP_COMMENT, REPLY, LABEL}. The OP_COMMENT
        and REPLY columns consist of text entries while LABEL is {0, 1}.
    """
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
    :return:
    """
    tokenizer = transformers.BertTokenizer.from_pretrained(BERT_BASE_CASED)
    op_comments = list(corpus_df[OP_COMMENT])
    replies = list(corpus_df[REPLY])
    verbosity = transformers.logging.get_verbosity()
    # Ensure that each sequence of tokens is padded/truncated to transformer model limit.
    transformers.logging.set_verbosity_error()
    dataset = tokenizer(op_comments, replies, padding=True, truncation=True)
    transformers.logging.set_verbosity(verbosity)
    dataset[LABEL] = list(corpus_df[LABEL])
    return dataset


class CMVDataset(torch.utils.data.Dataset):
    def __init__(self, cmv_dataset):
        self.cmv_dataset = cmv_dataset.to_dict()
        self.num_examples = cmv_dataset.num_rows

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.cmv_dataset.items()}
        return item

    def __len__(self):
        return self.num_examples


if __name__ == "__main__":
    dataset_path = os.path.join(os.getcwd(), CMV_DATASET_NAME)
    if not (CMV_DATASET_NAME in os.listdir(os.getcwd())):
        cmv_corpus = Corpus(filename=download(CONVOKIT_DATASET_NAME))
        corpus_df = preprocess_corpus(cmv_corpus)
        corpus_df.to_json(dataset_path)
    else:
        corpus_df = pd.read_json(dataset_path)
    dataset = Dataset.from_dict(tokenize(corpus_df)).train_test_split()
    train_dataset = CMVDataset(dataset[TRAIN])
    test_dataset = CMVDataset(dataset[TEST])
    training_args = TrainingArguments(
        output_dir='./results',  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=10,
    )
    model = BertForSequenceClassification.from_pretrained(BERT_BASE_CASED, num_labels=2)
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=test_dataset  # evaluation dataset
    )
    trainer.train()
