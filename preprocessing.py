import typing

import datasets
import numpy as np
import os.path
import pandas as pd

from convokit import Corpus, download
from datasets import Dataset
import torch
import transformers

import cmv_modes.preprocessing_cmv_ampersand as cmv_probing
import constants
import metrics


class CMVProbingDataset(torch.utils.data.Dataset):
    """A Change My View dataset for probing."""

    def __init__(self, cmv_probing_dataset):
        self.cmv_probing_dataset = cmv_probing_dataset.to_dict()
        self.hidden_states = cmv_probing_dataset[constants.HIDDEN_STATE]
        self.labels = cmv_probing_dataset[constants.LABEL]
        self.num_examples = cmv_probing_dataset.num_rows

    def __getitem__(self, idx):
        return {constants.HIDDEN_STATE: torch.tensor(self.hidden_states[idx]),
                constants.LABEL: torch.tensor(self.labels[idx])}

    def __len__(self):
        return self.num_examples


class CMVDataset(torch.utils.data.Dataset):
    """A Change My View dataset for fine tuning.."""

    def __init__(self, cmv_dataset):
        self.cmv_dataset = cmv_dataset.to_dict()
        self.num_examples = cmv_dataset.num_rows

    def __getitem__(self, idx):
        item = {}
        for key, value in self.cmv_dataset.items():
            if key in [constants.INPUT_IDS, constants.TOKEN_TYPE_IDS, constants.ATTENTION_MASK, constants.LABEL]:
                item[key] = torch.tensor(value[idx])
        return item

    def __len__(self):
        return self.num_examples


def find_comment_id_by_shortened_id(cmv_corpus, shortened_comment_id):
    """

    :param cmv_corpus: A convokit corpus instance as described in https://convokit.cornell.edu/documentation/model.html.
    :param shortened_comment_id: A string
    :raises Exception if shortened_comment_id is not found within they cmv_corpus.
    :return: The utterance ID corresponding to the shortened comment ID.
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
        for shortened_comment_id in cmv_corpus.utterances[conversation_id].meta[constants.REPLIES]:
            full_comment_id = find_comment_id_by_shortened_id(cmv_corpus, shortened_comment_id)
            related_utterances.add(full_comment_id)
    else:
        for key, val in cmv_corpus.utterances.items():
            if val.conversation_id == conversation_id and key != val.conversation_id:
                related_utterances.add(key)

    return related_utterances


def divide_to_positive_and_negative_lists(cmv_corpus, related_utterances):
    """

    :param cmv_corpus: A convokit corpus instance as described in https://convokit.cornell.edu/documentation/model.html.
    :param related_utterances: A set of utterance IDs representing utterances within a thread which respond to OP's
        initial opinion.
    :return: A tuple containing two sets -- (positive_labeled_utterances, negative_labeled_utterances).
        * positive_labeled_utterances -- IDs of utterances which have earned a delta from OP (successful arguments).
        * negative_labeled_utterances -- IDs of utterances which did not earn a delta from OP.
    """
    positive_labeled_utterances = set()
    negative_labeled_utterances = set()

    for comment_id in related_utterances:
        success = cmv_corpus.utterances[comment_id].meta[constants.SUCCESS]
        if success == 1:
            positive_labeled_utterances.add(comment_id)
        elif success == 0:
            negative_labeled_utterances.add(comment_id)

    return positive_labeled_utterances, negative_labeled_utterances


def balance_pos_and_neg_labels(positive_labeled, negative_labeled):
    """Given non-balanced sets of positive and negative utterances, we return a randomly sampled balanced version of
    these sets.

    :param positive_labeled: IDs of utterances which have earned a delta from OP (successful arguments).
    :param negative_labeled: IDs of utterances which did not earn a delta from OP.
    :return: A tuple containing two sets -- (positive_labeled_utterances, negative_labeled_utterances) as above.
    """
    min_len = min(len(positive_labeled), len(negative_labeled))
    sampled_positive_labals = np.random.choice(list(positive_labeled), min_len)
    sampled_negative_labels = np.random.choice(list(negative_labeled), min_len)
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
    """Create a pandas Dataframe representing the Reddit CMV Dataset.

    The produced dataframe maps pairs of OP_COMMENT and REPLY utterances to whether or not the REPLY earned a "Delta."

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
        pos_list, neg_list = divide_to_positive_and_negative_lists(cmv_corpus, related_utterances)
        balanced_pos, balanced_neg = balance_pos_and_neg_labels(pos_list, neg_list)
        op, rep, lab = generate_data_points_from_related_utterances(cmv_corpus,
                                                                    conversation_id,
                                                                    balanced_pos,
                                                                    balanced_neg)
        op_comment += op
        reply += rep
        label += lab

    corpus_df = pd.DataFrame({constants.OP_COMMENT: op_comment, constants.REPLY: reply, constants.LABEL: label})
    return corpus_df


def tokenize_for_task(task_name: str,
                      corpus_df: pd.DataFrame,
                      tokenizer: transformers.PreTrainedTokenizer,
                      premise_mode: str = None) -> typing.Mapping[str, torch.tensor]:
    """

    :param task_name: A string. One of {'multiclass', 'binary_premise_mode_prediction', 'intra_argument_relations',
        'binary_cmv_delta_prediction'}
    :param corpus_df: A pandas Dataframe instances with the following columns depending on the task:
        binary_cmv_delta_prediction columns: {OP_COMMENT, REPLY, LABEL}
            * OP_COMMENT and REPLY are text entries.
            * LABEL is a binary label that is True if the reply was given a "delta".
        binary_premise_mode_prediction OR multiclass columns: {CLAIM_TEXT, PREMISE_TEXT, PREMISE_MODE}
            * PREMISE_MODE the consists of text entries
            * PREMISE_MODE is en entry in {LOGOS, ETHOS, PATHOS}.
            * LABEL is a binary label if `multiclass` is False. If `multiclass` is False, then True examples are ones
              whose premises consist of the argumentative mode (ethos/logos/pathos) currently under study. If
              `multiclass` is True, then the label is an integer representing the exact combination of premise modes
              which the premise exhibits.
        intra_argument_relations columns: {SENTENCE_1, SENTENCE_2, LABEL}.
            * SENTENCE_1 and SENTENCE_2 both consist of argument prepositions (either claims or premises).
            * LABEL is a binary label which signifies whether there is a relation starting from SENTENCE_1 and directed
              towards SENTENCE_2. The label of a given example is True if such a relation exists, and False otherwise.
    :param tokenizer: The pre-trained tokenizer (transformers.PreTrainedTokenizer) used to map words and word-pieces
        to token IDs.
    :param premise_mode: A string in the set {'ethos', 'logos', 'pathos'}.
    :return: A datasets.Dataset instance with features 'input_ids', 'token_type_ids', and 'attention_mask' and their
        corresponding values. The returned dataset has labels of the same type as the labels in the inputted dataframe.
    """
    assert task_name in [constants.BINARY_CMV_DELTA_PREDICTION,
                         constants.BINARY_PREMISE_MODE_PREDICTION,
                         constants.MULTICLASS,
                         constants.INTRA_ARGUMENT_RELATIONS], f"{task_name} is not supported."

    if task_name == constants.BINARY_CMV_DELTA_PREDICTION:
        first_text = list(corpus_df[constants.OP_COMMENT])
        second_text = list(corpus_df[constants.REPLY])
        label = list(corpus_df[constants.LABEL])
    elif task_name == constants.INTRA_ARGUMENT_RELATIONS:
        first_text = list(corpus_df[constants.SENTENCE_1])
        second_text = list(corpus_df[constants.SENTENCE_2])
        label = list(corpus_df[constants.LABEL])
    else:
        first_text = list(corpus_df[constants.CLAIM_TEXT])
        second_text = list(corpus_df[constants.PREMISE_TEXT])
        if task_name == constants.MULTICLASS:
            label = [constants.PREMISE_MODE_TO_INT[premise_mode] for premise_mode in corpus_df[constants.PREMISE_MODE]]
        else:
            label = [1 if premise_mode in premise_label else 0 for premise_label in corpus_df[constants.PREMISE_MODE]]

    verbosity = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()
    dataset = tokenizer(first_text, second_text, padding=True, truncation=True)
    transformers.logging.set_verbosity(verbosity)

    dataset[constants.FIRST_TEXT] = first_text
    dataset[constants.SECOND_TEXT] = second_text

    dataset[constants.LABEL] = label
    return dataset


def transform_df_to_dataset(task_name: str,
                            corpus_df: pd.DataFrame,
                            tokenizer: transformers.PreTrainedTokenizer,
                            save_text_datasets: bool,
                            premise_mode=None) -> datasets.Dataset:
    """

    :param task_name: A string. One of {'multiclass', 'binary_premise_mode_prediction', 'intra_argument_relations',
        'binary_cmv_delta_prediction'}
    :param corpus_df: A pandas Dataframe instances with the following columns depending on the task:
        binary_cmv_delta_prediction columns: {OP_COMMENT, REPLY, LABEL}
            * OP_COMMENT and REPLY are text entries.
            * LABEL is a binary label that is True if the reply was given a "delta".
        binary_premise_mode_prediction OR multiclass columns: {CLAIM_TEXT, PREMISE_TEXT, PREMISE_MODE}
            * PREMISE_MODE the consists of text entries
            * PREMISE_MODE is en entry in {LOGOS, ETHOS, PATHOS}.
            * LABEL is a binary label if `multiclass` is False. If `multiclass` is False, then True examples are ones
              whose premises consist of the argumentative mode (ethos/logos/pathos) currently under study. If
              `multiclass` is True, then the label is an integer representing the exact combination of premise modes
              which the premise exhibits.
        intra_argument_relations columns: {SENTENCE_1, SENTENCE_2, LABEL}.
            * SENTENCE_1 and SENTENCE_2 both consist of argument prepositions (either claims or premises).
            * LABEL is a binary label which signifies whether there is a relation starting from SENTENCE_1 and directed
              towards SENTENCE_2. The label of a given example is True if such a relation exists, and False otherwise.
    :param tokenizer: The pre-trained tokenizer (transformers.PreTrainedTokenizer) used to map words and word-pieces
        to token IDs.
    :param save_text_datasets: True if we want to store the probing dataset in textual form in the appropriate
        directory (i.e., './probing/{mode}'}.
    :param premise_mode: A string in the set {'ethos', 'logos', 'pathos'}.
    :return: A datasets.Dataset instance for the inputted task.
    """
    if save_text_datasets:
        corpus_df.to_csv(f'{task_name}.csv', index=False)
        print(f'wrote pandas dataframe into memory: {os.path.join(os.getcwd(), f"{task_name}.csv")}')

    baseline_results = metrics.get_baseline_scores(task_name=task_name, corpus_df=corpus_df)
    print(f'Baseline results for {task_name} prediction are:\n{baseline_results}')

    dataset = Dataset.from_dict(
        tokenize_for_task(
            task_name=task_name,
            corpus_df=corpus_df,
            tokenizer=tokenizer,
            premise_mode=premise_mode))
    dataset.set_format(type='torch',
                       columns=[
                           constants.INPUT_IDS,
                           constants.TOKEN_TYPE_IDS,
                           constants.ATTENTION_MASK,
                           constants.LABEL])
    return dataset


def get_dataset(task_name: str,
                tokenizer: transformers.PreTrainedTokenizer,
                save_text_datasets: bool = False,
                dataset_name: str = None,
                premise_mode: str = None) -> datasets.Dataset:
    """

    :param task_name: A string. One of {'multiclass', 'binary_premise_mode_prediction', 'intra_argument_relations',
        'binary_cmv_delta_prediction'}
    :param tokenizer: The pre-trained tokenizer (transformers.PreTrainedTokenizer) used to map words and word-pieces
        to token IDs.
    :param save_text_datasets: True if we want to store the probing dataset in textual form in the appropriate
        directory (i.e., './probing/{mode}'}.
    :param dataset_name: The name with which the CMV dataset json object is saved (in the local directory).
    :param premise_mode: A string in the set {'ethos', 'logos', 'pathos'}.
    :return: A datasets.Dataset instance for the inputted task.
    """
    if task_name == constants.INTRA_ARGUMENT_RELATIONS:
        corpus_df = cmv_probing.get_claim_and_premise_with_relations_corpus()
    elif task_name == constants.BINARY_CMV_DELTA_PREDICTION:
        dataset_path = os.path.join(os.getcwd(), dataset_name)
        if dataset_name and dataset_name in os.listdir(os.getcwd()):
            corpus_df = pd.read_json(dataset_path)
        else:
            cmv_corpus = Corpus(filename=download(constants.CONVOKIT_DATASET_NAME))
            corpus_df = preprocess_corpus(cmv_corpus)
            corpus_df.to_json(dataset_path)
    elif task_name == constants.INTRA_ARGUMENT_RELATIONS:
        corpus_df = cmv_probing.get_claim_and_premise_with_relations_corpus()
    elif task_name == constants.MULTICLASS:
        corpus_df = cmv_probing.get_claim_and_premise_mode_corpus()
    else:  # Binary premise mode prediction.
        corpus_df = cmv_probing.get_claim_and_premise_mode_corpus()
    return transform_df_to_dataset(task_name=task_name,
                                   corpus_df=corpus_df,
                                   tokenizer=tokenizer,
                                   save_text_datasets=save_text_datasets,
                                   premise_mode=premise_mode)
