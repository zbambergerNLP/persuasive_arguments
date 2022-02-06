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


def tokenize_for_downstream_task(corpus_df, tokenizer):
    """

    :param corpus_df: A pandas DataFrame instance with the columns {OP_COMMENT, REPLY, LABEL}. The OP_COMMENT
        and REPLY columns consist of text entries while LABEL is {0, 1}.
    :param tokenizer: The pre-trained tokenizer (transformers.PreTrainedTokenizer) used to map words and word-pieces
        to token IDs.
    :return: A datasets.Dataset instance with features 'input_ids', 'token_type_ids', and 'attention_mask' and their
        corresponding values. The returned dataset also consists of binary labels.
    """
    op_comments = list(corpus_df[constants.OP_COMMENT])
    replies = list(corpus_df[constants.REPLY])

    verbosity = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()
    # Ensure that each sequence of tokens is padded/truncated to transformer model limit.
    dataset = tokenizer(op_comments, replies, padding=True, truncation=True)
    transformers.logging.set_verbosity(verbosity)

    dataset[constants.LABEL] = list(corpus_df[constants.LABEL])
    return dataset


def tokenize_for_premise_mode_probing(corpus_df, tokenizer, mode):
    """

    :param corpus_df: A pandas DataFrame instance with the columns {PREMISE_TEXT, PREMISE_MODE}. The PREMISE_MODE
        consists of text entries, while PREMISE_MODE is en entry in {LOGOS, ETHOS, PATHOS}.
    :param tokenizer: The pre-trained tokenizer (transformers.PreTrainedTokenizer) used to map words and word-pieces
        to token IDs.
    :param mode: A string in the set {'ethos', 'logos', 'pathos'}.
    :return: A datasets.Dataset instance with features 'input_ids', 'token_type_ids', and 'attention_mask' and their
        corresponding values. The returned dataset also consists of binary labels.
    """
    premise_text = list(corpus_df[constants.PREMISE_TEXT])
    dataset = tokenizer(premise_text, padding=True, truncation=True)
    dataset[constants.LABEL] = [1 if mode in premise_mode else 0 for premise_mode in corpus_df[constants.PREMISE_MODE]]
    return dataset


def tokenize_for_multi_class_premise_mode_probing(corpus_df, tokenizer, with_claim):
    """

    :param corpus_df: A pandas DataFrame instance with the columns {PREMISE_TEXT, PREMISE_MODE}. The PREMISE_MODE
        consists of text entries, while PREMISE_MODE is en entry in {LOGOS, ETHOS, PATHOS}.
    :param tokenizer: The pre-trained tokenizer (transformers.PreTrainedTokenizer) used to map words and word-pieces
        to token IDs.
    :param with_claim: True if we intend to tokenize the claim that the premise of interest either supports or attacks.
        We are still interested in classifying the argumentative mode of the premise (the second entry to the
        transformer encoder model).
    :return: A datasets.Dataset instance with features 'input_ids', 'token_type_ids', and 'attention_mask' and their
        corresponding values. The returned dataset has a label space with dimensionality equal to the cardinality of
        the superset of {LOGOS, ETHOS, PATHOS}.
    """
    premise_text = list(corpus_df[constants.PREMISE_TEXT])
    if with_claim:
        claim_text = list(corpus_df[constants.CLAIM_TEXT])
        dataset = tokenizer(premise_text, claim_text, padding=True, truncation=True)
    else:
        dataset = tokenizer(premise_text, padding=True, truncation=True)
    dataset[constants.LABEL] = (
        [constants.PREMISE_MODE_TO_INT[premise_mode] for premise_mode in corpus_df[constants.PREMISE_MODE]])
    return dataset


def tokenize_from_premise_mode_probing_with_claim(corpus_df, tokenizer, mode):
    """

    :param corpus_df: A pandas DataFrame instance with the columns {CLAIM_TEXT, PREMISE_TEXT, PREMISE_MODE}.
        The PREMISE_MODE consists of text entries, while PREMISE_MODE is en entry in {LOGOS, ETHOS, PATHOS}.
    :param tokenizer: The pre-trained tokenizer (transformers.PreTrainedTokenizer) used to map words and word-pieces
        to token IDs.
    :param mode: A string in the set {'ethos', 'logos', 'pathos'}.
    :return: A datasets.Dataset instance with features 'input_ids', 'token_type_ids', and 'attention_mask' and their
        corresponding values.
    """
    premise_text = list(corpus_df[constants.PREMISE_TEXT])
    claim_text = list(corpus_df[constants.CLAIM_TEXT])
    dataset = tokenizer(premise_text, claim_text, padding=True, truncation=True)
    dataset[constants.LABEL] = [1 if mode in premise_mode else 0 for premise_mode in corpus_df[constants.PREMISE_MODE]]
    return dataset


def tokenize_for_intra_relation_probing(corpus_df, tokenizer):
    """

    :param corpus_df:
    :param tokenizer:
    :return:
    """
    sentence_1_text = list(corpus_df[constants.SENTENCE_1])
    sentence_2_text = list(corpus_df[constants.SENTENCE_2])
    dataset = tokenizer(sentence_1_text, sentence_2_text, padding=True, truncation=True)
    dataset[constants.LABEL] = list(corpus_df[constants.LABEL])
    return dataset


class CMVDataset(torch.utils.data.Dataset):
    """A Change My View specific wrapper to the Torch Dataset class."""

    def __init__(self, cmv_dataset):
        self.cmv_dataset = cmv_dataset.to_dict()
        self.num_examples = cmv_dataset.num_rows

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.cmv_dataset.items()}
        return item

    def __len__(self):
        return self.num_examples


class CMVPremiseModes(torch.utils.data.Dataset):
    """A Change My View specific wrapper to the Torch Dataset class.

     Unlike CMVDataset, premises are preceded by their respective claims."""
    def __init__(self, cmv_premise_mode_dataset):
        self.cmv_premise_mode_dataset = cmv_premise_mode_dataset.to_dict()
        self.num_examples = cmv_premise_mode_dataset.num_rows

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.cmv_premise_mode_dataset.items()}
        return item

    def __len__(self):
        return self.num_examples


def get_cmv_downstream_dataset(dataset_name, tokenizer):
    """Create a dataset mapping textual arguments to their persuasiveness label (whether or not the argument was awarded
    a delta).

    :param dataset_name: The name with which the CMV dataset json object is saved (in the local directory).
    :param tokenizer: The pre-trained tokenizer (transformers.PreTrainedTokenizer) used to map words and word-pieces
        to token IDs.
    :return: A transformers.Dataset instance containing a CMV dataset mapping arguments to their binary persuasiveness
            label.
    """
    dataset_path = os.path.join(os.getcwd(), dataset_name)
    if not (dataset_name in os.listdir(os.getcwd())):
        cmv_corpus = Corpus(filename=download(constants.CONVOKIT_DATASET_NAME))
        corpus_df = preprocess_corpus(cmv_corpus)
        corpus_df.to_json(dataset_path)
    else:
        corpus_df = pd.read_json(dataset_path)
    return Dataset.from_dict(tokenize_for_downstream_task(corpus_df=corpus_df, tokenizer=tokenizer))


def get_intra_argument_relations_probing_dataset(tokenizer, save_text_datasets=False):
    """

    :param tokenizer:
    :param save_text_datasets:
    :return:
    """
    corpus_df = cmv_probing.get_claim_and_premise_with_relations_corpus()
    return transform_binary_cmv_relations_df_to_dataset(
        corpus_df,
        tokenizer,
        to_dict_func=tokenize_for_intra_relation_probing,
        save_text_datasets=save_text_datasets)


def get_cmv_probing_datasets(tokenizer, save_text_datasets=False, with_claims=False):
    """Create a pandas dataframe containing a CMV premise mode corpus, and transform it into three datasets.
    Each of these three datasets (corresponding to ETHOS, LOGOS, and PATHOS) maps argumentative text to whether or not
    that text employs the given argumentative mode.

    :param tokenizer: The pre-trained tokenizer (transformers.PreTrainedTokenizer) used to map words and word-pieces
        to token IDs.
    :return: A transformers.Dataset instance containing a mapping from argumentative premises to the argumentative
        modes they employ.
    """
    if with_claims:
        corpus_df = cmv_probing.get_cmv_modes_corpus()
        to_dict_func = tokenize_for_premise_mode_probing
    else:
        corpus_df = cmv_probing.get_claim_and_premise_mode_corpus()
        to_dict_func = tokenize_from_premise_mode_probing_with_claim

    ethos_dataset = (
        transform_binary_cmv_premise_mode_df_to_dataset(
            corpus_df,
            tokenizer,
            mode=constants.ETHOS,
            to_dict_func=to_dict_func,
            save_text_datasets=save_text_datasets))
    logos_dataset = (
        transform_binary_cmv_premise_mode_df_to_dataset(
            corpus_df,
            tokenizer,
            mode=constants.LOGOS,
            to_dict_func=to_dict_func,
            save_text_datasets=save_text_datasets))
    pathos_dataset = (
        transform_binary_cmv_premise_mode_df_to_dataset(
            corpus_df,
            tokenizer,
            mode=constants.PATHOS,
            to_dict_func=to_dict_func,
            save_text_datasets=save_text_datasets))
    return ethos_dataset, logos_dataset, pathos_dataset


def get_multi_class_cmv_probing_dataset(tokenizer, with_claims, save_text_datasets=False):
    """Transform a pandas dataframe containing a CMV premise mode corpus into three datasets. Each of these three
    datasets (corresponding to ETHOS, LOGOS, and PATHOS) maps argumentative text to whether or not that text employs
    the given argumentative mode.

    :param tokenizer: The pre-trained tokenizer (transformers.PreTrainedTokenizer) used to map words and word-pieces
        to token IDs.
    :param save_text_datasets: True if we want to store the probing dataset in textual form in the appropriate
        directory (i.e., './probing/{mode}'}.
    :return: A transformer.Dataset instance with columns corresponding to BERT inputs and a label. The sentences
        provided as an input are drawn from claim + premise pairs. The label is the enumerated argumentative mode
        of the premise. The possible argumentative modes are a superset of {ETHOS, LOGOS, PATHOS}.
    """
    if with_claims:
        corpus_df = cmv_probing.get_claim_and_premise_mode_corpus()
    else:
        corpus_df = cmv_probing.get_cmv_modes_corpus()
    return transform_multi_class_cmv_premise_mode_df_to_dataset(corpus_df=corpus_df,
                                                                tokenizer=tokenizer,
                                                                to_dict_func=tokenize_for_multi_class_premise_mode_probing,
                                                                with_claims=with_claims,
                                                                save_text_datasets=save_text_datasets)


# TODO: Update documentation to this function.
# TODO: Make the CSV file name a parameter, or at least more descriptive.
def transform_multi_class_cmv_premise_mode_df_to_dataset(corpus_df,
                                                         tokenizer,
                                                         to_dict_func,
                                                         with_claims,
                                                         save_text_datasets=False):
    """Transform a dataframe (which maps argumentative text to the argumentative mode it employs) into a
    transformers.Dataset instance. Argumentative mode labels are integers pointing to an element in the superset of
    {ETHOS, LOGOS, PATHOS}.

    :param corpus_df: A pandas DataFrame instance with columns 'input_ids', 'token_type_ids', 'attention_mask', and
        'label' and their corresponding values. The label is a string which might contain the substring 'mode'.
    :param tokenizer: The pre-trained tokenizer (transformers.PreTrainedTokenizer) used to map words and word-pieces
        to token IDs.
    :param to_dict_func: A function mapping the string labels of premises to integers. In the case of binary
        classification, these integer labels are 0 and 1. In the multi-class case, these labels are integer
        representations of possible combinations of premise modes (e.g., 'ethos' as opposed to 'ethos_logos').
    :param with_claims:
    :param save_text_datasets: True if we want to store the probing dataset in textual form in the appropriate
        directory (i.e., './probing/{mode}'}.
    :return: A transformers.Dataset instance with columns corresponding to 'input_ids', 'token_type_ids',
        'attention_mask', and 'label' as well as their corresponding values. The label for this dataset is 1 if the
        claim + premise pair contains the label `mode`, and 0 otherwise.
    """
    if save_text_datasets:
        corpus_df.to_csv('corpus_dataframe_tmp.csv', index=False)
        print(f'wrote pandas dataframe into memory: {os.path.join(os.getcwd(), "corpus_dataframe_tmp.csv")}')

    baseline_results = metrics.get_baseline_scores(corpus_df)
    print(f'Baseline results for multiclass prediction are:\n{baseline_results}')

    dataset = Dataset.from_dict(
        to_dict_func(
            corpus_df=corpus_df,
            tokenizer=tokenizer,
            with_claim=with_claims,
        ))
    dataset.set_format(type='torch',
                       columns=[
                           constants.INPUT_IDS,
                           constants.TOKEN_TYPE_IDS,
                           constants.ATTENTION_MASK,
                           constants.LABEL])
    return dataset


def transform_binary_cmv_premise_mode_df_to_dataset(
        corpus_df,
        tokenizer,
        mode,
        to_dict_func,
        save_text_datasets=False):
    """Create three datasets mapping argumentative text to whether or not that text entails a given argumentative mode.

    :param corpus_df: A pandas DataFrame instance with columns 'input_ids', 'token_type_ids', 'attention_mask', and
        'label' and their corresponding values. The label is a string which might contain the substring 'mode'.
    :param tokenizer: The pre-trained tokenizer (transformers.PreTrainedTokenizer) used to map words and word-pieces
        to token IDs.
    :param mode: A string describing the argumentation mode of the premise. Possible labels are a superset of
        {"ethos", "logos", "pathos"}.
    :param to_dict_func: A function mapping the string labels of premises to integers. In the case of binary
        classification, these integer labels are 0 and 1. In the multi-class case, these labels are integer
        representations of possible combinations of premise modes (e.g., 'ethos' as opposed to 'ethos_logos').
    :param save_text_datasets: True if we want to store the probing dataset in textual form in the appropriate
        directory (i.e., './probing/{mode}'}.
    :return: A transformers.Dataset instance with columns corresponding to 'input_ids', 'token_type_ids',
        'attention_mask', and 'label' as well as their corresponding values. The label for this dataset is 1 if the
        claim + premise pair contains the label `mode`, and 0 otherwise.
    """

    if save_text_datasets:
        corpus_df.to_csv('corpus_dataframe_tmp.csv', index=False)
        print(f'wrote pandas dataframe into memory: {os.path.join(os.getcwd(), "corpus_dataframe_tmp.csv")}')

    baseline_results = metrics.get_baseline_scores(corpus_df)
    print(f'Baseline results for {mode} binary prediction are:')
    print(baseline_results)

    dataset = Dataset.from_dict(
        to_dict_func(
            corpus_df=corpus_df,
            tokenizer=tokenizer,
            mode=mode))
    dataset.set_format(type='torch',
                       columns=[
                           constants.INPUT_IDS,
                           constants.TOKEN_TYPE_IDS,
                           constants.ATTENTION_MASK,
                           constants.LABEL])
    return dataset


def transform_binary_cmv_relations_df_to_dataset(
        corpus_df,
        tokenizer,
        to_dict_func,
        save_text_datasets=False):
    """

    :param corpus_df: A pandas DataFrame instance with columns 'sentence_1', 'sentence_2', 'distance', and
        'label' and their corresponding values. The label is a binary integer representing whether the there exists
        a relation from 'sentence_2' to 'sentence_1'. 'distance' is an integer corresponding to the distance between
        the two nodes.
    :param tokenizer: The pre-trained tokenizer (transformers.PreTrainedTokenizer) used to map words and word-pieces
        to token IDs.
    :param to_dict_func: A function mapping the corpus dataframe to a dictionary.
    :param save_text_datasets: True if we want to store the probing dataset in textual form in the appropriate
        directory (i.e., './probing/{mode}'}.
    :return: A transformers.Dataset instance with columns corresponding to 'input_ids', 'token_type_ids',
        'attention_mask', and 'label' as well as their corresponding values.
    """

    # TODO: Name of the produced CSV file should reflect the content of the file. What dataset is this? How would
    #  you differentiate it from other datasets?
    if save_text_datasets:
        corpus_df.to_csv('corpus_dataframe_tmp.csv', index=False)
        print(f'wrote pandas dataframe into memory: {os.path.join(os.getcwd(), "corpus_dataframe_tmp.csv")}')

    # TODO: Extend get_baseline_scores to the binary relation prediction task.
    baseline_results = metrics.get_baseline_scores(corpus_df)
    print('Baseline results for binary relation prediction are:')
    print(baseline_results)

    dataset = Dataset.from_dict(
        to_dict_func(
            corpus_df=corpus_df,
            tokenizer=tokenizer))
    dataset.set_format(type='torch',
                       columns=[
                           constants.INPUT_IDS,
                           constants.TOKEN_TYPE_IDS,
                           constants.ATTENTION_MASK,
                           constants.LABEL])
    return dataset
