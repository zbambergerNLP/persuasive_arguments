import os
from bs4 import BeautifulSoup
import pandas as pd
import constants
from random import sample

XML = "xml"
PREMISE = "premise"
TYPE = "type"

v1_path = os.path.join('cmv_modes', 'change-my-view-modes-master', 'v1.0')
v2_path = os.path.join('cmv_modes', 'change-my-view-modes-master', 'v2.0')

cmv_modes_versions = [v1_path]
cmv_modes_with_claims_versions = [v2_path]

POSITIVE = 'positive'
NEGATIVE = 'negative'
sign_lst = [POSITIVE, NEGATIVE]


def get_cmv_modes_corpus():
    """Create a pandas dataframe used to probe language models an argumentative premise modes.

    :return: A pandas DataFrame instance with two columns: [premise_text, premise_mode]. Both of these columns contain
        string value. The premise mode is in the superset of {ethos, logos, pathos}.
    """
    text = []
    label = []
    current_path = os.getcwd()
    for version in cmv_modes_versions:
        for sign in sign_lst:
            thread_directories = os.path.join(current_path, version, sign)
            for file_name in os.listdir(thread_directories):
                if file_name.endswith(XML):
                    with open(os.path.join(thread_directories, file_name), 'r') as f:
                        data = f.read()
                        bs_data = BeautifulSoup(data, XML)
                        premises = bs_data.find_all(PREMISE)
                        for premise in premises:
                            text.append(premise.contents[0])
                            label.append(premise.attrs[TYPE])
    return pd.DataFrame({constants.PREMISE_TEXT: text, constants.PREMISE_MODE: label})


def get_claim_and_premise_mode_corpus():
    """Create a pandas dataframe used to probe language models an argumentative premise modes (along with claims).

    :return: A pandas DataFrame instance with three columns: [claim_text, premise_text, premise_mode].
        The claim in the first column is the claim that the premise either supports or attacks. In reality, the claim
        con occur either before or after the premise. For simplicity, we always prepend the claim to the premise. The
        label is within the superset of {ethos, logos, pathos}.
    """
    claims_lst = []
    premises_lst = []
    label_lst = []
    current_path = os.getcwd()
    for sign in sign_lst:
        thread_directories = os.path.join(current_path, v2_path, sign)
        for file_name in os.listdir(thread_directories):
            if file_name.endswith(XML):
                with open(os.path.join(thread_directories, file_name), 'r') as f:
                    data = f.read()
                    bs_data = BeautifulSoup(data, XML)
                    premises = bs_data.find_all(PREMISE)
                    for premise in premises:
                        if "ref" in premise.attrs:
                            claim_id = premise.attrs["ref"]
                            claim = bs_data.find(id=claim_id)
                            claims_lst.append(claim.contents[0]) if claim else claims_lst.append('')
                        else:
                            claims_lst.append('')
                        premises_lst.append(premise.contents[0])
                        label_lst.append(premise.attrs[TYPE])
    return pd.DataFrame({
        constants.CLAIM_TEXT: claims_lst,
        constants.PREMISE_TEXT: premises_lst,
        constants.PREMISE_MODE: label_lst})


# TODO(Eli): Document this function and perhaps consider breaking it up into helper functions. Ensure that constants
#  are drawn from the constants.py file.
def get_claim_and_premise_with_relations_corpus(root_dir='/Users/zachary/PycharmProjects/persuasive_arguments/'):
    true_examples_1 = []
    true_examples_2 = []
    neg_examples_1 = []
    neg_examples_2 = []
    pos_distances = []
    neg_distances = []

    current_path = root_dir if root_dir else os.getcwd()
    for version in cmv_modes_with_claims_versions:
        for sign in sign_lst:
            thread_directories = os.path.join(current_path, version, sign)
            for file_name in os.listdir(thread_directories):
                if file_name.endswith(XML):
                    with open(os.path.join(thread_directories, file_name), 'r') as f:
                        data = f.read()
                        bs_data = BeautifulSoup(data, XML)
                        reply_or_op = bs_data.find_all(['OP', 'reply'])

                        for rep in reply_or_op:
                            res = rep.find_all(['premise', 'claim'])
                            id_to_idx = {}

                            for i, item in enumerate(res):
                                id_to_idx[item['id']] = i

                            idx_to_id = {value: key for key, value in id_to_idx.items()}

                            for comment in res:
                                attrs = vars(comment)
                                if 'ref' in comment.attrs and 'name' in attrs and comment['ref'] in id_to_idx and \
                                        comment['id'] in id_to_idx:
                                    ref_idx = id_to_idx[comment['ref']]
                                    comment_idx = id_to_idx[comment['id']]

                                    if comment.name == 'claim' and comment['ref'] in id_to_idx.keys():
                                        true_examples_1.append(comment.contents[0])
                                        true_examples_2.append(res[ref_idx].contents[0])
                                        pos_distance = abs(comment_idx - ref_idx) - 1
                                        pos_distances.append(pos_distance)

                                        if len(idx_to_id) > 2:
                                            random_false_idx = \
                                            sample(list(set(idx_to_id.keys()) ^ {ref_idx, comment_idx}), 1)[0]
                                            neg_examples_1.append(comment.contents[0])
                                            neg_examples_2.append(res[random_false_idx].contents[0])
                                            neg_distance = abs(random_false_idx - comment_idx) - 1
                                            neg_distances.append(neg_distance)

                                    elif comment.name == 'premise' and comment['ref'] in id_to_idx.keys():
                                        true_examples_1.append(res[ref_idx].contents[0])
                                        true_examples_2.append(comment.contents[0])
                                        pos_distance = abs(comment_idx - ref_idx) - 1
                                        pos_distances.append(pos_distance)

                                        if len(idx_to_id) > 2:
                                            random_false_idx = \
                                            sample(list(set(idx_to_id.keys()) ^ {ref_idx, comment_idx}), 1)[0]
                                            neg_examples_1.append(res[random_false_idx].contents[0])
                                            neg_examples_2.append(comment.contents[0])
                                            neg_distance = abs(random_false_idx - comment_idx) - 1
                                            neg_distances.append(neg_distance)

    min_len = min(len(true_examples_1), len(neg_examples_1))
    true_examples_1 = true_examples_1[:min_len]
    true_examples_2 = true_examples_2[:min_len]
    neg_examples_1 = neg_examples_1[:min_len]
    neg_examples_2 = neg_examples_2[:min_len]
    pos_distances = pos_distances[:min_len]
    neg_distances = neg_distances[:min_len]
    labels = [1] * min_len + [0] * min_len

    df = pd.DataFrame({
        constants.SENTENCE_1: true_examples_1 + neg_examples_1,
        constants.SENTENCE_2: true_examples_2 + neg_examples_2,
        constants.PREPOSITION_DISTANCE: pos_distances + neg_distances,
        constants.LABEL: labels
    })

    df = df[df[constants.SENTENCE_1] != df[constants.SENTENCE_2]]
    df = df.drop_duplicates()

    return df
