import os
import typing

import torch
import transformers
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.data import HeteroData

import constants
from cmv_modes.preprocessing_knowledge_graph import make_op_reply_graphs, create_bert_inputs, GraphExample, find_op_ukp, \
    find_label_ukp, parse_ann_file


# TODO: Ensure that both the BERT model and its corresponding tokenizer are accessible even without internet connection.


class CMVProbingDataset(torch.utils.data.Dataset):
    """A Change My View dataset for probing."""

    def __init__(self, cmv_probing_dataset):
        """

        :param cmv_probing_dataset:
        """
        self.cmv_probing_dataset = cmv_probing_dataset.to_dict()
        self.hidden_states = cmv_probing_dataset[constants.HIDDEN_STATE]
        self.labels = cmv_probing_dataset[constants.LABEL]
        self.num_examples = cmv_probing_dataset.num_rows

    def __getitem__(self, idx):
        """

        :param idx:
        :return:
        """
        return {constants.HIDDEN_STATE: torch.tensor(self.hidden_states[idx]),
                constants.LABEL: torch.tensor(self.labels[idx])}

    def __len__(self):
        return self.num_examples


class CMVDataset(torch.utils.data.Dataset):
    """A Change My View dataset for fine tuning.."""

    def __init__(self, cmv_dataset):
        """

        :param cmv_dataset:
        """
        self.cmv_dataset = cmv_dataset.to_dict()
        self.num_examples = cmv_dataset.num_rows

    def __getitem__(self, idx):
        """

        :param idx:
        :return:
        """
        item = {}
        for key, value in self.cmv_dataset.items():
            if key in [constants.INPUT_IDS, constants.TOKEN_TYPE_IDS, constants.ATTENTION_MASK, constants.LABEL]:
                item[key] = torch.tensor(value[idx])
        return item

    def __len__(self):
        return self.num_examples


class BaselineLoader(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        """

        :param features:
        :param labels:
        """
        self.features = features
        self.labels = labels
        self.num_examples = len(labels)

    def __getitem__(self, item):
        """

        :param item:
        :return:
        """
        return {'features': self.features[item],
                constants.LABEL: self.labels[item]}

    def __len__(self):
        return self.num_examples


class CMVKGDataset(torch.utils.data.Dataset):
    """

    """

    def __init__(self,
                 directory_path: str,
                 version: str,
                 model_name: str = constants.BERT_BASE_CASED,
                 debug: bool = False):
        """

        :param directory_path: The string path to the 'change-my-view-modes-master' directory, which contains versions
            versions of the change my view dataset.
        :param version: A version of the cmv datasets (i.e.. one of 'v2.0', 'v1.0', and 'original') included within the
            'chamge-my-view-modes-master' directory.
        :param model_name: The name of the model corresponding to the desired tokenizer.
        :param debug: A boolean denoting whether or not we are in debug mode (in which our input dataset is
            significantly smaller).
        """
        super(CMVKGDataset, self).__init__()
        self.dataset = []
        self.labels = []
        self.model_name = model_name
        for sign in constants.SIGN_LIST:
            thread_directory = os.path.join(directory_path, version, sign)
            for file_name in tqdm(os.listdir(thread_directory)):
                if file_name.endswith(constants.XML):
                    file_path = os.path.join(thread_directory, file_name)
                    with open(file_path, 'r') as fileHandle:
                        data = fileHandle.read()
                        bs_data = BeautifulSoup(data, constants.XML)
                        examples = make_op_reply_graphs(
                            bs_data=bs_data,
                            file_name=file_name,
                            is_positive=(sign == constants.POSITIVE))
                        examples = create_bert_inputs(examples,
                                                      tokenizer=transformers.AutoTokenizer.from_pretrained(model_name))
                        self.dataset.extend(examples)
                        example_labels = list(map(lambda example: 0 if sign == 'negative' else 1, examples))
                        self.labels.extend(example_labels)
                        if debug:
                            if len(self.labels) >= 20:
                                break

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        """

        :param index:
        :return:
        """
        bert_input_key_names = [f'id_to_{constants.INPUT_IDS}',
                                f'id_to_{constants.TOKEN_TYPE_IDS}',
                                f'id_to_{constants.ATTENTION_MASK}']

        # SBERT does not consider two inputs in the way BERT does, so the "token type ID" input is not necessary.
        if self.model_name == "sentence-transformers/all-distilroberta-v1":
            bert_input_key_names.pop(1)

        formatted_bert_inputs = {}
        for input_name in bert_input_key_names:
            formatted_bert_inputs[input_name] = torch.cat(
                [ids.unsqueeze(dim=1) for ids in self.dataset[index][input_name].values()],
                dim=1,
            )
        stacked_bert_inputs = torch.stack([t for t in formatted_bert_inputs.values()], dim=1)
        return Data(x=stacked_bert_inputs.T,
                    edge_index=torch.tensor(self.dataset[index]['edges']).T,
                    y=torch.tensor(self.labels[index]))


class CMVKGHetroDataset(CMVKGDataset):
    """

    """

    def __init__(self,
                 directory_path: str,
                 version: str,
                 debug: bool = False):
        """

        :param directory_path:
        :param version:
        :param debug:
        """
        super(CMVKGHetroDataset, self).__init__(directory_path=directory_path, version=version, debug=debug)

    @staticmethod
    def calc_bert_inputs(dataset_values, relevant_ids=None):
        """

        :param dataset_values:
        :param relevant_ids:
        :return:
        """
        if relevant_ids is None:
            relevant_ids = list(range(len(dataset_values[constants.INDEX_TO_ID])))
        bert_input_key_names = [f'id_to_{constants.INPUT_IDS}',
                                f'id_to_{constants.TOKEN_TYPE_IDS}',
                                f'id_to_{constants.ATTENTION_MASK}']
        formatted_bert_inputs = {}
        for input_name in bert_input_key_names:
            for key in dataset_values[input_name]:
                if key in relevant_ids:
                    if input_name not in formatted_bert_inputs.keys():
                        formatted_bert_inputs[input_name] = dataset_values[input_name][key].unsqueeze(dim=1)
                    else:
                        formatted_bert_inputs[input_name] = torch.cat(
                            (formatted_bert_inputs[input_name], dataset_values[input_name][key].unsqueeze(dim=1)),
                            dim=1)

        stacked_bert_inputs = torch.stack([t for t in formatted_bert_inputs.values()], dim=1)
        return stacked_bert_inputs

    @staticmethod
    def rearrange_edge_index(out_list: list,
                             in_list: list,
                             out_idx: int,
                             in_idx: int):
        """

        :param out_list:
        :param in_list:
        :param out_idx:
        :param in_idx:
        :return:
        """
        new_idx_out_node = out_list.index(out_idx)
        new_idx_in_node = in_list.index(in_idx)
        return [new_idx_out_node, new_idx_in_node]

    @staticmethod
    def convert_edge_indexes(edge_list):
        """

        :param edge_list:
        :return:
        """
        # Update edges after a claim node and a premise node were added.
        # Add an edge from every added node to the other so each graph will have all kinds of edges.
        edge_list = torch.tensor(edge_list, dtype=torch.long).T + 2
        edge_to_add = torch.tensor([0, 1]).unsqueeze(dim=1)
        edge_list = torch.concat((edge_to_add, edge_list), dim=1)
        return edge_list

    def __getitem__(self, index: int) -> HeteroData:
        claim_list = []
        premise_list = []
        for key in self.dataset[index][constants.ID_TO_NODE_TYPE]:
            if self.dataset[index][constants.ID_TO_NODE_TYPE][key] == constants.CLAIM:
                claim_list.append(self.dataset[index][constants.ID_TO_INDEX][key])
            elif self.dataset[index][constants.ID_TO_NODE_TYPE][key] == constants.PREMISE:
                premise_list.append(self.dataset[index][constants.ID_TO_INDEX][key])
            else:
                raise Exception("unknown value = " + key)

        stacked_bert_inputs_claim = self.calc_bert_inputs(self.dataset[index], claim_list)
        stacked_bert_inputs_premise = self.calc_bert_inputs(self.dataset[index], premise_list)

        claim_claim_e = []
        claim_premise_e = []
        premise_premise_e = []
        premise_claim_e = []
        for e in self.dataset[index][constants.EDGES]:
            idx_out = e[0]
            idx_in = e[1]
            id_out = self.dataset[index][constants.INDEX_TO_ID][idx_out]
            id_in = self.dataset[index][constants.INDEX_TO_ID][idx_in]

            if self.dataset[index][constants.ID_TO_NODE_TYPE][id_out] == constants.CLAIM:
                if self.dataset[index][constants.ID_TO_NODE_TYPE][id_in] == constants.CLAIM:
                    claim_claim_e.append(self.rearrange_edge_index(claim_list, claim_list, idx_out, idx_in))
                elif self.dataset[index][constants.ID_TO_NODE_TYPE][id_in] == constants.PREMISE:
                    claim_premise_e.append(self.rearrange_edge_index(claim_list, premise_list, idx_out, idx_in))
                else:
                    raise Exception(f'not implemented')
            elif self.dataset[index][constants.ID_TO_NODE_TYPE][id_out] == constants.PREMISE:
                if self.dataset[index][constants.ID_TO_NODE_TYPE][id_in] == constants.CLAIM:
                    premise_claim_e.append(self.rearrange_edge_index(premise_list, claim_list, idx_out, idx_in))
                elif self.dataset[index][constants.ID_TO_NODE_TYPE][id_in] == constants.PREMISE:
                    premise_premise_e.append(self.rearrange_edge_index(premise_list, premise_list, idx_out, idx_in))
                else:
                    raise Exception(f'not implemented')

        # Add 2 claim node and 2 premise node
        two_empty_nodes = torch.concat(
            (torch.zeros_like(stacked_bert_inputs_claim[:, :, 0]).unsqueeze(dim=2),
             torch.zeros_like(stacked_bert_inputs_claim[:, :, 0]).unsqueeze(dim=2)),
            dim=2)
        stacked_bert_inputs_claim = torch.concat((two_empty_nodes, stacked_bert_inputs_claim), dim=2)
        stacked_bert_inputs_premise = torch.concat((two_empty_nodes, stacked_bert_inputs_premise), dim=2)

        data = HeteroData()
        data[constants.CLAIM].x = stacked_bert_inputs_claim.T.long()
        data[constants.CLAIM].y = [self.labels[index]] * data[constants.CLAIM].x.shape[0]
        data[constants.PREMISE].x = stacked_bert_inputs_premise.T.long()
        data[constants.PREMISE].y = [self.labels[index]] * data[constants.PREMISE].x.shape[0]

        data[constants.CLAIM, 'relation', constants.CLAIM].edge_index = self.convert_edge_indexes(claim_claim_e)
        data[constants.CLAIM, 'relation', constants.PREMISE].edge_index = self.convert_edge_indexes(claim_premise_e)
        data[constants.PREMISE, 'relation', constants.CLAIM].edge_index = self.convert_edge_indexes(premise_claim_e)
        data[constants.PREMISE, 'relation', constants.PREMISE].edge_index = self.convert_edge_indexes(premise_premise_e)
        return data


class CMVKGHetroDatasetEdges(CMVKGHetroDataset):
    def __init__(self, directory_path: str,
                 version: str,
                 debug: bool = False):
        super(CMVKGHetroDatasetEdges, self).__init__(directory_path=directory_path, version=version, debug=debug)

    def __getitem__(self, index: int):
        stacked_bert_inputs = self.calc_bert_inputs(self.dataset[index])

        support_e = []
        attack_e = []

        support_types = ['agreement', 'support', 'partial_agreement', 'understand']
        attack_types = ['rebuttal', 'partial_disagreement', 'undercutter', 'disagreement', 'partial_attack',
                        'rebuttal_attack', 'attack', 'undercutter_attack']

        for i, e in enumerate(self.dataset[index][constants.EDGES]):
            if self.dataset[index][constants.EDGES_TYPES][i] in support_types:
                support_e.append(e)
            elif self.dataset[index][constants.EDGES_TYPES][i] in attack_types:
                attack_e.append(e)
            else:
                raise Exception(f'not implemented {self.dataset[index][constants.EDGES_TYPES][i] } ')

        two_empty_nodes = torch.concat((torch.zeros_like(stacked_bert_inputs[:, :, 0]).unsqueeze(dim=2),
                                        torch.zeros_like(stacked_bert_inputs[:, :, 0]).unsqueeze(dim=2)), dim=2)
        stacked_bert_inputs = torch.concat((two_empty_nodes, stacked_bert_inputs), dim=2)



        data = HeteroData()
        data[constants.NODE].x = stacked_bert_inputs.T.long()
        data[constants.NODE].y = [self.labels[index]] * data[constants.NODE].x.shape[0]

        data[constants.NODE, constants.SUPPORT, constants.NODE].edge_index =  self.convert_edge_indexes(support_e)
        data[constants.NODE, constants.ATTACK, constants.NODE].edge_index = self.convert_edge_indexes(attack_e)

        return data


class UKPDataset(torch.utils.data.Dataset):
    """

    """
    def __init__(self,
                 debug: bool = False):
        """

        :param directory_path: The string path to the 'change-my-view-modes-master' directory, which contains versions
            versions of the change my view dataset.
        :param debug: A boolean denoting whether or not we are in debug mode (in which our input dataset is
            significantly smaller).
        """
        self.dataset = []
        self.labels = []

        for file_name in tqdm(os.listdir(constants.UKP_DATA)):
            if file_name.endswith(constants.ANN):
                examples = self.make_op_reply_graphs(file_name=file_name)
                examples = create_bert_inputs([examples],
                                              tokenizer=transformers.BertTokenizer.from_pretrained(
                                                  constants.BERT_BASE_CASED))
                self.dataset.extend(examples)
                example_label = find_label_ukp(file_name)
                self.labels.append(example_label)
                if debug:
                    if len(self.labels) >= 20:
                        break

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        """

        :param index:
        :return:
        """
        bert_input_key_names = [f'id_to_{constants.INPUT_IDS}',
                                f'id_to_{constants.TOKEN_TYPE_IDS}',
                                f'id_to_{constants.ATTENTION_MASK}']
        formatted_bert_inputs = {}
        for input_name in bert_input_key_names:
            formatted_bert_inputs[input_name] = torch.cat(
                [ids.unsqueeze(dim=1) for ids in self.dataset[index][input_name].values()],
                dim=1,
            )
        stacked_bert_inputs = torch.stack([t for t in formatted_bert_inputs.values()], dim=1)
        return Data(x=stacked_bert_inputs.T,
                    edge_index=torch.tensor(self.dataset[index]['edges']).T,
                    y=torch.tensor(self.labels[index]))



    def make_op_reply_graphs(self, file_name) -> GraphExample:
        """

        :param file_name:
        :return:
        """
        # Get OP text
        op_txt = find_op_ukp(file_name)
        d = parse_ann_file(file_name)

        # Get nodes
        id_to_idx = {constants.TITLE: 0}
        id_to_text = {constants.TITLE: op_txt}
        id_to_node_type = {constants.TITLE: constants.CLAIM}

        for i, item in enumerate(d[constants.ENTITIES]):
            node_idx = i + 1
            id_to_idx[item] = node_idx
            id_to_text[item] = d[constants.ENTITIES][item].data
            id_to_node_type[item] = d[constants.ENTITIES][item].type

        idx_to_id = {value: key for key, value in id_to_idx.items()}

        # Get edges
        edges = []
        edges_types = []
        for e in d[constants.ATTRIBUTES]:
            src = id_to_idx[d[constants.ATTRIBUTES][e].entity]
            dest = 0  # Title or OP
            edges.append([src, dest])
            # TODO: talk to Zach if we should add value = For/Against or type = Stance
            edges_types.append(d[constants.ATTRIBUTES][e].value)
        for e in d[constants.RELATIONS]:
            src = id_to_idx[d[constants.RELATIONS][e].source]
            dest = id_to_idx[d[constants.RELATIONS][e].target]
            edges.append([src, dest])
            edges_types.append(d[constants.RELATIONS][e].type)

        # Create results
        result: GraphExample = {
            constants.ID_TO_INDEX: id_to_idx,
            constants.ID_TO_TEXT: id_to_text,
            constants.ID_TO_NODE_TYPE: id_to_node_type,
            constants.ID_TO_NODE_SUB_TYPE: {},  # No subtypes in UKP database
            constants.INDEX_TO_ID: idx_to_id,
            constants.EDGES: edges,
            constants.EDGES_TYPES: edges_types
        }
        return result