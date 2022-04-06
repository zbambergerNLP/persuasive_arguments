import os

import torch
import transformers
from bs4 import BeautifulSoup
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.data import HeteroData

import constants
from cmv_modes.preprocessing_knowledge_graph import make_op_reply_graphs, create_bert_inputs


# TODO: Ensure that both the BERT model and its corresponding tokenizer are accessible even without internet connection.

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


class BaselineLoader(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.num_examples = len(labels)

    def __getitem__(self, item):
        return {'features': self.features[item],
                constants.LABEL: self.labels[item]}

    def __len__(self):
        return self.num_examples


class CMVKGDataset(torch.utils.data.Dataset):
    def __init__(self,
                 directory_path: str,
                 version: str,
                 debug: bool = False):
        """

        :param directory_path: The string path to the 'change-my-view-modes-master' directory, which contains versions
            versions of the change my view dataset.
        :param version: A version of the cmv datasets (i.e.. one of 'v2.0', 'v1.0', and 'original') included within the
            'chamge-my-view-modes-master' directory.
        :param debug: A boolean denoting whether or not we are in debug mode (in which our input dataset is
            significantly smaller).
        """
        self.dataset = []
        self.labels = []
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
                                                      tokenizer=transformers.BertTokenizer.from_pretrained(
                                                          constants.BERT_BASE_CASED))
                        self.dataset.extend(examples)
                        example_labels = list(map(lambda example: 0 if sign == 'negative' else 1, examples))
                        self.labels.extend(example_labels)
                        if debug:
                            if len(self.labels) >= 20:
                                break

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
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


class CMVKGHetroDataset(CMVKGDataset):
    def __init__(self, directory_path: str,
                 version: str,
                 debug: bool = False):
        super(CMVKGHetroDataset, self).__init__(directory_path=directory_path, version=version, debug=debug)

    def calc_bert_inputs(self, dataset_values, relevant_ids):
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

    def rearrange_edge_index(self, out_list: list, in_list: list, out_idx: int, in_idx: int):
        new_idx_out_node = out_list.index(out_idx)
        new_idx_in_node = in_list.index(in_idx)
        return [new_idx_out_node, new_idx_in_node]

    def convert_edge_indexes(self, edge_list):
        #update edges after a claim node and a premise node were added
        #add an edge from every added node to the other so each graph will have all kinds of edges
        edge_list= torch.tensor(edge_list, dtype=torch.long).T + 2
        edge_to_add = torch.tensor([0,1]).unsqueeze(dim =1)
        edge_list = torch.concat((edge_to_add,edge_list), dim=1)
        return  edge_list

    def __getitem__(self, index: int):
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

        #add 2 claim node and 2 premise node
        two_empty_nodes = torch.concat((torch.zeros_like(stacked_bert_inputs_claim[:,:,0]).unsqueeze(dim=2), torch.zeros_like(stacked_bert_inputs_claim[:,:,0]).unsqueeze(dim=2)), dim =2)
        stacked_bert_inputs_claim = torch.concat((two_empty_nodes ,stacked_bert_inputs_claim), dim= 2 )
        stacked_bert_inputs_premise = torch.concat((two_empty_nodes ,stacked_bert_inputs_premise), dim= 2 )

        data = HeteroData()
        # data.has_self_loops()
        data[constants.CLAIM].x = stacked_bert_inputs_claim.T.long()
        data[constants.CLAIM].y = [self.labels[index]] * data[constants.CLAIM].x.shape[0]
        data[constants.PREMISE].x = stacked_bert_inputs_premise.T.long()
        data[constants.PREMISE].y = [self.labels[index]] * data[constants.PREMISE].x.shape[0]

        data[constants.CLAIM, 'relation', constants.CLAIM].edge_index = self.convert_edge_indexes(claim_claim_e)
        data[constants.CLAIM, 'relation', constants.PREMISE].edge_index = self.convert_edge_indexes(claim_premise_e)
        data[constants.PREMISE, 'relation', constants.CLAIM].edge_index = self.convert_edge_indexes(premise_claim_e)
        data[constants.PREMISE, 'relation', constants.PREMISE].edge_index = self.convert_edge_indexes(premise_premise_e)

        return data
