from bs4 import BeautifulSoup
import constants
import os
import torch
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.data import HeteroData
import torch_geometric.loader as geom_data
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import SubsetRandomSampler
import transformers
import typing

from cmv_modes.preprocessing_knowledge_graph import make_op_reply_graphs
from cmv_modes.preprocessing_knowledge_graph import create_bert_inputs
from cmv_modes.preprocessing_knowledge_graph import GraphExample
from cmv_modes.preprocessing_knowledge_graph import find_op_ukp
from cmv_modes.preprocessing_knowledge_graph import find_label_ukp
from cmv_modes.preprocessing_knowledge_graph import parse_ann_file


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
                 debug: bool = False,
                 super_node: bool = False,
                 iter_nodes: bool = False):
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
        self.super_node = super_node
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
                            is_positive=(sign == constants.POSITIVE),
                            inter_nodes = iter_nodes)
                        examples = create_bert_inputs(examples,
                                                      tokenizer=transformers.AutoTokenizer.from_pretrained(model_name))
                        self.dataset.extend(examples)
                        example_labels = list(map(lambda example: 0 if sign == 'negative' else 1, examples))
                        self.labels.extend(example_labels)
                        if debug:
                            if len(self.labels) >= 20:
                                break
        print('done')
    def stats(self):
        num_of_examples = len(self.dataset)
        print(f"num of positive examples ={sum(self.labels)}")
        print(f"num of negative examples ={num_of_examples-sum(self.labels)}")
        num_of_nodes = []
        num_of_edges = []
        num_of_words = []
        for e in self.dataset:
            num_of_nodes.append(len(e['id_to_idx']))
            num_of_edges.append(len(e['edges']))
            for t in e["id_to_text"].keys():
                num_of_words.append(len(e["id_to_text"][t].split()))

        print(f"avg num of nodes = {sum(num_of_nodes) /num_of_examples}")
        print(f"avg num of edges = {sum(num_of_edges) /num_of_examples}")
        print(f"avg num of words in nodes ={sum(num_of_words)/ sum(num_of_nodes)}")

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
        if self.super_node: #TODO remove edges from intermidate nodes
            #add super node first
            super_node_embedding = torch.empty(stacked_bert_inputs[:, :, 0].shape).unsqueeze(dim=2)
            super_node_embedding.data.uniform_(-constants.initial_range, constants.initial_range).long()
            stacked_bert_inputs = torch.concat((super_node_embedding,stacked_bert_inputs),dim= 2)

            #add edges from all nodes to the super node
            edges = self.dataset[index][constants.EDGES]
            edges = torch.tensor(edges) +1
            new_edges = torch.concat((torch.tensor(
                range(1, len(self.dataset[index][constants.INDEX_TO_ID]) + 1)).unsqueeze(dim=1),
                                      torch.zeros((len(self.dataset[index][constants.INDEX_TO_ID]), 1))), dim=1)

            edges = torch.concat((edges, new_edges))
            return Data(x=stacked_bert_inputs.T,
                        edge_index=torch.tensor(edges, dtype=torch.long).T,
                        y=torch.tensor(self.labels[index]))

            # import matplotlib.pyplot as plt
            # import networkx as nx
            # import torch_geometric
            # g = torch_geometric.utils.to_networkx(d)
            # nx.draw(g, arrows=True, with_labels=True, node_size=400, node_color ='#419fde')
            # plt.show()

        else:
            return Data(x=stacked_bert_inputs.T,
                        edge_index=torch.tensor(self.dataset[index][constants.EDGES]).T,
                        y=torch.tensor(self.labels[index]))


class CMVKGHetroDataset(CMVKGDataset):
    """

    """

    def __init__(self,
                 directory_path: str,
                 model_name: str,
                 version: str,
                 debug: bool = False):
        """

        :param directory_path:
        :param version:
        :param debug:
        """
        super(CMVKGHetroDataset, self).__init__(directory_path=directory_path,
                                                model_name=model_name,
                                                version=version,
                                                debug=debug)

    @staticmethod
    def calc_bert_inputs(dataset_values, model_name, relevant_ids=None):
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


        # SBERT does not consider two inputs in the way BERT does, so the "token type ID" input is not necessary.
        if model_name == "sentence-transformers/all-distilroberta-v1":
            bert_input_key_names.pop(1)

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

    @staticmethod
    def add_super_node_edges(num_of_nodes):
        edge_list = []
        for n in range(num_of_nodes):
            edge_list.append([n,0])
        edge_list = torch.tensor(edge_list, dtype=torch.long).T
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

        # bert_input_key_names = [f'id_to_{constants.INPUT_IDS}',
        #                         f'id_to_{constants.TOKEN_TYPE_IDS}',
        #                         f'id_to_{constants.ATTENTION_MASK}']
        #
        # # SBERT does not consider two inputs in the way BERT does, so the "token type ID" input is not necessary.
        # if self.model_name == "sentence-transformers/all-distilroberta-v1":
        #     bert_input_key_names.pop(1)
        #
        # formatted_bert_inputs = {}
        # for input_name in bert_input_key_names:
        #     formatted_bert_inputs[input_name] = torch.cat(
        #         [ids.unsqueeze(dim=1) for ids in self.dataset[index][input_name].values()],
        #         dim=1,
        #     )
        # stacked_bert_inputs = torch.stack([t for t in formatted_bert_inputs.values()], dim=1)

        stacked_bert_inputs_claim = self.calc_bert_inputs(self.dataset[index],
                                                          model_name=self.model_name,
                                                          relevant_ids=claim_list)
        stacked_bert_inputs_premise = self.calc_bert_inputs(self.dataset[index],
                                                            model_name=self.model_name,
                                                            relevant_ids=premise_list)

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


        data[constants.SUPER_NODE].x = torch.empty(stacked_bert_inputs_claim[:, :, 0].shape).unsqueeze(dim=2).T
        data[constants.SUPER_NODE].x.data.uniform_(-constants.initial_range, constants.initial_range).long()
        data[constants.SUPER_NODE].y = [self.labels[index]] * data[constants.SUPER_NODE].x.shape[0]

        data[constants.CLAIM, constants.RELATION, constants.CLAIM].edge_index = self.convert_edge_indexes(claim_claim_e)
        data[constants.CLAIM, constants.RELATION, constants.PREMISE].edge_index = self.convert_edge_indexes(claim_premise_e)
        data[constants.PREMISE, constants.RELATION, constants.CLAIM].edge_index = self.convert_edge_indexes(premise_claim_e)
        data[constants.PREMISE, constants.RELATION, constants.PREMISE].edge_index = self.convert_edge_indexes(premise_premise_e)
        data[constants.PREMISE, constants.RELATION, constants.SUPER_NODE].edge_index = self.add_super_node_edges(len(premise_list))
        data[constants.CLAIM, constants.RELATION, constants.SUPER_NODE].edge_index = self.add_super_node_edges(len(claim_list))
        return data


class CMVKGHetroDatasetEdges(CMVKGHetroDataset):
    def __init__(self,
                 directory_path: str,
                 version: str,
                 debug: bool = False):
        super(CMVKGHetroDatasetEdges, self).__init__(
            directory_path=directory_path,
            version=version,
            debug=debug)

    def __getitem__(self, index: int):
        stacked_bert_inputs = self.calc_bert_inputs(
            self.dataset[index],
            model_name=self.model_name)

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

        data[constants.NODE, constants.SUPPORT, constants.NODE].edge_index = self.convert_edge_indexes(support_e)
        data[constants.NODE, constants.ATTACK, constants.NODE].edge_index = self.convert_edge_indexes(attack_e)

        return data


class UKPDataset(torch.utils.data.Dataset):
    """

    """
    def __init__(self,
                 model_name: str = constants.BERT_BASE_CASED,
                 debug: bool = False,
                 super_node: bool = False):
        """

        :param directory_path: The string path to the 'change-my-view-modes-master' directory, which contains versions
            versions of the change my view dataset.
        :param debug: A boolean denoting whether or not we are in debug mode (in which our input dataset is
            significantly smaller).
        """
        self.dataset = []
        self.labels = []
        self.model_name = model_name
        self.super_node = super_node
        for file_name in tqdm(os.listdir(constants.UKP_DATA)):
            if file_name.endswith(constants.ANN):
                examples = self.make_op_reply_graphs(file_name=file_name)
                examples = create_bert_inputs([examples],
                                   tokenizer=transformers.AutoTokenizer.from_pretrained(model_name))
                self.dataset.extend(examples)
                example_label = find_label_ukp(file_name)
                self.labels.append(example_label)
                if debug:
                    if len(self.labels) >= 20:
                        break
        print("done")
    def stats(self):
        num_of_examples = len(self.dataset)
        print(f"num of positive examples ={sum(self.labels)}")
        print(f"num of negative examples ={num_of_examples-sum(self.labels)}")
        num_of_nodes = []
        num_of_edges = []
        num_of_words = []
        for e in self.dataset:
            num_of_nodes.append(len(e['id_to_idx']))
            num_of_edges.append(len(e['edges']))
            for t in e["id_to_text"].keys():
                num_of_words.append(len(e["id_to_text"][t].split()))

        print(f"avg num of nodes = {sum(num_of_nodes) /num_of_examples}")
        print(f"avg num of edges = {sum(num_of_edges) /num_of_examples}")
        print(f"avg num of words in nodes ={sum(num_of_words)/ sum(num_of_nodes)}")

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

        if self.super_node:
            # add super node first
            super_node_embedding = torch.empty(stacked_bert_inputs[:, :, 0].shape).unsqueeze(dim=2)
            super_node_embedding.data.uniform_(-constants.initial_range, constants.initial_range).long()
            stacked_bert_inputs = torch.concat((super_node_embedding, stacked_bert_inputs), dim=2)

            # add edges from all nodes to the super node
            edges = self.dataset[index][constants.EDGES]
            edges = torch.tensor(edges) + 1
            new_edges = torch.concat((torch.tensor(
                range(1, len(self.dataset[index][constants.INDEX_TO_ID]) + 1)).unsqueeze(dim=1),
                                      torch.zeros((len(self.dataset[index][constants.INDEX_TO_ID]), 1))), dim=1)

            edges = torch.concat((edges, new_edges))
            return Data(x=stacked_bert_inputs.T,
                        edge_index=torch.tensor(edges, dtype=torch.long).T,
                        y=torch.tensor(self.labels[index]))

        else:
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
        id_to_node_sub_type ={constants.TITLE: constants.CLAIM}


        claim_types = ['Claim', 'MajorClaim']
        premise_types = ['Premise']



        for i, item in enumerate(d[constants.ENTITIES]):
            node_idx = i + 1
            id_to_idx[item] = node_idx
            id_to_text[item] = d[constants.ENTITIES][item].data
            id_to_node_sub_type[item] = d[constants.ENTITIES][item].type
            id_to_node_type[item] =  constants.CLAIM if d[constants.ENTITIES][item].type  in claim_types else constants.PREMISE if d[constants.ENTITIES][item].type  in premise_types else d[constants.ENTITIES][item].type

            if d[constants.ENTITIES][item].type not in claim_types:
                if d[constants.ENTITIES][item].type not in premise_types:
                    raise NotImplementedError(f'{d[constants.ENTITIES][item].type}')
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
            constants.ID_TO_NODE_SUB_TYPE: id_to_node_sub_type,  # No subtypes in UKP database
            constants.INDEX_TO_ID: idx_to_id,
            constants.EDGES: edges,
            constants.EDGES_TYPES: edges_types
        }
        return result


class UKPHetroDataset(UKPDataset):
    def __init__(self,
                 model_name: str = constants.BERT_BASE_CASED,
                 debug: bool = False,
                 super_node: bool = False):
        super(UKPHetroDataset, self).__init__(model_name=model_name, debug=debug, super_node=super_node)

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

        stacked_bert_inputs_claim = CMVKGHetroDataset.calc_bert_inputs(self.dataset[index],model_name=self.model_name,relevant_ids=claim_list)
        stacked_bert_inputs_premise = CMVKGHetroDataset.calc_bert_inputs(self.dataset[index],model_name=self.model_name,relevant_ids=premise_list)

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
                    claim_claim_e.append(CMVKGHetroDataset.rearrange_edge_index(claim_list, claim_list, idx_out, idx_in))
                elif self.dataset[index][constants.ID_TO_NODE_TYPE][id_in] == constants.PREMISE:
                    claim_premise_e.append(CMVKGHetroDataset.rearrange_edge_index(claim_list, premise_list, idx_out, idx_in))
                else:
                    raise Exception(f'not implemented')
            elif self.dataset[index][constants.ID_TO_NODE_TYPE][id_out] == constants.PREMISE:
                if self.dataset[index][constants.ID_TO_NODE_TYPE][id_in] == constants.CLAIM:
                    premise_claim_e.append(CMVKGHetroDataset.rearrange_edge_index(premise_list, claim_list, idx_out, idx_in))
                elif self.dataset[index][constants.ID_TO_NODE_TYPE][id_in] == constants.PREMISE:
                    premise_premise_e.append(CMVKGHetroDataset.rearrange_edge_index(premise_list, premise_list, idx_out, idx_in))
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


        data[constants.SUPER_NODE].x = torch.empty(stacked_bert_inputs_claim[:, :, 0].shape).unsqueeze(dim=2).T
        data[constants.SUPER_NODE].x.data.uniform_(-constants.initial_range, constants.initial_range).long()
        data[constants.SUPER_NODE].y = [self.labels[index]] * data[constants.SUPER_NODE].x.shape[0]

        data[constants.CLAIM, constants.RELATION, constants.CLAIM].edge_index = CMVKGHetroDataset.convert_edge_indexes(claim_claim_e)
        data[constants.CLAIM, constants.RELATION, constants.PREMISE].edge_index = CMVKGHetroDataset.convert_edge_indexes(claim_premise_e)
        data[constants.PREMISE, constants.RELATION, constants.CLAIM].edge_index = CMVKGHetroDataset.convert_edge_indexes(premise_claim_e)
        data[constants.PREMISE, constants.RELATION, constants.PREMISE].edge_index = CMVKGHetroDataset.convert_edge_indexes(premise_premise_e)
        data[constants.PREMISE, constants.RELATION, constants.SUPER_NODE].edge_index = CMVKGHetroDataset.add_super_node_edges(len(premise_list))
        data[constants.CLAIM, constants.RELATION, constants.SUPER_NODE].edge_index = CMVKGHetroDataset.add_super_node_edges(len(claim_list))
        return data


def create_dataloaders_for_k_fold_cross_validation(
        dataset: Dataset,
        dataset_type: str,
        num_of_examples: int,
        shuffled_indices,
        batch_size: int,
        val_percent: float,
        test_percent: float,
        k_fold_index: int,
        num_workers: int = 0,
        index_mapping: typing.Dict[typing.Tuple[int, int], int] = None,
        sentence_level: bool = False) -> (
        typing.Union[
            typing.Tuple[geom_data.DataLoader, geom_data.DataLoader, geom_data.DataLoader],
            typing.Tuple[DataLoader, DataLoader, DataLoader]]):
    """

    :param dataset:
    :param dataset_type:
    :param num_of_examples:
    :param shuffled_indices:
    :param batch_size:
    :param val_percent:
    :param test_percent:
    :param k_fold_index:
    :param num_workers:
    :param index_mapping:
    :param sentence_level:
    :return:
    """
    test_len = int(test_percent * num_of_examples)
    val_len = int(val_percent * num_of_examples)
    held_out_len = test_len + val_len
    held_out_indices = shuffled_indices[k_fold_index * held_out_len:(k_fold_index + 1) * held_out_len]
    train_indices = shuffled_indices[:k_fold_index * held_out_len]
    train_indices.extend(shuffled_indices[(k_fold_index + 1) * held_out_len:])
    val_indices = held_out_indices[:val_len]
    test_indices = held_out_indices[val_len:]
    if dataset_type == "graph":
        dl_train = geom_data.DataLoader(dataset,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        sampler=SubsetRandomSampler(train_indices))
        dl_val = geom_data.DataLoader(dataset,
                                      batch_size=batch_size,
                                      num_workers=num_workers,
                                      sampler=SubsetRandomSampler(val_indices))
        dl_test = geom_data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       num_workers=num_workers,
                                       sampler=SubsetRandomSampler(test_indices))
    elif dataset_type == "language_model":
        if sentence_level:
            dl_train = DataLoader(dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  sampler=SubsetSequenceRandomSampler(train_indices, index_mapping=index_mapping),
                                  collate_fn=lambda x: x)
            dl_val = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                sampler=SubsetSequenceRandomSampler(val_indices, index_mapping=index_mapping),
                                collate_fn=lambda x: x)
            dl_test = DataLoader(dataset,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 sampler=SubsetSequenceRandomSampler(test_indices, index_mapping=index_mapping),
                                 collate_fn=lambda x: x)
        else:
            dl_train = DataLoader(dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  sampler=SubsetRandomSampler(train_indices))
            dl_val = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                sampler=SubsetRandomSampler(val_indices))
            dl_test = DataLoader(dataset,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 sampler=SubsetRandomSampler(test_indices))
    else:
        raise RuntimeError(f"Un-supported dataset type: {dataset_type}")
    return dl_train, dl_val, dl_test


class SubsetSequenceRandomSampler(SubsetRandomSampler):
    def __init__(self,
                 example_indices: typing.List[int],
                 index_mapping: typing.Mapping[int, typing.List[int]],
                 random_generator: torch.Generator = None) -> None:
        # Initialize 'indices' and 'generator' fields for sampler.
        super(SubsetSequenceRandomSampler, self).__init__(indices=example_indices, generator=random_generator)
        self.index_mapping = index_mapping

    def __iter__(self) -> typing.Iterator:
        """
        We override the SubsetRandomSampler's __iter__ function in order to ensure that all sequences corresponding
        to a chosen example are returned.

        :return: Indices of sequences within the flattened dataset which correspond with the sampled example.
        """
        # Map an example index to all sequences associated with that example within the flattened dataset.
        for i in torch.randperm(len(self.indices), generator=self.generator):
            example = self.indices[i]
            example_flattened_indices = self.index_mapping[example]
            yield example_flattened_indices
