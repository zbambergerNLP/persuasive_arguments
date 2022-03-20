import os

import torch
import transformers
from bs4 import BeautifulSoup
from tqdm import tqdm
from torch_geometric.data import Data

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
                            if len(self.labels) >= 5:
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