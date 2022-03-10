import copy
import os

import torch
import transformers
# import transformers.tokenization_bert
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertModel
import typing
import typing
from torch.utils.data import Dataset
from torch_geometric.data import Data
import constants
from tqdm import tqdm



cmv_modes_versions = [constants.v1_path]
cmv_modes_with_claims_versions = [constants.v2_path]


sign_lst = [constants.POSITIVE, constants.NEGATIVE]


def create_bert_inputs(dataset: (
        typing.Sequence[typing.Mapping[str, typing.Mapping[typing.Union[int, str], typing.Union[int, str]]]]),
                       tokenizer: transformers.PreTrainedTokenizer) -> (
        typing.Sequence[typing.Mapping[
            str,
            typing.Mapping[
                typing.Union[int, str],
                typing.Union[int, str, typing.Mapping[
                    str,
                    torch.Tensor]]]]]):
    """Add language model inputs to nodes within a graph dataset.

    :param dataset: A mapping from graph features to their values.
    :param tokenizer: A tokenizer instance used to create inputs for language models.
    :return: A modified dataset which includes entries for language model inputs.
    """
    for graph_id, graph in enumerate(dataset):
        graph_texts = []
        graph_indices = []
        for node_id, node_text in graph['id_to_text'].items():
            graph_texts.append(node_text)
            graph_indices.append(node_id)
        graph_lm_inputs = tokenizer(
            graph_texts,
            return_tensors="pt",
            padding='max_length',
            max_length=constants.NODE_DIM,
            truncation=True)
        # model = BertModel.from_pretrained("bert-base-uncased")
        # model_outputs = model(**graph_lm_inputs)
        # last_hidden_states = model_outputs.last_hidden_state[:, 0, :]
        # dataset[graph_id]['id_to_embedding'] = {
        #     node_id: node_embedding for node_id, node_embedding in zip(graph_indices, last_hidden_states)}
        for bert_input_type, bert_input_value in graph_lm_inputs.items():
            dataset[graph_id][f'id_to_{bert_input_type}'] = {
                node_id: bert_input_value[node_id].float() for node_id in range(bert_input_value.shape[0])
            }
    return dataset


class CMVKGDataLoader(Dataset):

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
        for sign in sign_lst:
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
        print(f'getitem index = {index}')
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


def make_op_subgraph(bs_data: BeautifulSoup,
                     file_name: str = None,
                     is_positive: bool = None) -> (
        typing.Tuple[typing.Mapping[str, str],
                     typing.Mapping[str, int],
                     typing.Mapping[int, str],
                     typing.Sequence[typing.Sequence[int]]]):
    """

    :param bs_data: A BeautifulSoup instance containing a parsed .xml file.
    :param file_name: The name of the .xml file which we've parsed.
    :param is_positive: True if the replies in the file were awarded a "Delta". False otherwise.
    :return: A tuple consisting of the following:
        id_to_text: A mapping from string node IDs (where nodes are either claims or premises) to the text included
            within the node. This text later encoded into a fixed dimensional representation when creating the knowledge
            graph.
        id_to_idx: A mapping from string node IDs (where nodes are either claims or premises) to node indices in the
            resulting knowledge graph.
        idx_to_id: A mapping from node indices within the knowledge graph to the ID of the node within the corresponding
            .xml file.
        edges: A 2-d List where each entry corresponds to an individual "edge" (a list with 2 entries). The first
            entry within an edge list is the origin, and the second is the target.
    """
    op_data = bs_data.find(constants.OP)
    title = bs_data.find(constants.TITLE)
    assert len(title.contents) == 1, f"We expect the title to only contain a single claim/premise," \
                                     f" but the title in file {file_name} ({is_positive}) had {len(title.contents)}."
    title_node = title.contents[0]
    edges = []

    res = op_data.find_all(['premise', 'claim'])
    id_to_idx = {constants.TITLE: 0}
    id_to_text = {constants.TITLE: title.text}
    for i, item in enumerate(res):
        node_idx = i + 1
        id_to_idx[item[constants.NODE_ID]] = node_idx
        id_to_text[item[constants.NODE_ID]] = item.text
    idx_to_id = {value: key for key, value in id_to_idx.items()}
    for node_id, node_idx in id_to_idx.items():
        if node_id == constants.TITLE:
            node = title_node
        else:
            node = op_data.find(id=node_id)
        if 'ref' in node.attrs:
            refs = node['ref'].split('_')
            for ref in refs:
                ref_idx = id_to_idx[ref]
                edges.append([node_idx, ref_idx])
    return id_to_text, id_to_idx, idx_to_id, edges


def make_op_reply_graph(rep: BeautifulSoup,
                        id_to_idx: typing.Mapping[str, int],
                        id_to_text: typing.Mapping[str, str],
                        idx_to_id: typing.Mapping[int, str],
                        edges: typing.Sequence[typing.Sequence[int]]) -> (
        typing.Mapping[str, typing.Mapping[typing.Union[int, str], typing.Union[int, str]]]):
    """
        :param rep: reply object form a BeautifulSoup instance containing a parsed .xml file.
        :param id_to_idx: A mapping from string node IDs from OP (where nodes are either claims or premises) to node
            indices in the resulting knowledge graph.
        :param id_to_text A dictionary mapping from the node ID to its textual content.
        :param idx_to_id: A mapping from node indices within the OP knowledge graph to the ID of the node within
            the corresponding .xml file.
        :param edges: OP edges, 2-d List where each entry corresponds to an individual "edge" (a list with 2 entries).
            The first entry within an edge list is the origin, and the second is the target.
        :return: A dictionary consisting of the following (key, value) pairs (where keys are strings):
            id_to_idx: OP and reply node mapping from string node IDs (where nodes are either claims or premises) to
                node indices in the resulting knowledge graph.
            id_to_text: Mapping from the ID of a particular node to the text contained within it.
            idx_to_id: OP and reply node mapping from node indices within the knowledge graph to the ID of the node
                within the corresponding .xml file.
            edges: OP and reply edges, a 2-D List where each entry corresponds to an individual "edge" (a list with 2
                entries). The first entry within an edge list is the origin, and the second is the target.
        """
    res = rep.find_all(['premise', 'claim'])
    op_num_of_nodes = len(idx_to_id.keys())

    # Construct node mappings.
    for i, item in enumerate(res):
        node_idx = i + op_num_of_nodes
        id_to_idx[item[constants.NODE_ID]] = node_idx
        id_to_text[item[constants.NODE_ID]] = item.text

    idx_to_id = {value: key for key, value in id_to_idx.items()}

    # Construct edges.
    for node_id, node_idx in id_to_idx.items():
        if node_id == constants.TITLE or node_idx < op_num_of_nodes:
            pass
        else:
            node = rep.find(id=node_id)
            if 'ref' in node.attrs:
                refs = node['ref'].split('_')
                for ref in refs:
                    if ref in id_to_idx.keys():  # Ignore edges from a different reply.
                        ref_idx = id_to_idx[ref]
                        edges.append([node_idx, ref_idx])
    return {
        'id_to_idx': id_to_idx,
        'id_to_text': id_to_text,
        'idx_to_id': idx_to_id,
        'edges': edges,
    }


def make_op_reply_graphs(bs_data: BeautifulSoup,
                         file_name: str = None,
                         is_positive: bool = None) -> (
        typing.Sequence[typing.Mapping[str, typing.Mapping[typing.Union[int, str], typing.Union[int, str]]]]):
    """
        :param bs_data: A BeautifulSoup instance containing a parsed .xml file.
        :param file_name: The name of the .xml file which we've parsed.
        :param is_positive: True if the replies in the file were awarded a "Delta". False otherwise.
        :return: A list of all examples extracted from a file. Each example is a dictionary consisting of the
            following keys:
            id_to_idx: A mapping from string node IDs (where nodes are either claims or premises) to node indices in the
                resulting knowledge graph.
            idx_to_id: A mapping from node indices within the knowledge graph to the ID of the node within the corresponding
                .xml file.
            edges: A 2-d List where each entry corresponds to an individual "edge" (a list with 2 entries). The first
                entry within an edge list is the origin, and the second is the target.
        """
    orig_id_to_text, orig_id_to_idx, orig_idx_to_id, orig_edges = make_op_subgraph(
        bs_data=bs_data,
        file_name=file_name,
        is_positive=is_positive)
    rep_data = bs_data.find_all("reply")

    examples = []
    for reply_index, reply in enumerate(rep_data):
        id_to_idx = copy.deepcopy(orig_id_to_idx)
        id_to_text = copy.deepcopy(orig_id_to_text)
        idx_to_id = copy.deepcopy(orig_idx_to_id)
        edges = copy.deepcopy(orig_edges)
        reply_examples = make_op_reply_graph(
            rep=reply, idx_to_id=idx_to_id, id_to_idx=id_to_idx, id_to_text=id_to_text, edges=edges)
        examples.append(reply_examples)
    return examples




if __name__ == '__main__':
    claims_lst = []
    premises_lst = []
    label_lst = []
    database = []
    current_path = os.getcwd()
    kg_dataset = CMVKGDataLoader(current_path, version=constants.v2_path)
    print("Done")
