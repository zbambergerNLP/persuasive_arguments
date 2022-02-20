import copy
import os
from bs4 import BeautifulSoup
import typing
from torch.utils.data import Dataset


XML = "xml"
PREMISE = "premise"
TYPE = "type"

v1_path = 'v1.0'
v2_path = 'v2.0'

cmv_modes_versions = [v1_path]
cmv_modes_with_claims_versions = [v2_path]

POSITIVE = 'positive'
NEGATIVE = 'negative'
sign_lst = [POSITIVE, NEGATIVE]


class CMVKGDataLoader(Dataset):
    def __init__(self, directory_path, version):
        self.dataset = []
        self.labels = []
        for sign in sign_lst:
            thread_directory = os.path.join(directory_path, version, sign)
            for file_name in os.listdir(thread_directory):
                if file_name.endswith(XML):
                    file_path = os.path.join(thread_directory, file_name)
                    with open(file_path, 'r') as fileHandle:
                        data = fileHandle.read()
                        bs_data = BeautifulSoup(data, XML)
                        examples = make_op_reply_graphs(
                            bs_data=bs_data,
                            file_name=file_name,
                            is_positive=(sign == POSITIVE))
                        self.dataset.extend(examples)
                        example_labels = list(map(lambda example: 0 if sign == 'negative' else 1, examples))
                        self.labels.extend(example_labels)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return {
            'graph': self.dataset[index],
            'label': self.labels[index]
        }

    def embed_nodes(self, node_text):
        pass


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
    op_data = bs_data.find("OP")
    title = bs_data.find('title')
    assert len(title.contents) == 1, f"We expect the title to only contain a single claim/premise," \
                                     f" but the title in file {file_name} ({is_positive}) had {len(title.contents)}."
    title_node = title.contents[0]
    edges = []

    res = op_data.find_all(['premise', 'claim'])
    id_to_idx = {'title': 0}
    id_to_text = {'title': title.text}
    for i, item in enumerate(res):
        node_idx = i + 1
        id_to_idx[item['id']] = node_idx
        id_to_text[item['id']] = item.text
    idx_to_id = {value: key for key, value in id_to_idx.items()}
    for node_id, node_idx in id_to_idx.items():
        if node_id == 'title':
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
    typing.Tuple[typing.Mapping[str, int],
                 typing.Mapping[str, str],
                 typing.Mapping[int, str],
                 typing.Sequence[typing.Sequence[int]]]
):
    """
        :param rep: reply object form a BeautifulSoup instance containing a parsed .xml file.
        :param id_to_idx: A mapping from string node IDs from OP (where nodes are either claims or premises) to node
            indices in the resulting knowledge graph.
        :param id_to_text A dictionary mapping from the node ID to its textual content.
        :param idx_to_id: A mapping from node indices within the OP knowledge graph to the ID of the node within
            the corresponding .xml file.
        :param edges: OP edges, 2-d List where each entry corresponds to an individual "edge" (a list with 2 entries).
            The first entry within an edge list is the origin, and the second is the target.
        :return: A tuple consisting of the following:
            id_to_idx: OP and reply node mapping from string node IDs (where nodes are either claims or premises) to node
                indices in the resulting knowledge graph.
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
        id_to_idx[item['id']] = node_idx
        id_to_text[item['id']] = item.text

    idx_to_id = {value: key for key, value in id_to_idx.items()}

    # Construct edges.
    for node_id, node_idx in id_to_idx.items():
        if node_id == 'title' or node_idx < op_num_of_nodes:
            pass
        else:
            node = rep.find(id=node_id)
            if 'ref' in node.attrs:
                refs = node['ref'].split('_')
                for ref in refs:
                    if ref in id_to_idx.keys():  # Ignore edges from a different reply.
                        ref_idx = id_to_idx[ref]
                        edges.append([node_idx, ref_idx])
    return id_to_idx, id_to_text, idx_to_id, edges


def make_op_reply_graphs(bs_data: BeautifulSoup,
                         file_name: str = None,
                         is_positive: bool = None):
    """
        :param bs_data: A BeautifulSoup instance containing a parsed .xml file.
        :param file_name: The name of the .xml file which we've parsed.
        :param is_positive: True if the replies in the file were awarded a "Delta". False otherwise.
        :return: A list of all examples extracted from file. Each example is a tuple consisting of the following:
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
    for rep in rep_data:
        id_to_idx = copy.deepcopy(orig_id_to_idx)
        idx_to_id = copy.deepcopy(orig_idx_to_id)
        edges = copy.deepcopy(orig_edges)
        example = make_op_reply_graph(
            rep=rep, idx_to_id=idx_to_id, id_to_idx=id_to_idx, id_to_text=orig_id_to_text, edges=edges)
        examples.append(example)
    return examples


if __name__ == '__main__':
    claims_lst = []
    premises_lst = []
    label_lst = []
    database = []
    current_path = os.getcwd()
    kg_dataset = CMVKGDataLoader(current_path, version=v2_path)
    print("Done")
