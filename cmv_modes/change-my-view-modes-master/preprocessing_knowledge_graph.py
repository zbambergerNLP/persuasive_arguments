import os
from bs4 import BeautifulSoup
import typing

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


def make_op_subgraph(bs_data: BeautifulSoup,
                     file_name: str = None,
                     is_positive: bool = None) -> (
        typing.Tuple[typing.Mapping[str, int], typing.Mapping[int, str], typing.Sequence[typing.Sequence[int]]]):
    """

    :param bs_data: A BeautifulSoup instance containing a parsed .xml file.
    :param file_name: The name of the .xml file which we've parsed.
    :param is_positive: True if the replies in the file were awarded a "Delta". False otherwise.
    :return: A tuple consisting of the following:
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
    for i, item in enumerate(res):
        node_idx = i + 1
        id_to_idx[item['id']] = node_idx
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
    return id_to_idx, idx_to_id, edges


if __name__ == '__main__':
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
                    id_to_idx, idx_to_id, edges = make_op_subgraph(
                        bs_data=bs_data,
                        file_name=file_name,
                        is_positive=(sign == POSITIVE))
