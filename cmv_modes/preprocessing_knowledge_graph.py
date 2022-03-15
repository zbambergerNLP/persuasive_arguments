import copy
import os

import torch
import transformers
from bs4 import BeautifulSoup
import typing
import constants
from tqdm import tqdm


#################################################
### Pre-Processing Inputs for Language Models ###
#################################################

# Entry 0 is the title text. Entry 1 is the text content of the original post.
TitlePlusOriginalPost = typing.Tuple[str, str]

# The text content of the original post. This can either be strictly the original post content, or include a prepended,
# and flattened title (associated with the original post) as well.
OriginalPostContent = str

# Entry 0 is an OriginalPost as described above. Entry 1 is the text content of a reply.
OriginalPostPlusReply = typing.Tuple[OriginalPostContent, str]

# A sequence of 2-tuples. Each of the entries in the 2-tuple is a string that is inputted to a language model.
OriginalPostPlusReplySequence = typing.Sequence[OriginalPostPlusReply]

# Entry 0 is the pair of texts fed into a BERT model during training. Entry 1 is the label associated with the pair.
OriginalPostPlusReplyDataset = typing.Tuple[OriginalPostPlusReplySequence, typing.Sequence[int]]


def create_original_post_content(
        bs_data: BeautifulSoup,
        file_name: str = None,
        is_positive: bool = None,
        include_title: bool = True) -> OriginalPostContent:
    """Return the title and contents of the original post of a Reddit CMV thread.

    :param bs_data: A BeautifulSoup instance containing a parsed .xml file. Such an xml file consists of a title and
        original post, as well as replies to that post. The replies to OP are positive if the xml file is in the
        'positive' directory, and negative if they are in the 'negative' directory.
    :param file_name: The name of the .xml file which we've parsed.
    :param is_positive: True if the replies in the file were awarded a "Delta". False otherwise.
    :param include_title: True if the text of the title should be included in OP's component of the BERT input. False
        otherwise.
    :return: title_op_text - a list consisting of two entries: [title text, op text]. title_text is the title of the
        thread started by OP. op_text is the content of OP's post beyond the title.
    """
    # Process the title of the thread.
    title = bs_data.find(constants.TITLE)
    assert len(title.contents) == 1, f"We expect the title to only contain a single claim/premise," \
                                     f" but the title in file {file_name} ({is_positive}) had {len(title.contents)}."
    title_node = title.contents[0]
    title_text = title_node.text

    # Process the content of OP's post (beyond the title).
    op_data = bs_data.find(constants.OP)
    op_prepositions = op_data.find_all([constants.PREMISE, constants.CLAIM])
    op_text = ''
    for preposition in op_prepositions:
        op_text = op_text + " " + preposition.text

    if include_title:
        return title_text + ". " + op_text
    else:
        return op_text


def create_original_post_plus_reply_dataset(
        bs_data: BeautifulSoup,
        file_name: str = None,
        is_positive: bool = None,
        include_title: bool = True) -> OriginalPostPlusReplySequence:
    """Create a collection of textual examples, where each consists of a post and a reply.

    Process xml data into examples for language models. A single example consists of:
    1. An original post. This is a list with two entries. The first entry is the string content of the thread's title.
        The second entry is the remaining string content of the original post.
    2. A reply to the original post. This reply is represented as a single string.

    :param bs_data: A BeautifulSoup instance containing a parsed .xml file.
    :param file_name: The name of the .xml file which we've parsed.
    :param is_positive: True if the replies in the file were awarded a "Delta". False otherwise.
    :param include_title: True if the text of the title should be included in OP's component of the BERT input. False
        otherwise.
    :return: examples - a list of data examples extracted from file_name in the following construction:
        [[title text, op text], reply text]
    """

    original_post_content = OriginalPostContent(
        create_original_post_content(
            bs_data=bs_data,
            file_name=file_name,
            is_positive=is_positive,
            include_title=include_title)
    )
    rep_data = bs_data.find_all(constants.REPLY)

    examples = []
    for reply in rep_data:
        op_copy_content = copy.deepcopy(original_post_content)
        prepositions = reply.find_all([constants.PREMISE, constants.CLAIM])
        reply_text = ""
        for preposition in prepositions:
            reply_text = reply_text + " " + preposition.text
        original_post_plus_reply = tuple([op_copy_content, reply_text])
        examples.append(original_post_plus_reply)
    return examples


def create_simple_bert_inputs(directory_path: str,
                              version: str,
                              debug: bool = False) -> OriginalPostPlusReplyDataset:
    """
    Create input to BERT by taking relevant text from each xml file.

    :param directory_path: The string path to the 'change-my-view-modes-master' directory, which contains versions
            versions of the change my view dataset.
    :param version: A version of the cmv datasets (i.e.. one of 'v2.0', 'v1.0', and 'original') included within the
        'chamge-my-view-modes-master' directory.
    :param debug: A boolean denoting whether or not we are in debug mode (in which our input dataset is
        significantly smaller).
    :return: dataset - a list of data examples in the following construction:
        [[title text, op text], reply text ]
        labels - the label of the data examples 1 for positive and 0 for negative
    """
    dataset = []
    labels = []
    for sign in constants.SIGN_LIST:
        thread_directory = os.path.join(directory_path, version, sign)
        for file_name in tqdm(os.listdir(thread_directory)):
            if file_name.endswith(constants.XML):
                file_path = os.path.join(thread_directory, file_name)
                with open(file_path, 'r') as fileHandle:
                    data = fileHandle.read()
                    bs_data = BeautifulSoup(data, constants.XML)
                    examples = create_original_post_plus_reply_dataset(
                        bs_data=bs_data,
                        file_name=file_name,
                        is_positive=(sign == constants.POSITIVE))
                    dataset.extend(examples)
                    example_labels = list(map(lambda example: 0 if sign == 'negative' else 1, examples))
                    labels.extend(example_labels)
                    if debug:
                        if len(labels) >= 5:
                            break
    return dataset, labels

#######################################################
### Pre-Processing Inputs for Graph Neural Networks ###
#######################################################


IDToTextMapping = typing.Mapping[str, str]
IDToIndexMapping = typing.Mapping[str, int]
IndexToIDMapping = typing.Mapping[int, str]
Edges = typing.Sequence[typing.Tuple[int, int]]


NodeIndexToBERTInputIDs = typing.Mapping[int, torch.Tensor]
NodeIndexToBERTTokenTypeIDs = typing.Mapping[int, torch.Tensor]
NodeIndexToBERTAttentionMask = typing.Mapping[int, torch.Tensor]


class GraphExample(typing.TypedDict):
    id_to_text: IDToTextMapping
    id_to_index: IDToIndexMapping
    idx_to_id: IndexToIDMapping
    edges: Edges
    id_to_input_ids: NodeIndexToBERTInputIDs


GraphDataset = typing.Sequence[GraphExample]


def make_op_subgraph(
        bs_data: BeautifulSoup,
        file_name: str = None,
        is_positive: bool = None) -> typing.Tuple[IDToTextMapping, IDToIndexMapping, IndexToIDMapping, Edges]:
    """Create a collection of mappings that are used to construct `torch_geometric.data.Data` instances for GCN models.

    Given a parsed xml file, generate the following data related to the title and original post:
     1. A mapping from node names to their textual content.
     2. A mapping from node names to their indices in the graph.
     3. A mapping from the index of the node in the graph to its original name in the xml file.
     4. A collection of edges. Edges are 2-tuples where the first entry is the node ID of the origin of the directed
        edge. The second entry is the target node ID of the directed edge.

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
        edges: A list of 2-tuples where each entry corresponds to an individual "edge" (a list with 2 entries).
            The first entry within an edge tuple is the origin, and the second is the target.
    """
    op_data = bs_data.find(constants.OP)
    title = bs_data.find(constants.TITLE)
    assert len(title.contents) == 1, f"We expect the title to only contain a single claim/premise," \
                                     f" but the title in file {file_name} ({is_positive}) had {len(title.contents)}."
    title_node = title.contents[0]

    prepositions = op_data.find_all([constants.PREMISE, constants.CLAIM])
    id_to_idx = {constants.TITLE: 0}
    id_to_text = {constants.TITLE: title.text}
    for i, item in enumerate(prepositions):
        node_idx = i + 1
        id_to_idx[item[constants.NODE_ID]] = node_idx
        id_to_text[item[constants.NODE_ID]] = item.text

    idx_to_id = {value: key for key, value in id_to_idx.items()}

    edges = []
    for node_id, node_idx in id_to_idx.items():
        if node_id == constants.TITLE:
            node = title_node
        else:
            node = op_data.find(id=node_id)
        if constants.REFERENCE in node.attrs:
            references = node[constants.REFERENCE].split('_')
            for edge_destination_id in references:
                edge_destination_id = id_to_idx[edge_destination_id]
                edges.append(tuple([node_idx, edge_destination_id]))

    return id_to_text, id_to_idx, idx_to_id, edges


def make_op_reply_graph(reply_data: BeautifulSoup,
                        id_to_idx: IDToIndexMapping,
                        id_to_text: IDToTextMapping,
                        idx_to_id: IndexToIDMapping,
                        edges: Edges) -> GraphExample:
    """Given a graph subset (containing the title and the original post), create an expanded graph that includes a
    reply's nodes and edges as well.

    :param reply_data: reply object form a BeautifulSoup instance containing a parsed .xml file.
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
    reply_prepositions = reply_data.find_all([constants.PREMISE, constants.CLAIM])
    op_num_of_nodes = len(idx_to_id.keys())

    # Construct node mappings.
    for i, item in enumerate(reply_prepositions):
        node_idx = i + op_num_of_nodes
        id_to_idx[item[constants.NODE_ID]] = node_idx
        id_to_text[item[constants.NODE_ID]] = item.text

    idx_to_id = {value: key for key, value in id_to_idx.items()}

    # Construct edges.
    for node_id, node_idx in id_to_idx.items():
        if node_id == constants.TITLE or node_idx < op_num_of_nodes:
            pass
        else:
            node = reply_data.find(id=node_id)
            if constants.REFERENCE in node.attrs:
                refs = node[constants.REFERENCE].split('_')
                for ref in refs:
                    if ref in id_to_idx.keys():  # Ignore edges from a different reply.
                        ref_idx = id_to_idx[ref]
                        edges.append([node_idx, ref_idx])
    result: GraphExample = {
        constants.ID_TO_INDEX: id_to_idx,
        constants.ID_TO_TEXT: id_to_text,
        constants.INDEX_TO_ID: idx_to_id,
        constants.EDGES: edges,
    }
    return result


def make_op_reply_graphs(bs_data: BeautifulSoup,
                         file_name: str = None,
                         is_positive: bool = None) -> GraphDataset:
    """Generate a dataset where each example is a knowledge graph representing an argument.

    Each example of the knowledge graph consists of nodes and edges corresponding to the original post and some reply.
    Each example is accompanied by a binary label indicating whether or not the argument represented by the graph was
    awarded a "delta" (considered by the author of the original post as a persuasive counter-opinion to the original
    post).

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

    # We create a graph for each reply. This graph consists of nodes and edges from both the reply and the original
    # post.
    examples = []
    replies_data = bs_data.find_all(constants.REPLY)
    for reply_index, reply in enumerate(replies_data):
        id_to_idx = copy.deepcopy(orig_id_to_idx)
        id_to_text = copy.deepcopy(orig_id_to_text)
        idx_to_id = copy.deepcopy(orig_idx_to_id)
        edges = copy.deepcopy(orig_edges)
        reply_examples = make_op_reply_graph(
            reply_data=reply,
            idx_to_id=idx_to_id,
            id_to_idx=id_to_idx,
            id_to_text=id_to_text,
            edges=edges)
        examples.append(reply_examples)
    return examples


def create_bert_inputs(
        graph_dataset: GraphDataset,
        tokenizer: transformers.PreTrainedTokenizer) -> GraphDataset:
    """Add language model inputs to nodes within a graph dataset.

    :param graph_dataset: A mapping from graph features to their values.
    :param tokenizer: A tokenizer instance used to create inputs for language models.
    :return: A modified dataset which includes entries for language model inputs.
    """
    for graph_id, graph in enumerate(graph_dataset):
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

        # Add BERT inputs to the dataset (corresponding to each node).
        # TODO: Consider separating BERT inputs to a separate output.
        for bert_input_type, bert_input_value in graph_lm_inputs.items():
            graph_dataset[graph_id][f'id_to_{bert_input_type}'] = {
                node_id: bert_input_value[node_id].float() for node_id in range(bert_input_value.shape[0])
            }
    return graph_dataset


if __name__ == '__main__':
    claims_lst = []
    premises_lst = []
    label_lst = []
    database = []
    current_path = os.getcwd()
    dataset, labels = create_simple_bert_inputs(
        current_path,
        version=constants.v2_path, debug=False)
