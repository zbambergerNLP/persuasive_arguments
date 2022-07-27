import copy
import os
from typing import Tuple, Dict, Union, Any, List, Sequence
import sys
sys.path.append('/home/b.noam/persuasive_arguments')
import torch
import transformers
from bs4 import BeautifulSoup
import typing
import constants
from tqdm import tqdm
import pandas as pd

from UKP.parser import process_entity, process_attribute, process_relation

#################################################
### UKP helper functions ###
#################################################

def find_label_ukp(file_name: str,
                   num_labels: int = 2) -> int:
    """

    :param file_name: The name of the file containing the labels for the UKP dataset.
    :param num_labels: An integer. Either 2 or 3, representing the number of possible labels. If two labels are
        selected, they correspond to whether the (binary) decision argument was voted as persuasive. If three labels
        are selected, they correspond to whether the reader's opinion changed negatively/didn't change/changed
        positively.
    :return: The label corresponding to the provided example (context + argument).
    """
    file_name = file_name.split(".")[0]
    df = pd.read_csv(constants.UKP_LABELS_FILE)
    deltas_pos = df.loc[(df[constants.ID_CSV] == file_name) & (df[constants.DELTA_CSV] > 0), constants.DELTA_CSV]
    deltas_neg = df.loc[(df[constants.ID_CSV] == file_name) & (df[constants.DELTA_CSV] <= 0), constants.DELTA_CSV]
    label = 1 if len(deltas_pos) > len(deltas_neg) else 0
    return label



def find_op_ukp(file_name):
    """

    :param file_name:
    :return:
    """
    file_name = file_name.split(".")[0]
    file_name = file_name + "." + constants.TXT
    with open(os.path.join(constants.UKP_DATA, file_name), "r") as fileHandle:
        lines = fileHandle.readlines()
        return lines[0]



def parse_ann_file(file_name):
    """

    :param file_name:
    :return:
    """
    with open(os.path.join(constants.UKP_DATA, file_name), "r") as fileHandle:
        d = {
            constants.NAME: file_name,
            constants.ENTITIES: {},
            constants.ATTRIBUTES: {},
            constants.RELATIONS: {}
        }

        lines = fileHandle.readlines()
        for line in lines:
            if line[0] == constants.T:
                process_entity(line, d)
            elif line[0] == constants.A:
                process_attribute(line, d)
            elif line[0] == constants.R:
                process_relation(line, d)
            else:
                raise Exception("Unknown node/edge type encountered." +
                                "See line which caused error below:\n" + line + " " + file_name)
    return d

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

# A sequence of utterances (usually sentences) consisting of the title, OP's utterances, and the utterances in the
# argumentative reply.
SentenceLevelDataset = typing.Tuple[typing.Sequence[OriginalPostContent], typing.Sequence[int]]


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
        # TODO: Do prepositions come in order?
        prepositions = reply.find_all([constants.PREMISE, constants.CLAIM])
        reply_text = ""
        for preposition in prepositions:
            reply_text = reply_text + " " + preposition.text
        original_post_plus_reply = tuple([op_copy_content, reply_text])
        examples.append(original_post_plus_reply)
    return examples


def create_simple_bert_inputs(directory_path: str,
                              version: str,
                              debug: bool = False,
                              sentence_level: str = False) -> (
        typing.Union[SentenceLevelDataset, OriginalPostPlusReplyDataset]):
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
                    if sentence_level:
                        new_examples = []
                        for context, reply in examples:
                            context = list(filter(lambda text: len(text) > 0,
                                                  [sentence.strip() for sentence in context.split('.')]))
                            reply = list(filter(lambda text: len(text) > 0,
                                                [sentence.strip() for sentence in reply.split('.')]))
                            new_examples.append(context + reply)
                        examples = new_examples
                    dataset.extend(examples)
                    example_labels = list(map(lambda example: 0 if sign == 'negative' else 1, examples))
                    labels.extend(example_labels)
                    if debug:
                        if len(labels) >= 5:
                            break
    return dataset, labels


def create_original_post_plus_reply_dataset_ukp(
        file_name: str = None) -> OriginalPostPlusReplySequence:
    """Create a collection of textual examples, where each consists of a post and a reply.

    Process ann data into examples for language models. A single example consists of:
    1. An original post.
    2. A reply to the original post. This reply is represented as a single string.

    :param file_name: The name of the .ann file which we've parsed.
    :return: examples - a list of data examples extracted from file_name in the following construction:
        (op text, reply text)
    """
    original_post_content = find_op_ukp(file_name)

    d = parse_ann_file(file_name)
    reply_text = ""
    num_of_words_in_sentence = []
    for i, item in enumerate(d[constants.ENTITIES]):
        reply_text = reply_text + " " + d[constants.ENTITIES][item].data.strip() + "."
        num_of_words_in_sentence.append(len(d[constants.ENTITIES][item].data.split()))
    original_post_plus_reply = tuple([original_post_content, reply_text])
    avg_num_of_words_in_sentence = sum(num_of_words_in_sentence) / len(num_of_words_in_sentence)
    stats = tuple([len(d[constants.ENTITIES]), avg_num_of_words_in_sentence])
    return original_post_plus_reply, stats


def create_simple_bert_inputs_ukp(debug: bool = False,
                                  sentence_level: str = False) -> (
        typing.Union[SentenceLevelDataset, OriginalPostPlusReplyDataset]):
    """
    Create input to BERT by taking relevant text from each ann file.
    :param debug: A boolean denoting whether or not we are in debug mode (in which our input dataset is
        significantly smaller).
    :return: dataset - a list of data examples in one of the following construction:
        1. Paragraph Level: [[[title text, op text], reply text], label]
        2. Sentence Level: [[sentence_1, ..., sentence_m], label]

        sentence_i - the sentence at index i after concatenating reply sentences to context sentences.
        labels - the label of the data examples 1 for positive and 0 for negative
    """
    dataset = []
    labels = []
    for file_name in tqdm(os.listdir(constants.UKP_DATA)):
        if file_name.endswith(constants.ANN):
            (context, reply), stats = create_original_post_plus_reply_dataset_ukp(file_name=file_name)
            example_label = find_label_ukp(file_name)
            if sentence_level:
                print(f'context: {context}')
                print(f'reply: {reply}')
                context = list(filter(lambda text: len(text) > 0,
                                      [sentence.strip() for sentence in context.split('.')]))
                reply = list(filter(lambda text: len(text) > 0,
                                    [sentence.strip() for sentence in reply.split('.')]))
                dataset.append(context + reply)
                labels.append(example_label)
            else:
                dataset.append((context, reply))
                labels.append(example_label)
    return dataset, labels

#######################################################
### Pre-Processing Inputs for Graph Neural Networks ###
#######################################################


IDToTextMapping = typing.Mapping[str, str]
IDToIndexMapping = typing.Mapping[str, int]
IDToNodeTypeMapping = typing.Mapping[str, str]
IndexToIDMapping = typing.Mapping[int, str]
Edges = typing.Sequence[typing.Tuple[int, int]]
EdgesTypes = typing.Sequence[str]

NodeIndexToBERTInputIDs = typing.Mapping[int, torch.Tensor]
NodeIndexToBERTTokenTypeIDs = typing.Mapping[int, torch.Tensor]
NodeIndexToBERTAttentionMask = typing.Mapping[int, torch.Tensor]


class GraphExample(typing.TypedDict):
    id_to_text: IDToTextMapping
    id_to_index: IDToIndexMapping
    id_to_node_type:IDToNodeTypeMapping
    id_to_node_sub_type:IDToNodeTypeMapping
    idx_to_id: IndexToIDMapping
    edges: Edges
    id_to_input_ids: NodeIndexToBERTInputIDs


GraphDataset = typing.Sequence[GraphExample]


# TODO: Modify return type documentation for this function.
def make_op_subgraph(
        bs_data: BeautifulSoup,
        file_name: str = None,
        is_positive: bool = None) -> (
        typing.Tuple[
            typing.Dict[typing.Any, typing.Union[str, typing.Any]],
            typing.Dict[typing.Any, int],
            typing.Dict[typing.Any, typing.Any],
            typing.Dict[typing.Any, typing.Any],
            typing.Dict[typing.Any, typing.Any],
            typing.List[typing.Tuple[int, ...]],
            typing.List[typing.Union[str, typing.List[str]]]]):
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
    id_to_node_type ={constants.TITLE: title_node.name}
    id_to_node_sub_type= {constants.TITLE: title_node[constants.TYPE]}

    for i, item in enumerate(prepositions):
        node_idx = i + 1
        id_to_idx[item[constants.NODE_ID]] = node_idx
        id_to_text[item[constants.NODE_ID]] = item.text
        id_to_node_type[item[constants.NODE_ID]] = item.name
        id_to_node_sub_type[item[constants.NODE_ID]] = item[constants.TYPE]
    idx_to_id = {value: key for key, value in id_to_idx.items()}

    edges = []
    edges_types = []
    for node_id, node_idx in id_to_idx.items():
        if node_id == constants.TITLE:
            node = title_node
        else:
            node = op_data.find(id=node_id)
        if constants.REFERENCE in node.attrs:
            references = node[constants.REFERENCE].split('_')
            rels = node[constants.RELATION]
            for edge_destination_id in references:
                edge_destination_id = id_to_idx[edge_destination_id]
                edges.append(tuple([node_idx, edge_destination_id]))
                edges_types.append(rels)

    return id_to_text, id_to_idx, id_to_node_type, id_to_node_sub_type, idx_to_id, edges, edges_types


def make_op_reply_graph(reply_data: BeautifulSoup,
                        id_to_idx: IDToIndexMapping,
                        id_to_text: IDToTextMapping,
                        id_to_node_type: IDToNodeTypeMapping,
                        id_to_node_sub_type: IDToNodeTypeMapping,
                        idx_to_id: IndexToIDMapping,
                        edges: Edges,
                        edges_types: EdgesTypes) -> GraphExample:
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
        id_to_node_type[item[constants.NODE_ID]] = item.name
        id_to_node_sub_type[item[constants.NODE_ID]] = item[constants.TYPE]
    idx_to_id = {value: key for key, value in id_to_idx.items()}

    # Construct edges.
    for node_id, node_idx in id_to_idx.items():
        if node_id == constants.TITLE or node_idx < op_num_of_nodes:
            pass
        else:
            node = reply_data.find(id=node_id)
            if constants.REFERENCE in node.attrs:
                refs = node[constants.REFERENCE].split('_')
                rels = node[constants.RELATION]
                for i, ref in enumerate(refs):
                    if ref in id_to_idx.keys():  # Ignore edges from a different reply.
                        ref_idx = id_to_idx[ref]
                        edges.append([node_idx, ref_idx])
                        edges_types.append(rels)
    result: GraphExample = {
        constants.ID_TO_INDEX: id_to_idx,
        constants.ID_TO_TEXT: id_to_text,
        constants.ID_TO_NODE_TYPE: id_to_node_type,
        constants.ID_TO_NODE_SUB_TYPE: id_to_node_sub_type,
        constants.INDEX_TO_ID: idx_to_id,
        constants.EDGES: edges,
        constants.EDGES_TYPES: edges_types
    }
    return result

def convert_to_inter_nodes(reply: GraphDataset):
    num_of_nodes = len(reply[constants.ID_TO_INDEX])
    ind = num_of_nodes
    orig_edges = reply[constants.EDGES]
    orig_edges_type = reply[constants.EDGES_TYPES]
    reply[constants.EDGES] = []
    reply[constants.EDGES_TYPES] = []
    for i, edge in enumerate(orig_edges):
        reply[constants.ID_TO_INDEX][str(ind)] = ind
        reply[constants.ID_TO_NODE_SUB_TYPE][str(ind)] = orig_edges_type[i]
        reply[constants.ID_TO_NODE_TYPE][str(ind)] = orig_edges_type[i]
        reply[constants.ID_TO_TEXT][str(ind)] = orig_edges_type[i]
        reply[constants.INDEX_TO_ID][ind] = str(ind)
        reply[constants.EDGES].append([edge[0],ind])
        reply[constants.EDGES].append([ind, edge[1]])
        reply[constants.EDGES_TYPES].append(orig_edges_type[i])
        reply[constants.EDGES_TYPES].append(orig_edges_type[i])
        ind+=1
    return reply



def make_op_reply_graphs(bs_data: BeautifulSoup,
                         file_name: str = None,
                         is_positive: bool = None,
                         inter_nodes: bool = False) -> GraphDataset:
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
    orig_id_to_text, orig_id_to_idx, orig_id_to_node_type, orig_id_to_node_sub_type, orig_idx_to_id, orig_edges, orig_edges_types = make_op_subgraph(
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
        id_to_node_type = copy.deepcopy(orig_id_to_node_type)
        id_to_node_sub_type = copy.deepcopy(orig_id_to_node_sub_type)
        idx_to_id = copy.deepcopy(orig_idx_to_id)
        edges = copy.deepcopy(orig_edges)
        edges_types = copy.deepcopy(orig_edges_types)
        reply_examples = make_op_reply_graph(
            reply_data=reply,
            idx_to_id=idx_to_id,
            id_to_idx=id_to_idx,
            id_to_text=id_to_text,
            id_to_node_type=id_to_node_type,
            id_to_node_sub_type=id_to_node_sub_type,
            edges=edges,
            edges_types=edges_types)
        if inter_nodes:
            reply_examples = convert_to_inter_nodes(reply_examples)
        examples.append(reply_examples)
    return examples


def create_bert_inputs(
        graph_dataset: GraphDataset,
        tokenizer: typing.Union[transformers.PreTrainedTokenizer, transformers.AutoTokenizer]) -> GraphDataset:
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
        for bert_input_type, bert_input_value in graph_lm_inputs.items():
            graph_dataset[graph_id][f'id_to_{bert_input_type}'] = { #TODO make sure node id is what we need, it should not be range but the ids on the idx of the nodes
                node_id: bert_input_value[node_id].float() for node_id in range(bert_input_value.shape[0])
            }
    return graph_dataset


if __name__ == '__main__':
    d, l =create_simple_bert_inputs_ukp(debug=False)
    exit()
    claims_lst = []
    premises_lst = []
    label_lst = []
    database = []
    current_path = os.getcwd()
    dataset, labels = create_simple_bert_inputs(
        current_path,
        version=constants.v2_path, debug=False)
