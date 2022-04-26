import networkx as nx
import os
import sys
import pickle
# sys.path.insert(0, '../Main')
# import feature_extraction
# semanticFeaturesSetup = feature_extraction.SemanticFeaturesSetUp()


#################
### Constants ###
#################

# Dictionary Keys
NAME = "name"
ENTITIES = "entities"
ATTRIBUTES = "attributes"
RELATIONS = "relations"

# Parser related constants
T = "T"
A = "A"
R = "R"

# Others

OP = "OP"
STANCE = "Stance"
SUPPORTS = "supports"
ATTACKS = "attacks"
FOR = "For"
AGAINST = "Against"
OBJECT = "obj"


###############
### Classes ###
###############

class Entity:
    ID = 1
    TYPES = ["MajorClaim", "Claim", "Premise", "OP"]

    def __init__(self, start_ind, end_ind, entity_type, data, id=None):
        """

        :param start_ind: starting index of this entity in the essay. A character index. An int.
        :param end_ind: ending index of this entity in the essay. A character index. An int.
        :param entity_type: one of MajorClaim, Claim, or Premise. A string.
        :param data: the string representing the data of this entity.
        :param id: if ID is provided, assign it to self.id, otherwise assign internally tracked ID.
        """

        assert entity_type in Entity.TYPES, "invalid entity type, " + entity_type

        # Assign attribute values
        self.start_ind = start_ind
        self.end_ind = end_ind
        self.type = entity_type

        # Assign ID
        if id:
            self.id = id
        else:
            self.id = T + str(Entity.ID)
        self.data = data
        Entity.ID += 1

    def __str__(self):
        """

        :return: a string representation of this entity object.
        """
        attributes = (str(self.id),
                      str(self.type),
                      str(self.start_ind),
                      str(self.end_ind),
                      str(self.data))
        return "(" + ",".join(attributes) + ")"


class Attribute:
    ID = 1
    TYPES = [STANCE]
    VALUE_TYPES = [FOR, AGAINST]

    def __init__(self, entity, value, attribute_type, id=None):
        """

        :param entity: the entity towards which the OP has an opinion
        :param attribute_type: the type of the attribute
        :param value: either "for" or "against"
        :param id: if ID is provided, assign it to self.id, otherwise assign internally tracked ID.
        """
        assert value in Attribute.VALUE_TYPES, "invalid value type for this stance, " + value
        assert attribute_type in Attribute.TYPES, "invalid attribute type, " + attribute_type

        self.entity = entity
        self.type = attribute_type
        self.value = value
        if id:
            self.id = id
        else:
            self.id = A + str(Attribute.ID)
        Attribute.ID += 1

    def __str__(self):
        """

        :return: a string representation of this entity object.
        """
        attributes = (str(self.id),
                      str(self.type),
                      str(self.entity),
                      str(self.value))
        return "(" + ",".join(attributes) + ")"


class Relations:
    ID = 1
    TYPES = [SUPPORTS, ATTACKS]

    def __init__(self, source, target, relation_type, id=None):
        """

        :param source: the source of the relation
        :param target: the target of the relation
        :param relation_type: the nature of the relation. Either supports or attacks
        :param id: if ID is provided, assign it to self.id, otherwise assign internally tracked ID.
        """
        assert relation_type in Relations.TYPES, "invalid relation type, " + relation_type

        self.source = source
        self.target = target
        self.type = relation_type
        if id:
            self.id = id
        else:
            self.id = R + str(Relations.ID)
        Relations.ID += 1

    def __str__(self):
        """

        :return: a string representation of this entity object.
        """
        attributes = (str(self.id),
                      str(self.type),
                      str(self.source),
                      str(self.target))
        return "(" + ",".join(attributes) + ")"



######################
### Main Functions ###
######################


def get_data():
    """

    :return: A list of annotated essays. Each list entry consists of a dictionary d with
    the following structure:

                    { NAME : <str name>,
                      ENTITIES: {
                            <str entity_id> : <entity object>
                            ...
                            }
                      ATTRIBUTES : {
                            <str attribute_id> : <attribute object>
                            ...
                            }
                      RELATIONS : {
                            <str relation_id> : <relation object>
                            ...
                            }
                    }
    """

    # file navigation related operations
    # cwd = os.getcwd()
    # os.chdir("..")
    # os.chdir("UKP")
    # os.chdir("brat-project-final")
    files = os.listdir(os.getcwd())

    # Data collection:

    result = []
    for f in files:

        if f.endswith(".ann"):

            with open(f, "r") as fileHandle:

                d = {
                    NAME: f,
                    ENTITIES: {},
                    ATTRIBUTES: {},
                    RELATIONS: {}
                }

                lines = fileHandle.readlines()
                for line in lines:
                    if line[0] == T:
                        process_entity(line, d)
                    elif line[0] == A:
                        process_attribute(line, d)
                    elif line[0] == R:
                        process_relation(line, d)
                    else:
                        raise Exception("Unknown node/edge type encountered." +
                                        "See line which caused error below:\n" + line)

                result.append(d)

    # # return to previous cwd
    # os.chdir(cwd)

    return result


########################
### Helper Functions ###
########################

def process_entity(line, d):
    """

    :param line: a string representing an entity type line in a .ann file.
    :param d: the dictionary to which we are adding a new entity.

    :return: Nothing.

    Invariant:

    We updated dictionary d such that it is now  d'.
    In d', the inputted line for this function is represented by an entity object,
    and there exists a mapping in d' from this entity's ID to the entity object itself.
    """
    line.strip()
    line = line.strip().replace("\t", " ")
    words = line.split(" ")
    # print("processing entity:")
    # print(words)
    entity_id = words[0]
    entity_type = words[1]
    start_ind = words[2]
    end_ind = words[3]
    data = " ".join(words[4:])
    entity = Entity(id=entity_id,
                    start_ind=start_ind,
                    end_ind=end_ind,
                    entity_type=entity_type,
                    data=data)
    d[ENTITIES][entity.id] = entity
    # print("\n")


def process_attribute(line, d):
    """

    :param line: a string representing an attribute type line in a .ann file.
    :param d: the dictionary to which we are adding a new attribute.

    :return: Nothing.

    Invariant:

    We updated dictionary d such that it is now  d'.
    In d', the inputted line for this function is represented by an attribute object,
    and there exists a mapping in d' from this attribute's ID to the attribute object itself.
    """

    line = line.strip().replace("\t", " ")
    words = line.split(" ")
    # print("processing attribute:")
    # print(words)
    attribute_id = words[0]
    attribute_type = words[1]
    attribute_entity = words[2]
    attribute_value = words[3]
    attribute = Attribute(entity=attribute_entity,
                          value=attribute_value,
                          attribute_type=attribute_type,
                          id=attribute_id)
    d[ATTRIBUTES][attribute.id] = attribute
    # print("\n")


def process_relation(line, d):
    """

    :param line: a string representing a relation type line in a .ann file.
    :param d: the dictionary to which we are adding a new relation.

    :return: Nothing.

    Invariant:

    We updated dictionary d such that it is now  d'.
    In d', the inputted line for this function is represented by a relation object,
    and there exists a mapping in d' from this relation's ID to the relation object itself.
    """

    line = line.strip().replace("\t", " ")
    words = line.split(" ")
    # print("processing relation:")
    # print(words)
    id = words[0]
    type = words[1]
    source = words[2].split(":")[1]
    target = words[3].split(":")[1]
    relation = Relations(source=source,
                         target=target,
                         relation_type=type,
                         id=id)
    d[RELATIONS][relation.id] = relation
    # print("\n")


def convert_to_graph(d):
    """

    :param d: a dictionary with the following structure:

        { NAME : <str name>,
                      ENTITIES: {
                            <str entity_id> : <entity object>
                            ...
                            }
                      ATTRIBUTES : {
                            <str attribute_id> : <attribute object>
                            ...
                            }
                      RELATIONS : {
                            <str relation_id> : <relation object>
                            ...
                            }
                    }

    :return: A 3-tuple which consists of the following:
                index           content
                0           a nx.Graph object representing the argument structure.
                1           the node mapping. i.e.,
                                    node index -> node object
                2           the edge mapping, i.e.,
                                    (source node index, target node index) -> edge object
                3           the reverse node mapping, i.e., a mapping from
                                    entity ID -> node index
                4           the reverse edge mapping, i.e., a mapping from
                                    relation ID -> (souce index, target index)
    """

    node_id = 1

    # A mapping from the node/edge index in the graph to the entity/relation object it represents.
    node_mapping = {1: Entity(start_ind=-1,
                              end_ind=-1,
                              entity_type=OP,
                              data="<OP>")
                    }
    edge_mapping = {}

    # A mapping from the ID of a node/edge to its index in the graph
    node_rev_mapping = {OP: 1}
    edge_rev_mapping = {}

    g = nx.DiGraph()
    g.add_node(node_id, obj=node_mapping[node_id])
    node_id += 1

    entities = d[ENTITIES]
    attributes = d[ATTRIBUTES]
    relations = d[RELATIONS]

    for e_id, e_obj in entities.items():
        # features = feature_extraction.SemanticFeatures(e_obj.data, semanticFeaturesSetup)
        features = e_obj.data
        node_mapping[node_id] = e_obj
        node_rev_mapping[e_id] = node_id
        g.add_node(node_id, obj=e_obj, val=features)
        node_id += 1

    for _, a_obj in attributes.items():
        # print("Making an edge from an attribute")
        entity = a_obj.entity
        type = a_obj.type
        value = a_obj.value
        if type == STANCE:
            if value == FOR:
                relation_type = SUPPORTS
            else:
                relation_type = ATTACKS
            new_relation = Relations(source=OP,
                                     target=entity,
                                     relation_type=relation_type)

            source = node_rev_mapping[OP]
            target = node_rev_mapping[entity]
            edge = (source, target)
            edge_mapping[edge] = new_relation
            edge_rev_mapping[new_relation.id] = edge  # an edge from OP to the node of that entity
            g.add_edge(source, target, obj=a_obj)
        else:
            raise RuntimeError("Issue with following attribute: " + str(a_obj))

    for r_id, r_obj in relations.items():
        # print("Making an edge from a relation")
        source = node_rev_mapping[r_obj.source]
        target = node_rev_mapping[r_obj.target]
        edge = (source, target)
        edge_mapping[edge] = r_obj
        edge_rev_mapping[r_id] = (source, target)
        g.add_edge(source, target, obj=r_obj)

    return g, node_mapping, edge_mapping, node_rev_mapping, edge_rev_mapping


if __name__ == "__main__":
    annotated_essay_dicts = get_data()
    start_idx = 270
    for i in range(start_idx, len(annotated_essay_dicts)):
        annotated_essay = annotated_essay_dicts[i]
        name = annotated_essay[NAME]
        print("Converting:  " + name + " to graph")
        print("Currently on " + str(i))
        entities = annotated_essay[ENTITIES]
        attributes = annotated_essay[ATTRIBUTES]
        relations = annotated_essay[RELATIONS]

        g, node_mapping, edge_mapping, node_rev_mapping, edge_rev_mapping = convert_to_graph(annotated_essay)
        print("WRITING TO FILE")
        nx.write_gpickle(g, '../UKP/graphs/' + name + '-gpickle')
        with open( '../UKP/graphs/' + name + '-mappings', 'wb') as f:
            pickle.dump((node_mapping, edge_mapping, node_rev_mapping, edge_rev_mapping), f)
        print("FINISHED WRITING TO FILE")

        # print("\t", ENTITIES)
        # for key, value in entities.items():
        #     print("\t" * 2, key, ":", value)

        # print("\t", ATTRIBUTES)
        # for key, value in attributes.items():
        #     print("\t" * 2, key, ":", value)

        # print("\t", RELATIONS)
        # for key, value in relations.items():
        #     print("\t" * 2, key, ":", value)

        # print("\nPrinting graph:\n")
        

        # print("\t" * 2, "graph nodes: ")
        # for node in g.nodes.data():
        #     print("\t" * 3, node[1][OBJECT])

        # print("\t" * 2, "graph edges: ")
        # for edge in g.edges.data():
        #     print("\t" * 3, edge[2][OBJECT])

        # print("##### " * 5)
