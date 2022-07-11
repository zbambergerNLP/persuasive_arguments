import copy
import math
import typing

import numpy as np
import torch
import torch.nn.functional as F
import os
import random

import torch_geometric.data
import torch_geometric.loader as geom_data

import metrics
import utils
import wandb
from data_loaders import create_dataloaders_for_k_fold_cross_validation
from data_loaders import CMVKGDataset, CMVKGHetroDataset, CMVKGHetroDatasetEdges, UKPDataset, UKPHetroDataset
from models import HomophiliousGNN, HeteroGNN
import constants
import argparse
from tqdm import tqdm

"""
Example command:
srun --gres=gpu:1 -p nlp python3 train_and_eval.py \
    --data CMV \
    --num_epochs 30 \
    --batch_size 16 \
    --learning_rate 1e-3 \
    --weight_decay 1e-3 \
    --scheduler_gamma 0.9 \
    --dropout_probability 0.05 \
    --gcn_hidden_layer_dim "128 64 32" \
    --test_percent 0.1 \
    --val_percent 0.1 \
    --rounds_between_evals 1 \
    --model "GAT" \
    --encoder_type "sbert" \
    --debug "false" \
    --hetero "true" \
    --aggregation_type "super_node" \
    --hetero_type "nodes" \
    --use_k_fold_cross_validation "true" \
    --num_cross_validation_splits 5 \
    --seed 42 \
    --dropout_probability 0.3 \
    --positive_example_weight 1\
    --inter_nodes True
"""


parser = argparse.ArgumentParser(
    description='Process flags for experiments on processing graphical representations of arguments through GNNs.')
parser.add_argument('--data',
                    type=str,
                    default='CMV',
                    help="Defines which database to use CMV or UKP")
parser.add_argument('--encoder_type',
                    type=str,
                    default='bert',
                    help="The model used to both tokenize and encode the textual context of argumentative "
                         "prepositions.")
parser.add_argument('--hetero',
                    type=str,
                    default="False",
                    help="Use heterophilous graphs if true and homophilous if False")
parser.add_argument('--hetero_type',
                    type=str,
                    default='nodes',
                    help="Relevant only if herto is True. Possible values are 'nodes' or 'edges'. "
                         "If the value is 'nodes' then node type is used, if the value is 'edges' then edge type is "
                         "used")
parser.add_argument('--num_epochs',
                    type=int,
                    default=30,
                    help="The number of training rounds over the knowledge graph dataset.")
parser.add_argument('--batch_size',
                    type=int,
                    default=16,
                    help="The number of examples per batch per device during both training and evaluation.")
parser.add_argument('--learning_rate',
                    type=float,
                    default=1e-3,
                    help="The learning rate used by the GCN+BERT model during training.")
parser.add_argument('--weight_decay',
                    type=float,
                    default=1e-3,
                    help="The weight decay parameter supplied to the optimizer for use during training.")
parser.add_argument('--gcn_hidden_layer_dim',
                    type=str,
                    default='128 64 32',
                    help="The dimensionality of the hidden layer within the GCN component of the GCN+BERT model.")
parser.add_argument('--test_percent',
                    type=float,
                    default=0.1,
                    help='The proportion (ratio) of samples dedicated to the test set.')
parser.add_argument('--val_percent',
                    type=float,
                    default=0.1,
                    help='The proportion (ratio) of samples dedicated to the validation set.')
parser.add_argument('--rounds_between_evals',
                    type=int,
                    default=1,
                    help="An integer denoting the number of epcohs that occur between each evaluation run.")
parser.add_argument('--debug',
                    type=str,
                    default="True",
                    help="Work in debug mode")
parser.add_argument('--aggregation_type',
                    type=str,
                    default='super_node',
                    help='The name of the aggregation method for GNN classification. Options: super_node, max_pooling, avg_pooling')
parser.add_argument('--model',
                    type=str,
                    default='GAT',
                    help="chose which model to run with the options are: GCN, GAT , SAGE")
parser.add_argument('--use_k_fold_cross_validation',
                    type=str,
                    default="False",
                    help="True if we intend to perform cross validation on the dataset. False otherwise. Using this"
                         "option is advised if the dataset is small.")
parser.add_argument('--num_cross_validation_splits',
                    type=int,
                    default=5,
                    help="The number of cross validation splits we perform as part of k-fold cross validation.")
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="The seed used for random number generation and sampling.")
parser.add_argument('--scheduler_gamma',
                    type=float,
                    default=0.9,
                    help="Gamma value used for the learning rate scheduler during training.")
parser.add_argument('--fold_index',
                    type=int,
                    default=0,
                    help="The partition index of the held out data as part of k-fold cross validation.")
parser.add_argument('--positive_example_weight',
                    type=int,
                    default=1,
                    help="The weight given to positive examples in the loss function")
parser.add_argument('--dropout_probability',
                    type=float,
                    default=0.3,
                    help="The dropout probability across each layer of the convolution in the homophilous  model.")
parser.add_argument('--max_num_rounds_no_improvement',
                    type=int,
                    default=10,
                    help="The permissible number of validation set evaluations in which the desired metric does not "
                         "improve. If the desired validation metric does not improve within this number of evaluation "
                         "attempts, then early stopping is performed.")
parser.add_argument('--inter_nodes',
                    type=bool,
                    default=False,
                    help="Add intermidated nodes representing the edge types. Works in homophilous type")
# TODO: Fix documentation across this file.


def find_labels_for_batch(batch_data: torch_geometric.data.HeteroData) -> torch.Tensor:
    """

    :param batch_data: A batch of torch geometric heterogeneous data (i.e., a batch of heterophilous graphs).
    :return:
    """
    batch_labels = []
    key = batch_data.node_types[0]  # Return the name of the first node type (e.g., 'claim').

    # Iterate the labels of nodes of the selected type.
    for b in range(len(batch_data[key].y)):
        batch_labels.append(batch_data[key].y[b][0])
    return torch.tensor(batch_labels)


def train(model: torch.nn.Module,
          training_loader: geom_data.DataLoader,
          validation_loader: geom_data.DataLoader,
          epochs: int,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler,
          device,
          experiment_name: str,
          rounds_between_evals: int = 1,
          hetro: bool = False,
          metric_for_early_stopping: str = constants.ACCURACY,
          max_num_rounds_no_improvement: int = 20,
          weights: torch.Tensor = torch.Tensor([1, 1])) -> torch.nn.Module:
    """Train a GCNWithBERTEmbeddings model on examples consisting of persuasive argument knowledge graphs.

    :param model: A torch module consisting of a BERT model (used to produce node embeddings), followed by a GCN.
    :param training_loader: A torch geometric data loader used to feed batches from the training set to the model.
    :param validation_loader: A torch geometric data loader used to feed batches from the validation set to the model.
    :param epochs: The number of iterations over the training set during model training.
    :param optimizer: The torch optimizer used for weight updates during trianing.
    :param scheduler:
    :param device:
    :param experiment_name:
    :param rounds_between_evals: An integer denoting the number of epcohs that occur between each evaluation run.
    :param hetro:
    :param metric_for_early_stopping:
    :param max_num_rounds_no_improvement:
    :param weights:
    :return: A trained model.
    """
    model.to(device)
    highest_accuracy = 0
    lowest_loss = math.inf
    num_rounds_no_improvement = 0
    epoch_with_optimal_performance = 0
    best_model_dir_path = os.path.join(os.getcwd(), 'tmp')
    utils.ensure_dir_exists(best_model_dir_path)
    best_model_path = os.path.join(best_model_dir_path, f'tmp_{experiment_name}.pt')
    model_improved_on_validation_set = False
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        for sampled_data in tqdm(training_loader):
            sampled_data.to(device)
            optimizer.zero_grad()
            if hetro:
                y = find_labels_for_batch(batch_data=sampled_data)
                batch = None
                if args.hetero_type == constants.EDGES:
                    batch = sampled_data[constants.NODE].batch
                out = model.forward(
                    x_dict=sampled_data.x_dict,
                    edge_index_dict=sampled_data.edge_index_dict,
                    device=device)
            else:
                y = sampled_data.y
                out = model.forward(x=sampled_data.x,
                                    edge_index=sampled_data.edge_index,
                                    batch=sampled_data.batch,
                                    device=device)
            preds = torch.argmax(out, dim=1)
            y_one_hot = F.one_hot(y, 2)
            loss = F.cross_entropy(out.float(), y_one_hot.float().to(device), weight=weights.to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_correct_preds = (preds == y.to(device)).sum().float()
            accuracy = num_correct_preds / y.shape[0] * 100
            epoch_acc += accuracy
            num_batches += 1
        learning_rate = scheduler.get_last_lr()[0]
        scheduler.step()
        wandb.log({
            f"{constants.TRAIN} {constants.ACCURACY}": epoch_acc / num_batches,
            f"{constants.TRAIN} {constants.EPOCH}": epoch,
            f"{constants.TRAIN} {constants.LOSS}": epoch_loss / num_batches,
            f"learning rate": learning_rate
        })

        # Perform evaluation on the validation set.
        if epoch % rounds_between_evals == 0:
            model.eval()
            epoch_loss = 0.0
            epoch_acc = 0.0
            num_batches = 0
            model.eval()
            for sampled_data in tqdm(validation_loader):
                sampled_data.to(device)
                if hetro:
                    y = find_labels_for_batch(batch_data=sampled_data)
                    # TODO: Enable support for heterophilous edge GNNs.
                    if args.hetero_type == constants.EDGES:
                        raise NotImplementedError('Heterophilous Edge GNNs are not yet supported.')
                    out = model.forward(
                        x_dict=sampled_data.x_dict,
                        edge_index_dict=sampled_data.edge_index_dict,
                        device=device)
                else:
                    y = sampled_data.y
                    out = model.forward(x=sampled_data.x,
                                        edge_index=sampled_data.edge_index,
                                        batch=sampled_data.batch,
                                        device=device)
                preds = torch.argmax(out, dim=1)
                y_one_hot = F.one_hot(y, 2)
                loss = F.cross_entropy(out.float(), y_one_hot.float().to(device), weight=weights.to(device))
                num_correct_preds = (preds == y.to(device)).sum().float()
                accuracy = num_correct_preds / y.shape[0] * 100
                num_batches += 1
                epoch_loss += loss.item()
                epoch_acc += accuracy
            validation_loss = epoch_loss / num_batches
            validation_acc = epoch_acc / num_batches
            wandb.log({f"{constants.VALIDATION} {constants.ACCURACY}": validation_acc,
                       f"{constants.VALIDATION} {constants.EPOCH}": epoch,
                       f"{constants.VALIDATION} {constants.LOSS}": validation_loss})
            if metric_for_early_stopping == constants.LOSS and validation_loss <= lowest_loss:
                lowest_loss = validation_loss
                num_rounds_no_improvement = 0
                epoch_with_optimal_performance = epoch
                model_improved_on_validation_set = True
                torch.save(model.state_dict(), best_model_path)
            elif metric_for_early_stopping == constants.ACCURACY and validation_acc > highest_accuracy:
                highest_accuracy = validation_acc
                num_rounds_no_improvement = 0
                epoch_with_optimal_performance = epoch
                model_improved_on_validation_set = True
                torch.save(model.state_dict(), best_model_path)
            else:
                num_rounds_no_improvement += 1

            if num_rounds_no_improvement == max_num_rounds_no_improvement:
                print(f'Performing early stopping after {epoch} epochs.\n'
                      f'Optimal model obtained from epoch #{epoch_with_optimal_performance}')
                if model_improved_on_validation_set:
                    model.load_state_dict(torch.load(best_model_path))
                    os.remove(best_model_path)
                break
    return model


def evaluate(
        model: torch.nn.Module,
        dataset: geom_data.DataLoader,
        split_name: str,
        hetro: bool,
        device):
    """
    Evaluate the performance of a GCNWithBertEmbeddings model.

    The test set used for this evaluation consists of distinct examples from those used by the model during training.

    :param model: A torch module consisting of a BERT model (used to produce node embeddings), followed by a GCN.
    :param dataset: A CMVKGDataLoader instance
    """
    model.eval()
    model.to(device)
    with torch.no_grad():
        preds_list = []
        targets_list = []
        for sampled_data in tqdm(dataset):
            sampled_data.to(device)
            if hetro:
                y = find_labels_for_batch(batch_data=sampled_data)
                if args.hetero_type == constants.EDGES:
                    batch = sampled_data[constants.NODE].batch
                out = model.forward(
                    x_dict=sampled_data.x_dict,
                    edge_index_dict=sampled_data.edge_index_dict,
                    device=device)
            else:  # Homophilous GNN
                y = sampled_data.y
                out = model.forward(x=sampled_data.x,
                                    edge_index=sampled_data.edge_index,
                                    batch=sampled_data.batch,
                                    device=device)
            preds = torch.argmax(out, dim=1).cpu()
            preds_list.append(preds)
            targets_list.append(y.cpu())
        preds_list = np.concatenate(preds_list)
        targets_list = np.concatenate(targets_list)
        eval_metrics = metrics.compute_metrics(num_labels=constants.NUM_LABELS,
                                               preds=preds_list,
                                               targets=targets_list,
                                               split_name=split_name)
        for metric_name, metric_value in eval_metrics.items():
            wandb.summary[f"eval_{metric_name}"] = metric_value
        return eval_metrics


if __name__ == '__main__':
    args = parser.parse_args()

    assert args.encoder_type in {"bert", "sbert"}
    assert args.aggregation_type in {"super_node", "avg_pooling", "max_pooling"}

    num_classes = 2
    hetero_type = args.hetero_type
    hetero = utils.str2bool(args.hetero)

    use_max_pooling = args.aggregation_type == "max_pooling"
    use_super_node = args.aggregation_type == "super_node"

    args_dict = vars(args)
    for parameter, value in args_dict.items():
        print(f'{parameter}: {value}')
    utils.set_seed(args.seed)
    num_node_features = constants.BERT_HIDDEN_DIM
    current_path = os.getcwd()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Initializing model: {args.model} device: {device}')
    hidden_dim = list(map(int, args.gcn_hidden_layer_dim.split(" ")))

    # TODO: Creating the knowledge graph datasets takes a lot of time. As of now we make this a one time cost by saving
    #  and loading these datasets. In the future we should optimize the data creation process from a runtime
    #  perspective.
    if args.data == constants.CMV:
        dir_name = os.path.join(current_path, "cmv_modes", "knowledge_graph_datasets")
    elif args.data == constants.UKP:
        dir_name = os.path.join(current_path, constants.UKP)
    else:
        raise Exception(f'{args.data} not implemented')

    utils.ensure_dir_exists(dir_name)
    file_name = f'{"cmv" if args.data == constants.CMV else "ukp"}_{args.encoder_type}_' \
                f'{"supernode" if use_super_node else ""}_' \
                f'{"heterophelous" if hetero else "homophelous" }_dataset.pt'
    # TODO: Remove code duplication below by creating helper functions (either in this file or utils.py).

    if hetero:  # Todo add and option for hetero and UKP
        print('Initializing heterophealous dataset')
        if hetero_type == constants.NODES:
            if args.data == constants.CMV:
                if os.path.exists(os.path.join(dir_name, file_name)):
                    kg_dataset = torch.load(os.path.join(dir_name, file_name))
                else:
                    kg_dataset = CMVKGHetroDataset(
                        current_path + "/cmv_modes/change-my-view-modes-master",
                        version=constants.v2_path,
                        debug=utils.str2bool(args.debug),
                        model_name=(
                            constants.BERT_BASE_CASED if args.encoder_type == 'bert'
                            else "sentence-transformers/all-distilroberta-v1"
                        ),
                    )
                    torch.save(kg_dataset, os.path.join(dir_name, file_name))
            else:
                if os.path.exists(os.path.join(dir_name, file_name)):
                    kg_dataset = torch.load(os.path.join(dir_name, file_name))
                else:
                    kg_dataset = UKPHetroDataset(
                        model_name=(
                            constants.BERT_BASE_CASED if args.encoder_type == 'bert'
                            else "sentence-transformers/all-distilroberta-v1"
                        ),
                        debug=utils.str2bool(args.debug),
                        super_node=use_super_node)
                    torch.save(kg_dataset, os.path.join(dir_name, file_name))
        elif hetero_type == constants.EDGES:  # Todo not working
            if os.path.exists(os.path.join(dir_name, 'hetro_edges_dataset.pt')):
                kg_dataset = torch.load(os.path.join(dir_name, 'hetro_edges_dataset.pt'))
            else:
                kg_dataset = CMVKGHetroDatasetEdges(
                    current_path + "/cmv_modes/change-my-view-modes-master",
                    version=constants.v2_path,
                    debug=utils.str2bool(args.debug))
                torch.save(kg_dataset, os.path.join(dir_name, 'hetro_edges_dataset.pt'))
        else:
            raise RuntimeError(f"Unsupported hetero type: {hetero_type}")
        data = kg_dataset[2]

    else:
        print(f'initializing homophealous {args.data} dataset')

        if args.data == constants.CMV:
            kg_dataset = CMVKGDataset( #TODO remove once done creating intermidiate edges
                current_path + "/cmv_modes/change-my-view-modes-master",
                version=constants.v2_path,
                debug=utils.str2bool(args.debug),
                model_name=(
                    constants.BERT_BASE_CASED if args.encoder_type == 'bert'
                    else "sentence-transformers/all-distilroberta-v1"
                ),
                super_node=use_super_node,
                iter_nodes=args.inter_nodes
            )
            # if os.path.exists(os.path.join(dir_name, file_name)):
            #     kg_dataset = torch.load(os.path.join(dir_name, file_name))
            # else:
            #     kg_dataset = CMVKGDataset(
            #         current_path + "/cmv_modes/change-my-view-modes-master",
            #         version=constants.v2_path,
            #         debug=utils.str2bool(args.debug),
            #         model_name=(
            #             constants.BERT_BASE_CASED if args.encoder_type == 'bert'
            #             else "sentence-transformers/all-distilroberta-v1"
            #         ),
            #         super_node=use_super_node,
            #     )
            #     torch.save(kg_dataset, os.path.join(dir_name, file_name))
        elif args.data == constants.UKP:
            kg_dataset = UKPDataset( #Todo change back to save file option
                model_name=(
                    constants.BERT_BASE_CASED if args.encoder_type == 'bert'
                    else "sentence-transformers/all-distilroberta-v1"
                ),
                debug=utils.str2bool(args.debug),
                super_node=use_super_node
            )
            # if os.path.exists(os.path.join(dir_name, file_name)):
            #     kg_dataset = torch.load(os.path.join(dir_name, file_name))
            # else:
            #     kg_dataset = UKPDataset(
            #         model_name=(
            #             constants.BERT_BASE_CASED if args.encoder_type == 'bert'
            #             else "sentence-transformers/all-distilroberta-v1"
            #         ),
            #         debug=utils.str2bool(args.debug),
            #         super_node=use_super_node
            #     )
            #     torch.save(kg_dataset, os.path.join(dir_name, file_name))

    num_of_examples = len(kg_dataset.dataset)
    shuffled_indices = random.sample(range(num_of_examples), num_of_examples)

    print(f'Initializing model: {args.model} device: {device}')
    hidden_dim = list(map(int, args.gcn_hidden_layer_dim.split(" ")))

    if hetero:
        model = HeteroGNN(hidden_channels=hidden_dim,
                          out_channels=num_classes,
                          hetero_metadata=data.metadata(),
                          conv_type=args.model,
                          encoder_type=args.encoder_type,
                          dropout_prob=args.dropout_probability)
    else:
        model = HomophiliousGNN(hidden_channels=hidden_dim,
                                out_channels=num_classes,
                                conv_type=args.model,
                                use_max_pooling=use_max_pooling,
                                super_node=use_super_node,
                                encoder_type=args.encoder_type,
                                dropout_prob=args.dropout_probability)
    # TODO: Create functions which generate model, experiment, and run names for wandb given the relevant parameters
    #  provided via flags.
    if utils.str2bool(args.use_k_fold_cross_validation):
        train_metrics = []
        validation_metrics = []
        test_metrics = []
        for validation_split_index in range(args.num_cross_validation_splits):
            dl_train, dl_val, dl_test = create_dataloaders_for_k_fold_cross_validation(
                kg_dataset,
                dataset_type="graph",
                num_of_examples=num_of_examples,
                shuffled_indices=shuffled_indices,
                batch_size=args.batch_size,
                val_percent=args.val_percent,
                test_percent=args.test_percent,
                k_fold_index=validation_split_index)
            split_model = copy.deepcopy(model)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_gamma)
            model_name, group_name, run_name = utils.create_gnn_run_and_model_names(
                encoder_type=args.encoder_type,
                use_hetero_graph=utils.str2bool(args.hetero),
                graph_convolution_type=args.model,
                use_max_pooling=use_max_pooling,
                dataset_name=args.data,
                validation_split_index=args.fold_index,
                learning_rate=args.learning_rate,
                scheduler_gamma=args.scheduler_gamma,
                gcn_hidden_layer_dim=args.gcn_hidden_layer_dim,
                weight_decay=args.weight_decay,
                dropout_probability=args.dropout_probability,
                use_super_node=use_super_node,
                positive_example_weight=args.positive_example_weight,
                seed=args.seed,
            )
            run = wandb.init(project="persuasive_arguments",
                             entity="persuasive_arguments",
                             config=args,
                             name=run_name,
                             reinit=True)
            model = train(model=model,
                          training_loader=dl_train,
                          validation_loader=dl_val,
                          epochs=args.num_epochs,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          rounds_between_evals=args.rounds_between_evals,
                          hetro=hetero,
                          device=device,
                          experiment_name=run_name,
                          max_num_rounds_no_improvement=args.max_num_rounds_no_improvement,
                          weights=torch.Tensor([1, args.positive_example_weight]))
            train_metrics.append(
                evaluate(model,
                         dl_train,
                         hetro=hetero,
                         device=device,
                         split_name=constants.TRAIN)
            )
            validation_metrics.append(
                evaluate(model,
                         dl_val,
                         hetro=hetero,
                         device=device,
                         split_name=constants.VALIDATION)
            )
            test_metrics.append(
                evaluate(model,
                         dl_test,
                         hetro=hetero,
                         device=device,
                         split_name=constants.TEST)
            )
            run.finish()

        validation_metric_aggregates = utils.aggregate_metrics_across_splits(validation_metrics)
        train_metric_aggregates = utils.aggregate_metrics_across_splits(train_metrics)
        test_metric_aggregates = utils.aggregate_metrics_across_splits(test_metrics)
        print(f'\n*** Train Metrics: ***')
        train_metric_averages, train_metric_stds = utils.get_metrics_avg_and_std_across_splits(
            metric_aggregates=train_metric_aggregates,
            split_name=constants.TRAIN,
            print_results=True)
        print(f'\n*** Validation Metrics: ***')
        validation_metric_averages, validation_metric_stds = utils.get_metrics_avg_and_std_across_splits(
            metric_aggregates=validation_metric_aggregates,
            split_name=constants.VALIDATION,
            print_results=True)
        print(f'\n*** Test Metrics: ***')
        test_metric_averages, test_metric_stds = utils.get_metrics_avg_and_std_across_splits(
            metric_aggregates=test_metric_aggregates,
            split_name=constants.TEST,
            print_results=True)

    else:
        dl_train, dl_val, dl_test = create_dataloaders_for_k_fold_cross_validation(
            kg_dataset,
            dataset_type="graph",
            shuffled_indices=shuffled_indices,
            num_of_examples=num_of_examples,
            batch_size=args.batch_size,
            val_percent=args.val_percent,
            test_percent=args.test_percent,
            k_fold_index=args.fold_index)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_gamma)

        # The experiment names are unique within a sweep. Each experiment consists of k runs, where k is the number
        # of k-fold cross validation folds. These k runs within each experiment are grouped.
        model_name, group_name, run_name = utils.create_gnn_run_and_model_names(
            encoder_type=args.encoder_type,
            use_hetero_graph=utils.str2bool(args.hetero),
            graph_convolution_type=args.model,
            use_max_pooling=use_max_pooling,
            dataset_name=args.data,
            validation_split_index=args.fold_index,
            learning_rate=args.learning_rate,
            scheduler_gamma=args.scheduler_gamma,
            gcn_hidden_layer_dim=args.gcn_hidden_layer_dim,
            weight_decay=args.weight_decay,
            dropout_probability=args.dropout_probability,
            use_super_node=use_super_node,
            positive_example_weight=args.positive_example_weight,
            seed=args.seed,
        )

        wandb.init(
            project="persuasive_arguments",
            entity="persuasive_arguments",
            group=group_name,
            config=args,
            name=run_name,
            dir='.')
        config = wandb.config
        model = train(model=model,
                      training_loader=dl_train,
                      validation_loader=dl_val,
                      epochs=args.num_epochs,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      rounds_between_evals=args.rounds_between_evals,
                      max_num_rounds_no_improvement=args.max_num_rounds_no_improvement,
                      hetro=hetero,
                      device=device,
                      experiment_name=run_name,
                      weights=torch.Tensor([1, config.positive_example_weight]))
        print(metrics.perform_evaluation_on_splits(
            eval_fn=(
                lambda dataloader, split_name, device: evaluate(model, dataloader, split_name, hetero, device)),
            device=device,
            train_loader=dl_train,
            validation_loader=dl_val,
            test_loader=dl_test
        ))

