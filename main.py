import fine_tuning.fine_tuning as fine_tuning
import probing.probing as probing
import preprocessing
import constants

import os
import transformers
import argparse

import wandb

"""
How to run fine-tuning on the relation extraction probing task:
srun --gres=gpu:1 -p nlp python3 main.py \
    --fine_tune_model_on_argument_relations True \
    --probing_num_training_epochs 8
    
How to run probing experiments:
srun --gres=gpu:1 -p nlp python3 main.py --probing_model "logistic_regression" \
    --fine_tuned_model_path "results/checkpoint-500" \
    --fine_tune_model_on_argument_relations True \
    --probing_num_training_epochs 8
"""

parser = argparse.ArgumentParser(
    description='Process flags for fine-tuning transformers on an argumentation downstream task.')
parser.add_argument('--probing_wandb_entity',
                    type=str,
                    default='zbamberger',
                    help="The wandb entity used to track training.")


# General probing parameters
parser.add_argument('--probing_model',
                    type=str,
                    default=constants.LOGISTIC_REGRESSION,
                    required=False,
                    help="The string name of the model type used for probing. Either logistic regression or MLP.")
parser.add_argument('--fine_tuned_model_path',
                    type=str,
                    required=False,
                    default=os.path.join('results', 'checkpoint-1500'),
                    help='The path fine-tuned model trained on the argument persuasiveness prediction task.')
parser.add_argument('--model_checkpoint_name',
                    type=str,
                    default=constants.BERT_BASE_CASED,
                    required=False,
                    help="The name of the checkpoint from which we load our model and tokenizer.")
parser.add_argument('--probing_model_learning_rate',
                    type=float,
                    default=1e-3,
                    required=False,
                    help="The learning rate used by the probing model for the probing task.")
parser.add_argument('--probing_model_scheduler_gamma',
                    type=float,
                    default=0.9,
                    help="Decays the learning rate of each parameter group by gamma every epoch.")


# Probing on intra-argument relations:
parser.add_argument('--probe_model_on_intra_argument_relations',
                    type=bool,
                    default=True,
                    required=False,
                    help="Whether or not a pre-trained transformer language model should be trained and evaluated on a"
                         "probing task. In this case, the probing task involves classifying whether or not a relation"
                         "exist between the first argument preposition (either a claim or premise), and a second "
                         "argument preposition.")
parser.add_argument('--generate_new_relations_probing_dataset',
                    type=bool,
                    default=False,
                    required=False,
                    help='True instructs the fine-tuned model to generate new hidden embeddings corresponding to each'
                         ' example. These embeddings serve as the input to an MLP probing model. False assumes that'
                         ' such a dataset already exists, and is stored in json file within the ./probing directory.')
parser.add_argument('--fine_tune_model_on_argument_relations',
                    type=bool,
                    default=True,
                    required=False,
                    help='Fine tune the model specified in `model_checkpoint_name` on the relation prediction probing'
                         ' dataset.')


# Probing on premise modes:
parser.add_argument('--probe_model_on_premise_modes',
                    type=bool,
                    default=True,
                    required=False,
                    help=('Whether or not a pre-trained transformer language model should be trained and evaluated'
                          'on a probing task. In this case, the probing task involves classifying the argumentation '
                          'mode (i.e., the presence of ethos, logos, or pathos) within a premise.'))
parser.add_argument('--generate_new_premise_mode_probing_dataset',
                    type=bool,
                    default=False,
                    # default=True,
                    required=False,
                    help='True instructs the fine-tuned model to generate new hidden embeddings corresponding to each'
                         ' example. These embeddings serve as the input to an MLP probing model. False assumes that'
                         ' such a dataset already exists, and is stored in json file within the ./probing directory.')
parser.add_argument('--fine_tune_model_on_premise_modes',
                    type=bool,
                    default=False,
                    required=False,
                    help='Fine tune the model specified in `model_checkpoint_name` on the probing datasets.')
parser.add_argument('--probe_claim_and_premise_pair',
                    type=bool,
                    default=True,
                    required=False,
                    help='True if the probing dataset consists of a (claim, supporting_premise) pair. False if the '
                         'probing dataset consists strictly of premises.')
parser.add_argument('--multi_class_premise_mode_probing',
                    type=bool,
                    default=True,
                    required=False,
                    help='True if the label space for classifying premise mode (probing task) is the superset of '
                         '{"ethos", "logos", "pathos}. False if the label space is binary, where True indicates that'
                         'the premise entails the dataset-specific argumentative mode.')


# Configuration flags:
parser.add_argument('--probing_output_dir',
                    type=str,
                    default='./results',
                    required=False,
                    help="The directory in which probing model results are stored.")
parser.add_argument('--probing_num_training_epochs',
                    type=int,
                    default=2,
                    required=False,
                    help="The number of training rounds over the probing dataset.")
parser.add_argument('--probing_per_device_train_batch_size',
                    type=int,
                    default=16,
                    help="The number of examples per batch per device during probe training.")
parser.add_argument('--probing_per_device_eval_batch_size',
                    type=int,
                    default=64,
                    help="The number of examples per batch per device during probe evaluation.")
parser.add_argument('--probing_warmup_steps',
                    type=int,
                    default=500,
                    help="The number of warmup steps the model takes at the start of probing.")
parser.add_argument('--probing_weight_decay',
                    type=float,
                    default=0.01,
                    help="The weight decay parameter supplied to the optimizer for use during probing.")
parser.add_argument('--probing_logging_dir',
                    type=str,
                    default="./logs",
                    required=False,
                    help="The directory in which the model stores logs.")
parser.add_argument('--probing_logging_steps',
                    type=int,
                    default=10,
                    help="The number of steps a model takes between recording to logs.")


if __name__ == "__main__":
    args = parser.parse_args()

    if args.probing_wandb_entity:
        wandb.init(project="persuasive_arguments", entity=args.probing_wandb_entity)

    args_dict = vars(args)
    for parameter, value in args_dict.items():
        print(f'{parameter}: {value}')

    configuration = transformers.TrainingArguments(
        output_dir=args.probing_output_dir,
        num_train_epochs=args.probing_num_training_epochs,
        per_device_train_batch_size=args.probing_per_device_train_batch_size,
        per_device_eval_batch_size=args.probing_per_device_eval_batch_size,
        warmup_steps=args.probing_warmup_steps,
        weight_decay=args.probing_weight_decay,
        logging_dir=args.probing_logging_dir,
        logging_steps=args.probing_logging_steps,
        report_to=["wandb"],
    )
    model = transformers.BertForSequenceClassification.from_pretrained(
        args.model_checkpoint_name,
        num_labels=constants.NUM_LABELS)

    # We are now considering model performance on argumentative probing tasks.
    current_path = os.getcwd()
    probing_dir_path = os.path.join(current_path, constants.PROBING)
    tokenizer = transformers.BertTokenizer.from_pretrained(constants.BERT_BASE_CASED)

    if args.probe_model_on_intra_argument_relations:
        intra_argument_relations_probing_dataset = preprocessing.get_intra_argument_relations_probing_dataset(
                tokenizer=transformers.BertTokenizer.from_pretrained(constants.BERT_BASE_CASED))

        if args.fine_tune_model_on_argument_relations:
            _, intra_argument_relations_eval_metrics = (
                fine_tuning.fine_tune_on_task(dataset=intra_argument_relations_probing_dataset,
                                              model=model,
                                              configuration=configuration,
                                              task_name=constants.INTRA_ARGUMENT_RELATIONS,
                                              is_probing=True))
            print(f'intra_argument_relations_eval_metrics:\n{intra_argument_relations_eval_metrics}')

        if args.probe_model_on_intra_argument_relations:
            probing.probe_model_on_intra_argument_relations_dataset(
                dataset=intra_argument_relations_probing_dataset,
                current_path=current_path,
                fine_tuned_model_path=args.fine_tuned_model_path,
                pretrained_checkpoint_name=args.model_checkpoint_name,
                generate_new_probing_dataset=args.generate_new_relations_probing_dataset,
                probing_model=args.probing_model,
                learning_rate=args.probing_model_learning_rate,
                training_batch_size=args.probing_per_device_train_batch_size,
                eval_batch_size=args.probing_per_device_eval_batch_size,
                num_epochs=args.probing_num_training_epochs,
                scheduler_gamma=args.probing_model_scheduler_gamma)

    if args.multi_class_premise_mode_probing:
        pretrained_multiclass_model = transformers.BertForSequenceClassification.from_pretrained(
            args.model_checkpoint_name,
            num_labels=len(constants.PREMISE_MODE_TO_INT))

        dataset = preprocessing.get_multi_class_cmv_probing_dataset(tokenizer=tokenizer,
                                                                    with_claims=args.probe_claim_and_premise_pair)
        if args.fine_tune_model_on_premise_modes:
            _, multi_class_eval_metrics = (
                fine_tuning.fine_tune_on_task(dataset=dataset,
                                              model=model,
                                              configuration=configuration,
                                              task_name=constants.MULTICLASS,
                                              is_probing=True))
            print(f'multi_class_eval_metrics:\n{multi_class_eval_metrics}')
        if args.probe_model_on_premise_modes:
            probing.probe_model_on_multiclass_premise_modes(
                dataset,
                current_path,
                args.fine_tuned_model_path,
                pretrained_checkpoint_name=args.model_checkpoint_name,
                generate_new_probing_dataset=args.generate_new_premise_mode_probing_dataset,
                probing_model=args.probing_model,
                learning_rate=args.probing_model_learning_rate,
                training_batch_size=args.probing_per_device_train_batch_size,
                eval_batch_size=args.probing_per_device_eval_batch_size,
                num_epochs=args.probing_num_training_epochs,
                scheduler_gamma=args.probing_model_scheduler_gamma)

    # Create datasets for each premise mode for binary classification.
    if args.probe_claim_and_premise_pair:
        ethos_dataset, logos_dataset, pathos_dataset = preprocessing.get_cmv_probing_datasets(
            tokenizer=transformers.BertTokenizer.from_pretrained(constants.BERT_BASE_CASED), with_claims=True)
    else:
        ethos_dataset, logos_dataset, pathos_dataset = preprocessing.get_cmv_probing_datasets(
            tokenizer=transformers.BertTokenizer.from_pretrained(constants.BERT_BASE_CASED), with_claims=False)

    # Perform fine-tuning on each of the premise mode binary classification tasks.
    if args.fine_tune_model_on_premise_modes:
        ethos_model, ethos_eval_metrics = (
            fine_tuning.fine_tune_on_task(dataset=ethos_dataset,
                                          model=model,
                                          configuration=configuration,
                                          task_name=constants.BINARY_PREMISE_MODE_PREDICTION,
                                          is_probing=True,
                                          premise_mode=constants.ETHOS)
        )
        print(f'ethos_eval_metrics:\n{ethos_eval_metrics}')
        logos, logos_eval_metrics = (
            fine_tuning.fine_tune_on_task(dataset=logos_dataset,
                                          model=model,
                                          configuration=configuration,
                                          task_name=constants.BINARY_PREMISE_MODE_PREDICTION,
                                          is_probing=True,
                                          premise_mode=constants.LOGOS))
        print(f'logos_eval_metrics:\n{logos_eval_metrics}')
        pathos_model, pathos_eval_metrics = (
            fine_tuning.fine_tune_on_task(dataset=pathos_dataset,
                                          model=model,
                                          configuration=configuration,
                                          task_name=constants.BINARY_PREMISE_MODE_PREDICTION,
                                          is_probing=True,
                                          premise_mode=constants.PATHOS))
        print(f'pathos_eval_metrics:\n{pathos_eval_metrics}')
    if args.probe_model_on_premise_modes:
        probing.probe_model_on_premise_mode(mode=constants.ETHOS,
                                            dataset=ethos_dataset,
                                            current_path=current_path,
                                            fine_tuned_model_path=args.fine_tuned_model_path,
                                            model_checkpoint_name=args.model_checkpoint_name,
                                            generate_new_probing_dataset=args.generate_new_premise_mode_probing_dataset,
                                            probing_model=args.probing_model,
                                            learning_rate=args.probing_model_learning_rate,
                                            training_batch_size=args.probing_per_device_train_batch_size,
                                            eval_batch_size=args.probing_per_device_eval_batch_size,
                                            num_epochs=args.probing_num_training_epochs,
                                            scheduler_gamma=args.probing_model_scheduler_gamma)
        probing.probe_model_on_premise_mode(mode=constants.LOGOS,
                                            dataset=logos_dataset,
                                            current_path=current_path,
                                            fine_tuned_model_path=args.fine_tuned_model_path,
                                            model_checkpoint_name=args.model_checkpoint_name,
                                            generate_new_probing_dataset=args.generate_new_premise_mode_probing_dataset,
                                            probing_model=args.probing_model,
                                            learning_rate=args.probing_model_learning_rate,
                                            training_batch_size=args.probing_per_device_train_batch_size,
                                            eval_batch_size=args.probing_per_device_eval_batch_size,
                                            num_epochs=args.probing_num_training_epochs,
                                            scheduler_gamma=args.probing_model_scheduler_gamma)
        probing.probe_model_on_premise_mode(mode=constants.PATHOS,
                                            dataset=pathos_dataset,
                                            current_path=current_path,
                                            fine_tuned_model_path=args.fine_tuned_model_path,
                                            model_checkpoint_name=args.model_checkpoint_name,
                                            generate_new_probing_dataset=args.generate_new_premise_mode_probing_dataset,
                                            probing_model=args.probing_model,
                                            learning_rate=args.probing_model_learning_rate,
                                            training_batch_size=args.probing_per_device_train_batch_size,
                                            eval_batch_size=args.probing_per_device_eval_batch_size,
                                            num_epochs=args.probing_num_training_epochs,
                                            scheduler_gamma=args.probing_model_scheduler_gamma)
