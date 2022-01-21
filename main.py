import fine_tuning
import probing.probing as probing
import preprocessing
import constants

import os
import transformers
import argparse

import wandb

parser = argparse.ArgumentParser(
    description='Process flags for fine-tuning transformers on an argumentation downstream task.')
parser.add_argument('--fine_tune_model',
                    type=bool,
                    default=False,
                    required=False,
                    help='Whether or not a pre-trained transformer language model should undergo fine-tuning.')
parser.add_argument('--probe_model_before_fine_tuning',
                    type=bool,
                    default=True,
                    required=False,
                    help="Whether or not a pre-trained transformer language model should be probed before fine-tuning.")
parser.add_argument('--probe_model_on_premise_modes',
                    type=bool,
                    default=True,
                    required=False,
                    help=('Whether or not a pre-trained transformer language model should be trained and evaluated'
                          'on a probing task.\nIn this case, the probing task involves classifying the argumentation '
                          'mode (i.e., the presence of ethos, logos, or pathos) within a premise.'))
parser.add_argument('--generate_new_probing_dataset',
                    type=bool,
                    default=False,
                    # default=True,
                    required=False,
                    help='True instructs the fine-tuned model to generate new hidden embeddings corresponding to each'
                         ' example. These embeddings serve as the input to an MLP probing model. False assumes that'
                         ' such a dataset already exists, and is stored in json file within the ./probing directory.')
parser.add_argument('--probing_model',
                    type=str,
                    # default=constants.LOGISTIC_REGRESSION,
                    default=constants.MLP,
                    required=False,
                    help="The string name of the model type used for probing. Either logistic regression or MLP.")
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
parser.add_argument('--multi_class_cmv_probing',
                    type=bool,
                    default=True,
                    required=False,
                    help='True if the label space for classifying premise mode (probing task) is the superset of '
                         '{"ethos", "logos", "pathos}. False if the label space is binary, where True indicates that'
                         'the premise entails the dataset-specific argumentative mode.')
parser.add_argument('--fine_tuned_model_path',
                    type=str,
                    required=False,
                    default=os.path.join('results', 'checkpoint-1500'),
                    help='The path fine-tuned model trained on the argument persuasiveness prediction task.')
parser.add_argument('--dataset_name',
                    type=str,
                    default=constants.CMV_DATASET_NAME,
                    required=False,
                    help='The name of the file in which the downstream dataset is stored.')
parser.add_argument('--model_checkpoint_name',
                    type=str,
                    default=constants.BERT_BASE_CASED,
                    required=False,
                    help="The name of the checkpoint from which we load our model and tokenizer.")
parser.add_argument('--num_training_ephocs',
                    type=int,
                    default=50,
                    required=False,
                    help="The number of training rounds over the dataset.")
parser.add_argument('--probing_model_learning_rate',
                    type=float,
                    default=1e-1,
                    required=False,
                    help="The learning rate used by the probing model for the probing task.")
parser.add_argument('--output_dir',
                    type=str,
                    default='./results',
                    required=False,
                    help="The directory in which model results are stored.")
parser.add_argument('--logging_dir',
                    type=str,
                    default="./logs",
                    required=False,
                    help="The directory in which the model stores logs.")
parser.add_argument('--per_device_train_batch_size',
                    type=int,
                    default=16,
                    help="The number of examples per batch per device during training.")
parser.add_argument('--per_device_eval_batch_size',
                    type=int,
                    default=64,
                    help="The number of examples per batch per device during evaluation.")
parser.add_argument('--warmup_steps',
                    type=int,
                    default=500,
                    help="The number of warmup steps the model takes at the start of training.")
parser.add_argument('--weight_decay',
                    type=float,
                    default=0.01,
                    help="The weight decay parameter supplied to the optimizer for use during training.")
parser.add_argument('--logging_steps',
                    type=int,
                    default=10,
                    help="The number of steps a model takes between recording to logs.")
parser.add_argument('--wandb_entity',
                    type=str,
                    default='zbamberger',
                    help="The wandb entity used to track training.")


if __name__ == "__main__":
    args = parser.parse_args()

    if args.wandb_entity:
        wandb.init(project="persuasive_arguments", entity=args.wandb_entity)

    args_dict = vars(args)
    for parameter, value in args_dict.items():
        print(f'{parameter}: {value}')

    configuration = transformers.TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_training_ephocs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        report_to=["wandb"],
    )
    model = transformers.BertForSequenceClassification.from_pretrained(
        args.model_checkpoint_name,
        num_labels=constants.NUM_LABELS)

    if args.fine_tune_model:
        fine_tuned_trainer, downstream_metrics = fine_tuning.fine_tune_on_downstream_task(
            dataset_name=args.dataset_name,
            model=model,
            configuration=configuration)

    # We are now considering model performance on argumentative probing tasks.
    if args.probe_model_on_premise_modes:
        current_path = os.getcwd()
        probing_dir_path = os.path.join(current_path, constants.PROBING)
        if args.multi_class_cmv_probing:
            pretrained_multiclass_model = transformers.BertForSequenceClassification.from_pretrained(
                args.model_checkpoint_name,
                num_labels=len(constants.PREMISE_MODE_TO_INT))
            dataset = preprocessing.get_multi_class_cmv_probing_dataset_with_claims(
                tokenizer=transformers.BertTokenizer.from_pretrained(constants.BERT_BASE_CASED))
            if args.fine_tune_model_on_premise_modes:
                multi_class_model, multi_class_eval_metrics = fine_tuning.fine_tune_model_on_multiclass_premise_mode(
                    current_path,
                    probing_dataset=dataset,
                    model=pretrained_multiclass_model,
                    model_configuration=configuration)
                print(f'multi_class_eval_metrics: {multi_class_eval_metrics}')
            if args.probe_model_on_premise_modes:
                pretrained_probing_model, _, eval_metrics = probing.probe_model_on_multiclass_premise_modes(
                    dataset,
                    current_path,
                    args.fine_tuned_model_path,
                    pretrained_checkpoint_name=args.model_checkpoint_name,
                    generate_new_probing_dataset=args.generate_new_probing_dataset,
                    probing_model=args.probing_model,
                    learning_rate=args.probing_model_learning_rate,
                    training_batch_size=args.per_device_train_batch_size,
                    eval_batch_size=args.per_device_eval_batch_size,
                    num_epochs=args.num_training_ephocs)
        if args.probe_claim_and_premise_pair:
            ethos_dataset, logos_dataset, pathos_dataset = preprocessing.get_cmv_probing_datasets_with_claims(
                tokenizer=transformers.BertTokenizer.from_pretrained(constants.BERT_BASE_CASED))
        else:
            ethos_dataset, logos_dataset, pathos_dataset = preprocessing.get_cmv_probing_datasets(
                tokenizer=transformers.BertTokenizer.from_pretrained(constants.BERT_BASE_CASED))
        if args.fine_tune_model_on_premise_modes:
            ethos_model, ethos_eval_metrics = (
                fine_tuning.fine_tune_model_on_premise_mode(current_path,
                                                            premise_mode=constants.ETHOS,
                                                            probing_dataset=ethos_dataset,
                                                            model=model,
                                                            model_configuration=configuration))
            print(f'ethos_eval_metrics: {ethos_eval_metrics}')
            logos_model, logos_eval_metrics = (
                fine_tuning.fine_tune_model_on_premise_mode(current_path,
                                                            premise_mode=constants.LOGOS,
                                                            probing_dataset=logos_dataset,
                                                            model=model,
                                                            model_configuration=configuration))
            print(f'logos_eval_metrics: {logos_eval_metrics}')
            pathos_model, pathos_eval_metrics = (
                fine_tuning.fine_tune_model_on_premise_mode(current_path,
                                                            premise_mode=constants.PATHOS,
                                                            probing_dataset=pathos_dataset,
                                                            model=model,
                                                            model_configuration=configuration))
            print(f'pathos_eval_metrics: {pathos_eval_metrics}')
        if args.probe_model_on_premise_modes:
            probing.probe_model_on_premise_mode(mode=constants.ETHOS,
                                                dataset=ethos_dataset,
                                                current_path=current_path,
                                                fine_tuned_model_path=args.fine_tuned_model_path,
                                                model_checkpoint_name=args.model_checkpoint_name,
                                                generate_new_probing_dataset=args.generate_new_probing_dataset,
                                                probing_model=args.probing_model,
                                                learning_rate=args.probing_model_learning_rate,
                                                training_batch_size=args.per_device_train_batch_size,
                                                eval_batch_size=args.per_device_eval_batch_size,
                                                num_epochs=args.num_training_ephocs)
            probing.probe_model_on_premise_mode(mode=constants.LOGOS,
                                                dataset=logos_dataset,
                                                current_path=current_path,
                                                fine_tuned_model_path=args.fine_tuned_model_path,
                                                model_checkpoint_name=args.model_checkpoint_name,
                                                generate_new_probing_dataset=args.generate_new_probing_dataset,
                                                probing_model=args.probing_model,
                                                learning_rate=args.probing_model_learning_rate,
                                                training_batch_size=args.per_device_train_batch_size,
                                                eval_batch_size=args.per_device_eval_batch_size,
                                                num_epochs=args.num_training_ephocs)
            probing.probe_model_on_premise_mode(mode=constants.PATHOS,
                                                dataset=pathos_dataset,
                                                current_path=current_path,
                                                fine_tuned_model_path=args.fine_tuned_model_path,
                                                model_checkpoint_name=args.model_checkpoint_name,
                                                generate_new_probing_dataset=args.generate_new_probing_dataset,
                                                probing_model=args.probing_model,
                                                learning_rate=args.probing_model_learning_rate,
                                                training_batch_size=args.per_device_train_batch_size,
                                                eval_batch_size=args.per_device_eval_batch_size,
                                                num_epochs=args.num_training_ephocs)
