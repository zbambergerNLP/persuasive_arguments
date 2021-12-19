import os
import metrics
import preprocessing
import transformers
import constants
import probing.models as probing_models
import argparse

# TODO(zbamberger): Resolve Runtime Error when using wandb while probing.
# import wandb
#
# wandb.init(project="persuasive_arguments", entity="zbamberger")

parser = argparse.ArgumentParser(
    description='Process flags for fine-tuning transformers on an argumentation downstream task.')
parser.add_argument('--fine_tune_model',
                    type=bool,
                    default=False,
                    required=False,
                    help='Whether or not a pre-trained transformer language model should undergo fine-tuning.')
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
                    required=False,
                    help='True instructs the fine-tuned model to generate new hidden embeddings corresponding to each'
                         ' example. These embeddings serve as the input to an MLP probing model. False assumes that'
                         ' such a dataset already exists, and is stored in json file within the ./probing directory.')
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
                    default=3,
                    required=False,
                    help="The number of training rounds over the dataset.")
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


if __name__ == "__main__":
    args = parser.parse_args()
    fine_tune_model = args.fine_tune_model
    probe_model_on_premise_modes = args.probe_model_on_premise_modes
    fine_tuned_model_path = args.fine_tuned_model_path
    dataset_name = args.dataset_name
    model_checkpoint_name = args.model_checkpoint_name
    num_training_ephocs = args.num_training_ephocs
    output_dir = args.output_dir
    logging_dir = args.logging_dir
    per_device_train_batch_size = args.per_device_train_batch_size
    per_device_eval_batch_size = args.per_device_eval_batch_size
    warmup_steps = args.warmup_steps
    weight_decay = args.weight_decay
    logging_steps = args.logging_steps
    generate_new_probing_dataset = args.generate_new_probing_dataset

    print(f'dataset_name: {dataset_name}')
    print(f'fine-tuning model: {fine_tune_model}')
    print(f'model_checkpoint_name: {model_checkpoint_name}')
    print(f'probe_model_on_premise_modes: {probe_model_on_premise_modes}')
    if probe_model_on_premise_modes:
        print(f'generate_new_probing_dataset: {generate_new_probing_dataset}')
    else:
        print(f'num_training_ephocs: {num_training_ephocs}')
        print(f'output_dir: {output_dir}')
        print(f'logging_dir: {logging_dir}')
        print(f'per_device_train_batch_size: {per_device_train_batch_size}')
        print(f'per_device_eval_batch_size: {per_device_eval_batch_size}')
        print(f'warmup_steps: {warmup_steps}')
        print(f'weight_decay: {weight_decay}')
        print(f'logging_steps: {logging_steps}')

    if fine_tune_model:
        configuration = transformers.TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_training_ephocs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir=logging_dir,
            logging_steps=logging_steps,
            report_to=["wandb"],
        )
        model = transformers.BertForSequenceClassification.from_pretrained(
            model_checkpoint_name,
            num_labels=constants.NUM_LABELS)
        dataset = preprocessing.get_cmv_dataset(
            dataset_name=dataset_name,
            tokenizer=transformers.BertTokenizer.from_pretrained(constants.BERT_BASE_CASED)
        )
        dataset = dataset.train_test_split()
        train_dataset = preprocessing.CMVDataset(dataset[constants.TRAIN])
        test_dataset = preprocessing.CMVDataset(dataset[constants.TEST])
        trainer = transformers.Trainer(
            model=model,
            args=configuration,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=metrics.compute_metrics,
        )
        trainer.train()
        trainer.save_model()
        metrics = trainer.evaluate()
    if probe_model_on_premise_modes:
        current_path = os.getcwd()
        probing_dir_path = os.path.join(current_path, 'probing')
        ethos_dataset, logos_dataset, pathos_dataset = preprocessing.get_cmv_probing_datasets(
            tokenizer=transformers.BertTokenizer.from_pretrained(constants.BERT_BASE_CASED))
        ethos_probing_model, ethos_eval_metrics = (
            probing_models.probe_model_with_premise_mode(constants.ETHOS,
                                                         ethos_dataset,
                                                         current_path,
                                                         fine_tuned_model_path,
                                                         generate_new_probing_dataset=generate_new_probing_dataset,
                                                         learning_rate=1e-3,
                                                         training_batch_size=16,
                                                         eval_batch_size=64))
        print(ethos_eval_metrics[constants.CLASSIFICATION_REPORT])
        logos_probing_model, logos_eval_metrics = (
            probing_models.probe_model_with_premise_mode(constants.LOGOS,
                                                         logos_dataset,
                                                         current_path,
                                                         fine_tuned_model_path,
                                                         generate_new_probing_dataset=False,
                                                         learning_rate=1e-3,
                                                         training_batch_size=16,
                                                         eval_batch_size=64))
        print(logos_eval_metrics[constants.CLASSIFICATION_REPORT])
        pathos_probing_model, pathos_eval_metrics = (
            probing_models.probe_model_with_premise_mode(constants.PATHOS,
                                                         pathos_dataset,
                                                         current_path,
                                                         fine_tuned_model_path,
                                                         generate_new_probing_dataset=False,
                                                         learning_rate=1e-3,
                                                         training_batch_size=16,
                                                         eval_batch_size=64))
        print(pathos_eval_metrics[constants.CLASSIFICATION_REPORT])