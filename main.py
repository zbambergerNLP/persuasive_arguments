import metrics
import preprocessing
import transformers
import constants
import argparse


parser = argparse.ArgumentParser(
    description='Process flags for fine-tuning transformers on an argumentation downstream task.')
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

    print(f'dataset_name: {dataset_name}')
    print(f'model_checkpoint_name: {model_checkpoint_name}')
    print(f'num_training_ephocs: {num_training_ephocs}')
    print(f'output_dir: {output_dir}')
    print(f'logging_dir: {logging_dir}')
    print(f'per_device_train_batch_size: {per_device_train_batch_size}')
    print(f'per_device_eval_batch_size: {per_device_eval_batch_size}')
    print(f'warmup_steps: {warmup_steps}')
    print(f'weight_decay: {weight_decay}')
    print(f'logging_steps: {logging_steps}')

    dataset = preprocessing.get_cmv_dataset(
        dataset_name=dataset_name,
        tokenizer=transformers.BertTokenizer.from_pretrained(constants.BERT_BASE_CASED)
    )
    dataset = dataset.train_test_split()
    train_dataset = preprocessing.CMVDataset(dataset[constants.TRAIN])
    test_dataset = preprocessing.CMVDataset(dataset[constants.TEST])
    configuration = transformers.TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_training_ephocs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_dir=logging_dir,
        logging_steps=logging_steps,
    )
    model = transformers.BertForSequenceClassification.from_pretrained(
        model_checkpoint_name,
        num_labels=constants.NUM_LABELS)
    trainer = transformers.Trainer(
        model=model,
        args=configuration,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=metrics.compute_metrics,
    )
    trainer.save_model()
    trainer.train()
    trainer.evaluate()
