# -*- coding: UTF-8 -*- #
"""
@filename:main.py
@author:201300086
@time:2023-05-16
"""
import os
import time
import warnings

warnings.filterwarnings('ignore')
import pandas as pd
import torch
from InstructABSA.data_prep import DatasetLoader
from InstructABSA.utils import T5Generator, T5Classifier
from InstructABSA.config import Config
from instructions import InstructionsHandler
from dataset import load_test, label_transform, write_f, inference_transorm, \
    record_time, decode_result_1, decode_result_2

try:
    use_mps = True if torch.has_mps else False
except:
    use_mps = False


def train():
    task_name = 'joint_task'
    # model_checkpoint = 'allenai/tk-instruct-base-def-pos'
    model_checkpoint = 'kevinscaria/joint_tk-instruct-base-def-pos-neg-neut-combined'
    # model_checkpoint='kevinscaria/atsc_tk-instruct-base-def-pos-neg-neut-combined'

    model_out_path = './Models'
    model_out_path = os.path.join(model_out_path, task_name,
                                  f"{model_checkpoint.replace('/', '')}")
    print('Model output path: ', model_out_path)

    # Load the data
    id_train_file_path = 'Dataset/train_standard.csv'
    id_test_file_path = 'Dataset/test_standard.csv'
    id_tr_df = pd.read_csv(id_train_file_path)
    id_te_df = pd.read_csv(id_test_file_path)

    # Get the input text into the required format using Instructions
    instruct_handler = InstructionsHandler()

    # Set instruction_set1 for InstructABSA-1 and instruction_set2 for InstructABSA-2
    instruct_handler.load_instruction_set1()

    # Set bos_instruct1 for lapt14 and bos_instruct2 for rest14. For other datasets, modify the insructions.py file.
    loader = DatasetLoader(id_tr_df, id_te_df)
    if loader.train_df_id is not None:
        loader.train_df_id = loader.create_data_in_joint_task_format(loader.train_df_id, 'term', 'polarity', 'raw_text',
                                                                     'aspectTerms',
                                                                     instruct_handler.joint['bos_instruct1'],
                                                                     instruct_handler.joint['eos_instruct'])
    if loader.test_df_id is not None:
        loader.test_df_id = loader.create_data_in_joint_task_format(loader.test_df_id, 'term', 'polarity', 'raw_text',
                                                                    'aspectTerms',
                                                                    instruct_handler.joint['bos_instruct1'],
                                                                    instruct_handler.joint['eos_instruct'])

    # Create T5 utils object
    t5_exp = T5Generator(model_checkpoint)

    # Tokenize Dataset
    id_ds, id_tokenized_ds, ood_ds, ood_tokenized_ds = loader.set_data_for_training_semeval(
        t5_exp.tokenize_function_inputs)

    # Training arguments
    training_args = {
        'output_dir': model_out_path,
        'evaluation_strategy': "epoch",
        'learning_rate': 5e-5,
        'lr_scheduler_type': 'cosine',
        'per_device_train_batch_size': 8,
        'per_device_eval_batch_size': 16,
        'num_train_epochs': 4,
        'weight_decay': 0.01,
        'warmup_ratio': 0.1,
        'save_strategy': 'no',
        'load_best_model_at_end': False,
        'push_to_hub': False,
        'eval_accumulation_steps': 1,
        'predict_with_generate': True,
        'use_mps_device': use_mps
    }

    # Train model
    model_trainer = t5_exp.train(id_tokenized_ds, **training_args)
    return model_out_path


@record_time
def inference_sentence(sentence, word):
    # Set Global Values
    config = Config()
    instruct_handler = InstructionsHandler()
    instruct_handler.load_instruction_set2()
    model_checkpoint = config.model_checkpoint
    indomain = 'bos_instruct1'
    t5_exp = T5Classifier(model_checkpoint)

    bos_instruction_id = instruct_handler.atsc[indomain]
    eos_instruction = instruct_handler.atsc['eos_instruct']

    # config.test_input, aspect_term = config.test_input.split('|')[0], config.test_input.split('|')[1]
    labels = []
    for i in range(len(sentence)):
        model_input = bos_instruction_id + sentence[i] + f'. The aspect term is: {word[i]}' + eos_instruction
        a1 = time.time()
        input_ids = t5_exp.tokenizer(model_input, return_tensors="pt").input_ids
        a2 = time.time()
        outputs = t5_exp.model.generate(input_ids, max_length=config.max_token_length)
        a3 = time.time()
        print('t2:', a3 - a2, end="  ")
        label = t5_exp.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print()
        print(f'({i}/{len(sentence)}): {label_transform(label)} | {sentence[i]} | {word[i]}')
        labels.append(label_transform(label))
    return labels


@record_time
def inference(model_checkpoint, id_train_file_path='Dataset/example_train.csv'):
    # Load the data
    # id_train_file_path = 'Dataset/example_train.csv'
    id_test_file_path = 'Dataset/example_test.csv'
    id_tr_df = pd.read_csv(id_train_file_path)
    id_te_df = pd.read_csv(id_test_file_path)

    # Get the input text into the required format using Instructions
    instruct_handler = InstructionsHandler()

    # Set instruction_set1 for InstructABSA-1 and instruction_set2 for InstructABSA-2
    instruct_handler.load_instruction_set1()

    # Set bos_instruct1 for lapt14 and bos_instruct2 for rest14. For other datasets, modify the insructions.py file.
    loader = DatasetLoader(id_tr_df, id_te_df)
    if loader.train_df_id is not None:
        loader.train_df_id = loader.create_data_in_joint_task_format(loader.train_df_id, 'term', 'polarity', 'raw_text',
                                                                     'aspectTerms',
                                                                     instruct_handler.joint['bos_instruct1'],
                                                                     instruct_handler.joint['eos_instruct'])
    if loader.test_df_id is not None:
        loader.test_df_id = loader.create_data_in_joint_task_format(loader.test_df_id, 'term', 'polarity', 'raw_text',
                                                                    'aspectTerms',
                                                                    instruct_handler.joint['bos_instruct1'],
                                                                    instruct_handler.joint['eos_instruct'])

    # Model inference - Loading from Checkpoint
    print("Model inference :")
    # t5_exp = T5Generator(model_out_path)
    t5_exp = T5Generator(model_checkpoint)

    # Tokenize Datasets
    id_ds, id_tokenized_ds, ood_ds, ood_tokenzed_ds = loader.set_data_for_training_semeval(
        t5_exp.tokenize_function_inputs)

    # Get prediction labels - Training set
    id_tr_pred_labels = t5_exp.get_labels(tokenized_dataset=id_tokenized_ds, sample_set='train', batch_size=1)
    id_tr_labels = [i.strip() for i in id_ds['train']['labels']]
    return id_tr_pred_labels, id_tr_labels


if __name__ == "__main__":
    # 0.81@
    # sentence,word=load_test('Dataset/test1.txt')
    # results=inference_sentence(sentence, word)
    # # print(results)
    # write_f(results, 'Results/201300086.txt')

    # 0.7964@ kevinscaria/joint_tk-instruct-base-def-pos-neg-neut-combined
    # 0.8044(none=0) 0.80625@(none=1)kevinscaria/joint_tk-instruct-base-def-pos-combined
    # kevinscaria/joint_tk-instruct-base-def-pos-neg-neut-laptops
    # kevinscaria/joint_tk-instruct-base-def-pos-laptops
    # kevinscaria/joint_tk-instruct-base-def-pos-neg-neut-restaurants
    # kevinscaria/joint_tk-instruct-base-def-pos-restaurants
    pred_labels, labels = inference(model_checkpoint='kevinscaria/joint_tk-instruct-base-def-pos-restaurants'
                                    , id_train_file_path='Dataset/test_standard.csv')
    print(pred_labels)
    print(labels)
    pre_file = 'Results/pred_labels6.txt'
    file = 'Results/labels6.txt'
    # 将上面两个列表存入文件
    write_f(pred_labels, pre_file)
    write_f(labels, file)
    result = decode_result_1(pre_file, file)
    write_f(result, 'Results/201300086.txt')

    # results1 = []
    # for i in range(len(results)):
    #     results1.append(label_transform(results[i]))
    # print(results1)
    # write_f(results1, 'Results/201300086.txt')

    # model_out_path=train()

#
