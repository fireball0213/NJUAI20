"""
@filename:main.py
@author:201300086
@time:2023-05-16
"""
import os
import time
import warnings
import pandas as pd
import torch
from InstructABSA.data_prep import DatasetLoader
from InstructABSA.utils import T5Generator, T5Classifier
from InstructABSA.config import Config
from instructions import InstructionsHandler
from dataset import load_test, label_transform, write_f, \
    record_time, decode_result_1, decode_result_2

warnings.filterwarnings('ignore')
try:
    use_mps = True if torch.has_mps else False
except:
    use_mps = False

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def train(id_train_file_path, id_test_file_path, model_checkpoint, model_out_path):
    # 输出路径中加上当前时间#model_out_path=os.path.join(model_checkpoint,"train")

    # print('Model output path: ', model_out_path)

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
        'per_device_train_batch_size': 32,
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
    return model_trainer


@record_time
def inference_sentence(sentence, word, model_checkpoint):
    # Set Global Values
    config = Config()
    instruct_handler = InstructionsHandler()
    instruct_handler.load_instruction_set2()
    # model_checkpoint = config.model_checkpoint
    indomain = 'bos_instruct1'
    t5_exp = T5Classifier(model_checkpoint)

    bos_instruction_id = instruct_handler.atsc[indomain]
    eos_instruction = instruct_handler.atsc['eos_instruct']

    # config.test_input, aspect_term = config.test_input.split('|')[0], config.test_input.split('|')[1]
    labels = []
    for i in range(len(sentence)):
        model_input = bos_instruction_id + sentence[i] + f'. The aspect term is: {word[i]}' + eos_instruction
        input_ids = t5_exp.tokenizer(model_input, return_tensors="pt").input_ids
        outputs = t5_exp.model.generate(input_ids, max_length=config.max_token_length)
        label = t5_exp.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f'({i}/{len(sentence)}): {label_transform(label)} | {sentence[i]} | {word[i]}')
        labels.append(label_transform(label))
    return labels


@record_time
def inference(model_checkpoint, id_train_file_path, batch_size, mode='test'):
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
    id_tr_pred_labels = t5_exp.get_labels(tokenized_dataset=id_tokenized_ds, sample_set='train', batch_size=batch_size)
    id_tr_labels = [i.strip() for i in id_ds['train']['labels']]
    return id_tr_pred_labels, id_tr_labels


import numpy as np


def vote(a, b, c, d):
    a, b, c, d = int(a), int(b), int(c), int(d)
    tmp = 1
    if a == b or a == c or a == d:
        tmp = a
    elif b == c or b == d:
        tmp = b
    elif c == d:
        tmp = c
    else:
        tmp = 1
    return str(tmp)


def vote_3(a, b, c):
    a, b, c = int(a), int(b), int(c)
    tmp = 1
    if a == b or a == c:
        tmp = a
    elif b == c:
        tmp = b
    else:
        tmp = 1
    return str(tmp)


def model_sentence(num, model, eval=True):
    file = 'Results/sentence_' + str(num) + '_0.txt'
    if eval == False:
        with open(file, 'r') as f:
            results1 = f.readlines()
    else:
        sentence, word = load_test('Dataset/test.txt')
        results1 = inference_sentence(sentence, word, model)
        write_f(results1, file)
    return results1


def model_joint(num, model, eval=True, batch_size=8, mode='test'):
    file = 'Results/joint_' + str(num) + '_' + str(batch_size) + '.txt'
    # file='Results/201300086.txt'
    if num == 2:
        a = 0
        b = 1
    else:
        a = 1
        b = 1
    if eval == False:
        with open(file, 'r') as f:
            result = f.readlines()
    else:
        if mode == 'test':
            pred_labels, labels = inference(model_checkpoint=model, id_train_file_path='Dataset/test_standard.csv',
                                            batch_size=batch_size)
            # print(pred_labels)#打乱后的预测值
            # print(labels)#标签标记了原有顺序
            result = decode_result_2(pred_labels, labels, a, b)
            write_f(result, file)

        else:  # 计算测试集上准确率
            # TODO:输出带序号文件
            pred_labels, labels = inference(model_checkpoint=model, id_train_file_path='Dataset/train_standard.csv',
                                            batch_size=batch_size, mode=mode)
            result = decode_result_2(pred_labels, labels, a, b)
            file = 'Results/joint_train' + str(num) + '.txt'
            write_f(result, file)
            # 计算准确率并输出
    return result


def model_test(num, model_out_path, batch_size, eval=True):
    # file='Results/201300086.txt'
    file = 'Results/joint_train_' + model_out_path[-1] + '.txt'
    if eval == False:
        with open(file, 'r') as f:
            result = f.readlines()
    else:
        if num == 2:
            a = 0
            b = 1
        elif num == 1:
            a = 1
            b = 1
        else:
            a = 1
            b = 0
        pred_labels, labels = inference(model_checkpoint=model_out_path, id_train_file_path='Dataset/test_standard.csv',
                                        batch_size=batch_size)
        # print(pred_labels)#打乱后的预测值
        # print(labels)#标签标记了原有顺序
        result = decode_result_2(pred_labels, labels, a, b)
        write_f(result, file)
    return result


if __name__ == "__main__":
    # results1 = []
    # for i in range(len(results)):
    #     results1.append(label_transform(results[i]))
    # print(results1)
    # write_f(results1, 'Results/201300086.txt')

    # Train
    id_train_file_path = 'Dataset/train_standard.csv'
    id_test_file_path = 'Dataset/train_standard.csv'
    # model_checkpoint = 'Models/joint_task/kevinscariajoint_tk-instruct-base-def-pos-combined'
    model_checkpoint = 'Models/joint_task/kevinscariajoint_tk-instruct-base-def-pos-neg-neut-combined'  # num=1,0.7633# num=2,0.8160
    model_out_path = os.path.join(model_checkpoint, 'train1')
    # model_trainer=train(id_train_file_path, id_test_file_path,model_checkpoint=model_checkpoint,model_out_path=model_out_path)
    # Test

    result1 = model_test(2, model_out_path, 32, eval=False)
    model_out_path2 = 'Models/joint_task/kevinscariajoint_tk-instruct-base-def-pos-combined/train2'  # num=1,0.7553# num=2,0.8026
    result2 = model_test(2, model_out_path2, 32, eval=False)

    # 投票集成
    # result1=model_sentence(1,model='Models/astk_task/kevinscariaatsc_tk-instruct-base-def-pos-combined')# 0.81@
    # result2=model_sentence(2,model='Models/astk_task/kevinscariaatsc_tk-instruct-base-def-pos-neg-neut-combined')#0.7964
    # 0.8125（none=1,else=1）0.7919(none=1,else=0)0.7919(none=0,else=1)
    result3 = model_joint(1, eval=False, batch_size=32,
                          model='Models/joint_task/kevinscariajoint_tk-instruct-base-def-pos-neg-neut-combined')
    # 0,8069(none=1,else=1)0.8142(none=0,else=1)0.7982(none=1,else=0)
    result4 = model_joint(2, eval=False, batch_size=32,
                          model='Models/joint_task/kevinscariajoint_tk-instruct-base-def-pos-combined')

    result = []
    tmp_lst = []
    for i in range(len(result1)):
        result.append(vote_3(result1[i], result1[i], result4[i]))
        # result.append(vote(result1[i],result2[i],result3[i],result4[i]))
    # 将结果写入文件
    write_f(result, 'Results/201300086.txt')

# 1234：8151
# 234:8232
# 123:8098
# 124:8151
