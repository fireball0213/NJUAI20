# -*- coding: UTF-8 -*- #
"""
@filename:dataset.py
@author:201300086
@time:2023-05-16
"""
import pandas as pd
import time


def load_train(filename):
    with open(filename, encoding='utf-8') as f:
        text = f.readlines()
        sentence = []
        word = []
        labels = []
        for i in range(len(text)):
            if i % 3 == 0:
                sentence.append(text[i][:-1])
            elif i % 3 == 1:
                word.append(text[i][:-1])
            else:
                labels.append(text[i])
        for i in range(len(sentence)):
            sentence[i] = sentence[i].replace('$T$', word[i])
    return sentence, word, labels


def load_test(filename):
    with open(filename, encoding='utf-8') as f:
        text = f.readlines()
        sentence = []
        word = []
        for i in range(len(text)):
            if i % 3 == 0:
                sentence.append(text[i][:-1])
            elif i % 3 == 1:
                word.append(text[i][:-1])
        for i in range(len(sentence)):
            sentence[i] = sentence[i].replace('$T$', word[i])
    return sentence, word


# 转换成字典形式
def process_standard_train(sentence, word, label, out_path):  # [{'term':'cab ride', 'polarity':'positive'}]
    new_sentence = []
    new_words = []
    for i in range(len(word)):
        new_words.append("[{'term':'" + word[i] + "', 'polarity':'" + label_transform_vers(label[i]) + "'}]")

    df = pd.DataFrame({'raw_text': sentence, 'aspectTerms': new_words})
    # df输出csv文件
    df.to_csv(out_path, index=False, sep=',')
    return sentence, new_words, label


def process_standard(sentence, word, out_path):  # [{'term':'cab ride', 'polarity':'positive'}]
    new_sentence = []
    new_words = []
    for i in range(len(word)):
        new_words.append("[{'term':'" + word[i] + "', 'polarity':'" + str(i) + "'}]")

    df = pd.DataFrame({'raw_text': sentence, 'aspectTerms': new_words})
    # df输出csv文件
    df.to_csv(out_path, index=False, sep=',')
    return sentence, new_words


def process_standard_1(sentence, word, out_path):
    words = []
    # 有的句子对应了多个aspect term，需要将这些被同一个sentence对应的aspect term放在一起，保存在words中
    for i in range(len(sentence)):
        tmp = []
        for j in range(len(sentence)):
            if sentence[i] == sentence[j]:
                tmp.append(word[j])
        words.append(tmp)
    df = pd.DataFrame({'raw_text': sentence, 'aspectTerms': word, 'Terms': words})
    # df输出csv文件
    df.to_csv(out_path, index=False, sep=',')
    return sentence, word, words


def label_transform(labels):  # 把none分类为0效果最好
    if 'positive' in labels:
        labels = '1'
    elif 'negative' in labels:
        labels = '-1'
    elif 'none' in labels:
        labels = '0'
    else:
        labels = '0'
    return labels


def label_transform_vers(labels):  # 反向转换
    if labels == '1':
        labels = 'positive'
    elif labels == '-1':
        labels = 'negative'
    elif labels == '0':
        labels = 'none'
    return labels


# def label_transform(labels):#'positive':1,'negative':-1,'none'or'neutral':0
#     for i in range(len(labels)):
#         if labels[i]=='positive':
#             labels[i]='1'
#         elif labels[i]=='negative':
#             labels[i]='-1'
#         else:
#             labels[i]='0'
#     return labels

def inference_transorm(labels):
    # 将labels这个列表中的每个元素中冒号后的的值输出到一个列表中
    new_labels = []
    for i in range(len(labels)):
        tmp = labels[i].split(':')[1]
        new_labels.append(label_transform(tmp))
    return new_labels


def record_time(func):
    def wrapper(*args, **kwargs):  # 包装带参函数
        start_time = time.perf_counter()
        a = func(*args, **kwargs)  # 包装带参函数
        end_time = time.perf_counter()
        print('time=', end_time - start_time)
        return a  # 有返回值的函数必须给个返回

    return wrapper


def write_f(label, file='201300086.txt'):
    with open(file, 'w', encoding='utf-8') as f:
        for i in range(len(label)):
            f.write(label[i] + '\n')
    print(file, "成功输出")


def decode_result_1(pre_file, file):
    with open(file, encoding='utf-8') as f:
        text = f.readlines()
        num_lst = []
        word_lst = []
        for i in range(len(text)):
            # 读取每行从第一个冒号到第一个逗号的信息
            tmp = text[i][text[i].find(':') + 1:]
            num_lst.append(int(tmp))
            word_lst.append(text[i][:text[i].find(':')])
    with open(pre_file, encoding='utf-8') as f:
        text = f.readlines()
        pre_lst = []
        for i in range(len(text)):
            # 读取每行从word_lst[i]到第一个逗号的信息
            tmp = text[i][text[i].find(':') + 1:text[i].find(',')]
            pre_lst.append(label_transform(tmp))
    print(word_lst)
    print(pre_lst)
    # 将num_lst和pre_lst组合成字典，并按num_lst的值升序排序
    dic = dict(zip(num_lst, pre_lst))
    dic = sorted(dic.items(), key=lambda x: x[0])
    print(dic)

    # 提取dic中所有值
    values = []
    for i in range(len(dic)):
        values.append(dic[i][1])
    return values


def decode_result_2(pre_file, file):
    with open(file, encoding='utf-8') as f:
        text = f.readlines()
        num_lst = []
        word_lst = []
        for i in range(len(text)):
            # 读取每行从第一个冒号到第一个逗号的信息
            tmp = text[i][text[i].find(':') + 1:]
            num_lst.append(int(tmp))
            word_lst.append(text[i][:text[i].find(':')])
    with open(pre_file, encoding='utf-8') as f:
        text = f.readlines()
        pre_lst = []
        for i in range(len(text)):
            tmp = text[i][text[i].find(word_lst[i]) + len(word_lst[i]) + 1:]
            if tmp.find(',') == -1:
                # 除去tmp中换行符
                tmpp = tmp.replace('\n', '')
                # print(word_lst[i], '|', tmpp)
                pre_lst.append(label_transform(tmpp))
            else:
                tmpp = tmp[:tmp.find(',')].replace('\n', '')
                # print(word_lst[i], '|', tmpp)
                pre_lst.append(label_transform(tmpp))
    print(word_lst)
    print(pre_lst)
    # 将num_lst和pre_lst组合成字典，并按num_lst的值升序排序
    dic = dict(zip(num_lst, pre_lst))
    dic = sorted(dic.items(), key=lambda x: x[0])
    print(dic)

    # 提取dic中所有值
    values = []
    for i in range(len(dic)):
        values.append(dic[i][1])
    return values


if __name__ == "__main__":
    sentence, word = load_test('Dataset/test.txt')
    sentence, word = process_standard(sentence, word, 'Dataset/test_standard.csv')
    sentence, word, words = process_standard_1(sentence, word, 'Dataset/test_standard_plus.csv')

    sentence_tr, word_tr, label_tr = load_train('Dataset/train.txt')
    sentence_tr, word_tr, label_tr = process_standard_train(sentence_tr, word_tr, label_tr,
                                                            'Dataset/train_standard.csv')
    # print(result_lst2)
    #
    # results1=[]
    # for i in range(len(result_lst2)):
    #     results1.append(label_transform(result_lst2[i]))
    # print(results1)
    # write_f(results1, 'Results/201300086.txt')

    pre_file = 'Results/pred_labels3.txt'
    file = 'Results/labels3.txt'
    result = decode_result_1(pre_file, file)
    # result = decode_result_2(pre_file, file)
    write_f(result, 'Results/201300086.txt')

# 失败的造轮子：
# 将joint预训练模型输出的打乱的预测结果梳理成正常顺序的标记
# def decode_result(result_lst1,sentence,word,words):#sentence和word是两个等长列表，分别存放句子和对应的aspect term
#     # 将result_lst1中重复元素去除
#     result_lst1 = list(set(result_lst1))
#     print(result_lst1)
#     # 将result_lst1中每个元素都转化为一个字典，这个元素中的冒号的个数代表了的键值对的个数，其中每个键值对的键是冒号前的aspect term，值是冒号后的标记
#     result_lst2 = []
#     for i in range(len(result_lst1)):
#         tmp1 = result_lst1[i].split(',')
#         tmp=[]
#         #删除tmp中不含冒号的元素
#         for j in range(len(tmp1)):
#             if ":" in tmp1[j]:
#                 tmp.append(tmp1[j])
#         tmp_dict = {}
#         for j in range(len(tmp)):
#             tmp_dict[tmp[j].split(':')[0]] = tmp[j].split(':')[1]
#         #去掉tmp_dict中所有键首尾和值首尾的空格,将更新后的键值存成新的字典
#         tmp_dict1={}
#         for key,value in tmp_dict.items():
#             # tmp_dict1[key.replace(' ','')] = value.replace(' ','')
#             if key.startswith(' '):
#                 key=key[1:]
#             if key.endswith(' '):
#                 key=key[:-1]
#             if value.startswith(' '):
#                 value=value[1:]
#             if value.endswith(' '):
#                 value=value[:-1]
#             tmp_dict1[key]=value
#         result_lst2.append(tmp_dict1)
#     print(result_lst2)
#
#     fail_lst=[]
#     for i in range(len(sentence)):
#         success=0
#         for j in range(len(result_lst2)):
#             #如果words[i]中所有元素都是result_lst2[j]中的键值，而且len(words[i])==result_lst2[j]中键值对的个数，输出result_lst2[j]
#             if all([x in result_lst2[j].keys() for x in words[i]]) and len(words[i])-len(result_lst2[j])==0:#
#                 print(f'({i}/{len(sentence)}): {result_lst2[j]} | {words[i]} | {word[i]} | {sentence[i]}')
#                 success=1
#         if success==0:
#             fail_lst.append(i)
#     for i in fail_lst:
#         print(f'(fail:{i}/{len(sentence)}): {sentence[i]} | {words[i]} | {word[i]}')
#         # 输出result_lst2中所有包含以word[i]为键的字典
#         for j in range(len(result_lst2)):
#             if word[i] in result_lst2[j].keys():
#                 print(result_lst2[j])
#
#     print(len(fail_lst))

# fail_lst=[]
# for i in range(len(sentence)):
#     success=0
#     for j in range(len(result_lst1)):
#         #如果words[i]中所有元素都在result_lst1[j]中，而且len(words[i])==result_lst1[j]中冒号的个数，输出result_lst1[j]
#         if all([x in result_lst1[j] for x in words[i]]) and len(words[i])==result_lst1[j].count(':'):
#             #判断words[i]中所有元素在result_lst1[j]中时是否在开头位置，或者前一个非空格字符是逗号，或后一个字符非空格字符是冒号
#             if result_lst1[j][result_lst1[j].find(words[i][-1])+len(words[i])]==':':
#                 print(f'({i}/{len(sentence)}): {result_lst1[j]} | {words[i]} | {word[i]}')
#                 success=1
#     if success==0:
#         fail_lst.append(i)
# for i in fail_lst:
#     print(f'(fail:{i}/{len(sentence)}): {sentence[i]} | {words[i]} | {word[i]}')
# print(len(fail_lst))
