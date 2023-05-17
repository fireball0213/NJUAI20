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
        label[i] = label[i].replace('\n', '')
        print(label[i], label_transform_vers(label[i]))
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


def label_transform(labels):  # 把none分类为0效果最好
    if 'positive' in labels:
        labels = '1'
    elif 'negative' in labels:
        labels = '-1'
    elif 'none' in labels:
        labels = '0'
    elif 'neutral' in labels:
        labels = '0'
    else:
        labels = '1'
    return labels


def label_transform_new(labels, a, b):  # 根据不同模型调整a,b
    if 'positive' in labels:
        labels = '1'
    elif 'negative' in labels:
        labels = '-1'
    elif 'none' in labels:
        labels = str(a)
    elif 'neutral' in labels:
        labels = '0'
    else:
        labels = str(b)
    return labels


def label_transform_vers(labels):  # 反向转换
    labels = labels.replace('\n', '').replace(" ", "")
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


def record_time(func):
    def wrapper(*args, **kwargs):  # 包装带参函数
        start_time = time.perf_counter()
        a = func(*args, **kwargs)  # 包装带参函数
        end_time = time.perf_counter()
        print('time=', end_time - start_time)
        return a  # 有返回值的函数必须给个返回

    return wrapper


def write_f(label, file='201300086.txt'):
    # 检验file是否存在
    with open(file, 'w', encoding='utf-8') as f:
        for i in range(len(label)):
            f.write(label[i] + '\n')
    print(file, "成功输出")


def decode_result_1(labels, pred_labels):  # 反了，后面也是反的，懒得改
    num_lst = []
    word_lst = []
    for i in range(len(pred_labels)):
        # 读取每行从第一个冒号到第一个逗号的信息
        tmp = pred_labels[i][pred_labels[i].find(':') + 1:]
        num_lst.append(int(tmp))
        word_lst.append(pred_labels[i][:pred_labels[i].find(':')])

    pre_lst = []
    for i in range(len(labels)):
        # 读取每行从word_lst[i]到第一个逗号的信息
        tmp = labels[i][labels[i].find(':') + 1:labels[i].find(',')]
        pre_lst.append(label_transform(tmp))
    # print(word_lst)
    # print(pre_lst)
    # 将num_lst和pre_lst组合成字典，并按num_lst的值升序排序
    dic = dict(zip(num_lst, pre_lst))
    dic = sorted(dic.items(), key=lambda x: x[0])
    # print(dic)

    # 提取dic中所有值
    values = []
    for i in range(len(dic)):
        values.append(dic[i][1])
    return values


def decode_result_2(pred_labels, labels, a, b):
    num_lst = []
    word_lst = []
    for i in range(len(labels)):
        # 读取每行从第一个冒号到第一个逗号的信息
        tmp = labels[i][labels[i].find(':') + 1:]
        num_lst.append(int(tmp))
        word_lst.append(labels[i][:labels[i].find(':')])

    pre_lst = []
    for i in range(len(pred_labels)):
        tmp = pred_labels[i][pred_labels[i].find(word_lst[i]) + len(word_lst[i]) + 1:]
        if tmp.find(',') == -1:
            # 除去tmp中换行符
            tmpp = tmp.replace('\n', '')
            # print(word_lst[i], '|', tmpp)
            pre_lst.append(label_transform_new(tmpp, a, b))
        else:
            tmpp = tmp[:tmp.find(',')].replace('\n', '')
            # print(word_lst[i], '|', tmpp)
            pre_lst.append(label_transform_new(tmpp, a, b))
    # print(word_lst)
    # print(pre_lst)
    # 将num_lst和pre_lst组合成字典，并按num_lst的值升序排序
    dic = dict(zip(num_lst, pre_lst))
    dic = sorted(dic.items(), key=lambda x: x[0])
    # print(dic)

    # 提取dic中所有值
    values = []
    for i in range(len(dic)):
        values.append(dic[i][1])
    return values


if __name__ == "__main__":
    # sentence, word = load_test('Dataset/test.txt')
    # sentence, word = process_standard(sentence, word, 'Dataset/test_standard.csv')

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
    model_num = str(2)
    pre_file = 'Results/pred_labels' + model_num + '.txt'
    file = 'Results/labels' + model_num + '.txt'
    result = decode_result_1(pre_file, file)
    # result = decode_result_2(pre_file, file)
    write_f(result, 'Results/201300086.txt')
