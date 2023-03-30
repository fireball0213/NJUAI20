# -*- coding: UTF-8 -*- #
"""
@filename:model.py
@author:201300086
@time:2023-03-28
"""
from dataset import preprocess_keywords, get_pretrain_features
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import torch
import numpy as np
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
from dataset import record_time


# 朴素贝叶斯：0.724（不process最高）
def Multinomial_TF(train_data, train_labels, textwords):
    vectorizer = TfidfVectorizer()
    train_features = vectorizer.fit_transform(train_data)

    # 训练朴素贝叶斯分类器
    clf = MultinomialNB()
    clf.fit(train_features, train_labels)

    features = vectorizer.transform(textwords)
    preds = clf.predict(features)
    train_pred = clf.predict(train_features)
    return preds, train_pred


@record_time
def Ensemble(train_data, train_labels, textwords):
    _, _, train_features = get_pretrain_features(train_data)
    # 朴素贝叶斯不能有负输入，需要标准化
    # train_features = (train_features - train_features.min()) / (train_features.max() - train_features.min())

    # 基学习器
    lr = LogisticRegression(C=0.3, max_iter=10000, solver='liblinear', penalty='l2', random_state=42)

    nb = MultinomialNB(alpha=2, fit_prior=True)  # 0.97,0.725.2,0.727.#MultinomialNB()#0.724

    dt = DecisionTreeClassifier(random_state=42)  # 0.664

    """
    RandomForestClassifier
        n_estimators=100:0.804
        n_estimators=100,max_features='log2':0.8
        n_estimators=100, oob_score=True:0.804(8s)
        n_estimators=1000, oob_score=True:0.814(53s)
    """
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)  # , oob_score=True

    """
    SVC
        linear:783
        rbf:814
        poly,degree=3:818
        poly,degree=2:805
        poly,degree=4:811
        poly,coef0=1:0.823
        poly,coef0=2:0.833
        poly,coef0=3:0.831
        poly,coef0=2.5,C=1, gamma=0.1:0.835
        poly,coef0=2.5,C=0.8, gamma=0.1:0.837
    
    NuSVC
        nu=0.3:0.844
        nu=0.2:0.841
    """
    # sv = svm.SVC(random_state=42, kernel='poly', coef0=2.5,C=0.8, gamma=0.1 ) #
    sv = svm.SVC(random_state=42, C=10, gamma=0.1)  # 0.814#{'C': 10, 'gamma': 0.1}0.842
    svnu = svm.NuSVC(random_state=42, nu=0.2)

    knn = KNeighborsClassifier(n_jobs=-1)  # 0.735

    """
    Bagging调参
        默认决策树0.747
        n_estimators=10:0.83
        n_estimators=100:0.837
        n_estimators=10,warm_start=True:0.83
        n_estimators=10,oob_score=True:0.83
    """
    bag = BaggingClassifier(random_state=42, base_estimator=sv, n_estimators=10, n_jobs=-1)

    """
    AdaBoostClassifier
        默认决策树learning_rate=1,n_estimators=50:0.748
        learning_rate=0.1:0.727
        learning_rate=1,n_estimators=100:0.761
        learning_rate=0.9,n_estimators=100:0.773
        learning_rate=0.9,n_estimators=1000:0.791
        learning_rate=0.7,n_estimators=100:0.772
        learning_rate>1会明显变差
    """
    ada = AdaBoostClassifier(learning_rate=0.9, n_estimators=200, random_state=42)

    gb = GradientBoostingClassifier(random_state=42)  # 0.783

    # 参数寻优
    # clf = sv
    # # param_grid = {'nu': [0.1,0.2,0.3,0.4,0.5,0.7,0.9], }#'gamma': [0.01, 0.05, 0.1, 0.5]
    # param_grid = {'C':[0.3,0.5,1,2,5]}#,'gamma': [0.01, 0.1, 1,10]
    # # # }#'alpha': lst#[0.0001,0.01,0.1, 1.0, 2.0],#'fit_prior': [True, False]'penalty':['l1','l2','elasticnet','none']
    # grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5)
    # grid_search.fit(train_features, train_labels)
    # # 输出最优参数和交叉验证得分
    # print("Best parameters: ", grid_search.best_params_)
    # print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
    # # clf=svm.NuSVC(random_state=42,nu=grid_search.best_params_['nu'])
    # # clf=LogisticRegression(C=grid_search.best_params_['C'],max_iter=10000)#,penalty=grid_search.best_params_['penalty']
    # clf=svm.SVC(C=grid_search.best_params_['C'],random_state=42,kernel='poly',coef0=2.5,gamma=0.1)

    # Stacking学习法集成
    """
    StackingClassifier
        ('rf', rf), ('lr', lr),('gb',gb),('knn',knn),('bag',bag),('ada',ada),('sv',sv),('svnu',svnu)
        rf, lr, dt +sv=0.846
        rf, lr  +sv = 0.847   lr,rf, +sv = 0.847
        rf, lr,gb,knn + sv = 0.847
        rf, lr,bag,ada+ sv = 0.851
        rf, lr,bag+ sv = 0.851
        rf, lr,bag,gb+ sv = 0.846
    rf和bag调参后：
        lr,knn,rf,bag+sv=0.857
        lr,rf,bag+sv=0.855
    """
    clf = StackingClassifier(estimators=[('lr', lr), ('knn', knn), ('rf', rf), ('bag', bag)],
                             final_estimator=sv, passthrough=True, verbose=1,
                             n_jobs=-1)  # , stack_method='predict_proba'

    # clf=sv
    # Voting投票法集成
    """
    VotingClassifier：
        rf, lr,gb,knn + sv = 0.82左右
    """
    # clf = VotingClassifier(estimators=[('rf', rf), ('lr', lr), ('gb', gb), ('knn', knn)], voting='soft')
    # clf = VotingClassifier(estimators=[ ('knn', knn), ('lr', lr)], voting='soft')
    # clf = VotingClassifier(estimators=[('nb', nb), ('lr', lr)], voting='hard')

    clf.fit(train_features, train_labels)

    _, _, test_features = get_pretrain_features(textwords)
    # test_features = (test_features - test_features.min()) / (test_features.max() - test_features.min())
    preds = clf.predict(test_features)
    train_pred = clf.predict(train_features)

    return preds, train_pred

# 定义Embedding层
# embedding = torch.nn.Embedding.from_pretrained(pretrained_weights)
