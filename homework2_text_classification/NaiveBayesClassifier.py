'''
@File    :   NaiveBeyesClassifier.py
@Author  :   Zehong Ma
@Version :   1.0
@Contact :   zehongma@qq.com
@Desc    :   None
'''
import numpy as np
import jieba
import math
from load_data import *

def texts_classify():
    """
        测试集文本分类入口
    """
    labels, texts = load_tsv_data('./data/test.tsv')
    vocabs = load_vocabs('./data/vocab.txt')
    one_gram_likelihood = np.load('one_gram_likelihood.npy')
    pred_labels = []
    for text in texts:
        pred_label = text_classify_one_gram(text,one_gram_likelihood,vocabs)
        pred_labels.append(pred_label)
    accuracy, precision, recall, F1 = evaluate(labels,pred_labels)
    print('###### Naive Bayes Classify based on one gram ######')
    print('accuracy:   %.4f'%accuracy)
    print('precision:  %.4f'%precision)
    print('recall:     %.4f'%recall)
    print('F1:         %.4f'%F1)
    del one_gram_likelihood

    two_grams_likelihood = np.load('two_grams_likelihood.npy')
    neg_sum = np.sum(two_grams_likelihood[:,:,0])
    pos_sum = np.sum(two_grams_likelihood[:,:,1])
    pred_labels = []
    for text in texts:
        pred_label = text_classify_two_grams(text,two_grams_likelihood,vocabs,neg_sum,pos_sum)
        pred_labels.append(pred_label)
    accuracy, precision, recall, F1 = evaluate(labels,pred_labels)
    print('###### Naive Bayes Classify based on two grams ######')
    print('accuracy:   %.4f'%accuracy)
    print('precision:  %.4f'%precision)
    print('recall:     %.4f'%recall)
    print('F1:         %.4f'%F1)
    return 

def text_classify_one_gram(text,likelihood,vocabs):
    """
        基于一元语法的朴素贝叶斯分类的方法实现
    """
    tokens = jieba.cut(text)
    pos_prob = 0.0
    neg_prob = 0.0 
    p_neg = np.sum(likelihood[:,0])
    p_pos = np.sum(likelihood[:,1])
    
    for token in tokens:
        if token in vocabs.keys():
            if likelihood[vocabs[token]][1]<1e-9:  # 当该词的词频为0时，使用1e-8代替其在训练集中出现的频率，如果使用了平滑后，此处失效
                likelihood[vocabs[token]][1]=1e-8
                
            if likelihood[vocabs[token]][0]<1e-9:
                likelihood[vocabs[token]][0]=1e-8
            pos_prob += math.log( likelihood[vocabs[token]][1]/p_pos )
            neg_prob += math.log( likelihood[vocabs[token]][0]/p_neg )
    pos_prob+=math.log(p_pos)
    neg_prob+=math.log(p_neg)
    
    if pos_prob>neg_prob:
        return 1
    else:
        return 0

def text_classify_two_grams(text,likelihood,vocabs,neg_sum,pos_sum):
    """
        基于二元语法的朴素贝叶斯分类的方法实现
    """
    token_nums = 661587
    tokens = jieba.cut(text)
    pos_prob = 0.0
    neg_prob = 0.0
    last_token = 'start'
    for token in tokens:
        if token in vocabs.keys():
            if last_token=='start':  # 单独处理 [start,first_token]的情况
                likelihood_result_pos = (likelihood[vocabs[token]+1][0][1]+1)/(pos_sum+len(vocabs))
                likelihood_result_neg = (likelihood[vocabs[token]+1][0][0]+1)/(neg_sum+len(vocabs))
                pos_prob += math.log( likelihood_result_pos )
                neg_prob += math.log( likelihood_result_neg )
            else:
                likelihood_result_pos = (likelihood[vocabs[token]+1][vocabs[last_token]+1][1]+1)/(pos_sum+len(vocabs))
                likelihood_result_neg = (likelihood[vocabs[token]+1][vocabs[last_token]+1][0]+1)/(neg_sum+len(vocabs))
                pos_prob += math.log( likelihood_result_pos )
                neg_prob += math.log( likelihood_result_neg )
            last_token = token
    if likelihood[vocabs[last_token]+1][len(vocabs)+1][1]<1:
        likelihood[vocabs[last_token]+1][len(vocabs)+1][1]=1
    if likelihood[vocabs[last_token]+1][len(vocabs)+1][0]<1:
        likelihood[vocabs[last_token]+1][len(vocabs)+1][0]=1
    likelihood_result_pos = (likelihood[vocabs[last_token]+1][len(vocabs)+1][1]+1)/(pos_sum+len(vocabs))
    likelihood_result_neg = (likelihood[vocabs[last_token]+1][len(vocabs)+1][0]+1)/(neg_sum+len(vocabs))
    pos_prob += math.log(likelihood_result_pos )
    neg_prob += math.log(likelihood_result_neg )
    pos_prob+=math.log(pos_sum/token_nums)
    neg_prob+=math.log(neg_sum/token_nums)
    if pos_prob>neg_prob:
        return 1
    else:
        return 0
    
def evaluate(true_labels, pred_labels):
    """
        分类评价指标计算
    """
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    sample_nums = len(true_labels)
    for i in range(sample_nums):
        if true_labels[i]==1:
            if true_labels[i]==pred_labels[i]:
                tp+=1
            else:
                fn+=1
        if true_labels[i]==0:
            if true_labels[i]==pred_labels[i]:
                tn+=1
            else:
                fp+=1
    
    accuracy = (tp+tn)/sample_nums
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    F1 = 2*precision*recall/(precision+recall)
    return accuracy, precision, recall, F1

if __name__ == '__main__':
    texts_classify()