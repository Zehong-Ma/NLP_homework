'''
@File    :   build_language_model.py
@Author  :   Zehong Ma
@Version :   1.0
@Contact :   zehongma@qq.com
@Desc    :   None
'''
import numpy as np 
import jieba
import math
from load_data import *

def build_one_gram_LM():
    """
        利用训练集构建一元语法模型，并将模型保存在one_gram_likelihood.npy中
    """
    labels, texts = load_tsv_data('./data/train.tsv')
    vocabs = load_vocabs('./data/vocab.txt')
    total_tokens = 0
    total_tokens_used = 0
    likelihood = np.zeros((len(vocabs),2))
    for i in range(len(texts)):
        tokens = jieba.cut(texts[i])
        for token in tokens:
            if token in vocabs.keys():
                if int(labels[i]) == 0:
                    likelihood[vocabs[token]][0]+=1
                else:
                    likelihood[vocabs[token]][1]+=1
                total_tokens_used+=1
            total_tokens+=1
    print('using ratio of training tokens in one gram LM is %f  [%d/%d]'%(total_tokens_used/total_tokens,total_tokens_used,total_tokens))
    # likelihood_one_gram = likelihood/total_tokens_used
    likelihood_one_gram = np.zeros_like(likelihood)
    likelihood_one_gram[:,0] = (likelihood[:,0]+1)/(np.sum(likelihood[:,0])+len(vocabs))
    likelihood_one_gram[:,1] = (likelihood[:,1]+1)/(np.sum(likelihood[:,1])+len(vocabs))
    np.save('one_gram_likelihood.npy',likelihood_one_gram)

    return likelihood_one_gram

def evaluate_one_gram_LM():
    """
        使用困惑度评估一元语法模型
    """
    labels, texts = load_tsv_data('./data/test.tsv')
    vocabs = load_vocabs('./data/vocab.txt')
    likelihood = np.load('one_gram_likelihood.npy')
    likelihood_sum = np.sum(likelihood,axis=-1)
    texts_perplexity = []
    for text in texts:
        text_perplexity = 1.0
        tokens = jieba.cut(text)
        index=0
        for token in tokens:
            index+=1
        tokens = jieba.cut(text)
        for token in tokens:
            if token in vocabs.keys():
                text_perplexity*=math.pow(likelihood_sum[vocabs[token]],-1/index)
        texts_perplexity.append(text_perplexity)
    print('perpxity of one gram language model is: %f'%(np.mean(texts_perplexity)))
    return   

def build_two_grams_LM():
    """
        利用训练集构建二元语法模型，并将模型保存在two_grams_likelihood.npy中
    """
    labels, texts = load_tsv_data('./data/train.tsv')
    vocabs = load_vocabs('./data/vocab.txt')
    total_tokens = 0
    total_tokens_used = 0
    likelihood = np.zeros((len(vocabs)+2,len(vocabs)+2,2),np.uint16)
    for i in range(len(texts)):
        if int(labels[i])==0:
            index = 0
        else:
            index = 1
        last_token = 'start'
        tokens = jieba.cut(texts[i])
        for token in tokens:
            if token in vocabs.keys():
                if last_token=='start':
                    likelihood[vocabs[token]+1][0][index] += 1
                else:
                    likelihood[vocabs[token]+1][vocabs[last_token]+1][index] += 1
                    last_token = token
                total_tokens_used += 1
            total_tokens += 1
        if last_token=='start':
            likelihood[len(vocabs)+1][0][index] += 1  # 表示[start,end]
        else:
            likelihood[len(vocabs)+1][vocabs[last_token]+1][index] += 1  # 表示[last_token,end]
        total_tokens_used += 1
        total_tokens += 1
    print('using ratio of training tokens in two grams LM is %f  [%d/%d]'%(total_tokens_used/total_tokens,total_tokens_used,total_tokens))
    #likelihood/total_tokens_used
    np.save('two_grams_likelihood.npy',likelihood)
    return total_tokens_used

def evaluate_two_grams_LM():
    """
        使用困惑度评估二元语法模型
    """
    labels, texts = load_tsv_data('./data/test.tsv')
    vocabs = load_vocabs('./data/vocab.txt')
    likelihood = np.load('two_grams_likelihood.npy')
    likelihood_sum = np.sum(likelihood,axis=2)
    texts_perplexity = []
    for text in texts:
        text_perplexity = 1.0
        tokens = jieba.cut(text)
        index=1
        for token in tokens:
            index+=1
        tokens = jieba.cut(text)
        last_token='start'
        for token in tokens:
            if token in vocabs.keys():
                if last_token=='start':
                    likelihood_result = (likelihood_sum[vocabs[token]+1][0]+1)/(np.sum(likelihood_sum[:,0])+len(vocabs))
                    text_perplexity*=math.pow(likelihood_result,-1/index)
                else:
                    likelihood_result = (likelihood_sum[vocabs[token]+1][0]+1)/(np.sum(likelihood_sum[:,vocabs[token]+1])+len(vocabs))
                    text_perplexity*=math.pow(likelihood_result,-1/index)
                last_token=token
        if last_token=='start':
            likelihood_result = (likelihood_sum[len(vocabs)+1][0]+1)/(np.sum(likelihood_sum[:,0])+len(vocabs))
            text_perplexity*=math.pow(likelihood_result,-1/index)
        else:
            likelihood_result = (likelihood_sum[len(vocabs)+1][vocabs[last_token]+1]+1)/(np.sum(likelihood_sum[:,0])+len(vocabs))
            text_perplexity*=math.pow(likelihood_result,-1/index)
        texts_perplexity.append(text_perplexity)
    
    print('perpxity of two gram language model is: %f'%(np.mean(texts_perplexity)))
    return


if __name__ == '__main__':

    build_one_gram_LM()
    build_two_grams_LM()
    evaluate_one_gram_LM()
    evaluate_two_grams_LM()