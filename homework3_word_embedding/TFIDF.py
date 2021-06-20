'''
@File    :   TFIDF.py
@Author  :   Zehong Ma
@Version :   1.0
@Contact :   zehongma@qq.com
@Desc    :   word representation based on TFIDF
'''

import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

def build_TFIDF_vec(filename):
    vocab_dic = {} 
    vocab_inverse_dic = {}
    index = 0
    lines = None
    with open(filename,'r',encoding='utf-8') as f:
        lines = f.read().split('\n')
        for line in lines:
            tokens = line.split(' ')
            for token in tokens:
                if token not in vocab_dic.keys():
                    vocab_dic[token] = index 
                    vocab_inverse_dic[index]=token
                    index += 1
    vocab_size = len(vocab_dic.keys())
    print("vocab num is: ",vocab_size)
    if os.path.exists('./TFIDF_vec.npy'):
        mat = np.load('./TFIDF_vec.npy')
        return mat, vocab_dic,vocab_inverse_dic

    mat = np.zeros((vocab_size,vocab_size))
    mask = np.zeros((vocab_size,vocab_size),dtype=np.uint8)

    window = None
    window_size = 5

    N = vocab_size
    for line in lines:
        tokens = line.split(' ')
        token_size = len(tokens)
        
        for  i in range(token_size):
            center_token = tokens[i]
            if i==token_size-2:  #倒数第二个词作为中心词，需要补一个词
                window = tokens[i-2:]
                window.append('。')
            elif i==token_size-1:  #倒数第一个词作为中心词，需要补两个词
                window = tokens[i-2:]
                window.append('。')
                window.append('。')
            elif i==0:
                window = []
                window.append('。')
                window.append('。')
                window.extend(tokens[i:i+3])
            elif i==1:
                window = []
                window.append('。')
                window.extend(tokens[i-1:i+3])
            else:
                window = tokens[i-2:i+3]
            
            for i in range(window_size):
                mat[vocab_dic[center_token]][vocab_dic[window[i]]]+=1
                mask[vocab_dic[center_token]][vocab_dic[window[i]]]=1
    for j in range(vocab_size):
        mat[:][j] = mat[:][j]*(N/np.sum(mask[:,j]))    
    np.save('./TFIDF_vec.npy',mat)
    return mat, vocab_dic,vocab_inverse_dic

def get_similar_words():
    mat, vocab_dic, vocab_inverse_dic = build_TFIDF_vec('./data/input.txt')
    cos_dis = cosine_similarity(mat)
    word_list = ['人民','总统','市场','艰辛','企业','美国','研究','技术','航空','聪明']
    for word in word_list:
        top5=np.argsort(cos_dis[vocab_dic[word]])[-6:]
        print(word,end=': ')
        i = 5
        j = 0
        while i>=0:
            
            if top5[i]==vocab_dic[word]:
                i-=1
                continue
            print('"%s", '%vocab_inverse_dic[top5[i]],end=' ')
            i-=1
            j+=1
            if j==5:
                break
        print()


if __name__== '__main__':
    get_similar_words()
