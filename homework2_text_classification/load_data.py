'''
@File    :   load_data.py
@Author  :   Zehong Ma
@Version :   1.0
@Contact :   zehongma@qq.com
@Desc    :   None
'''

import jieba
import pandas as pd

def load_tsv_data(filename):
    """
        加载训练或测试的tsv文件
    """
    reader = pd.read_csv(filename,sep='\t')
    labels = reader['label']
    texts = reader['text_a'] 
    return labels, texts

def load_vocabs(filename):
    """
        加载vocab.txt,返回词典
    """
    with open(filename,'r',encoding='utf-8') as f:
        vocabs = {}
        vocabs_ori = f.readlines()
        index = 0
        for vocab in vocabs_ori:
            vocabs[vocab[:-1]] = index
            index+=1
        return vocabs



if __name__ == '__main__':
    load_tsv_data('./data/train.tsv')
    load_vocabs('./data/vocab.txt')