'''
@File    :   word2vec.py
@Author  :   Zehong Ma
@Version :   1.0
@Contact :   zehongma@qq.com
@Desc    :   None
'''

from gensim import models

def build_Word2Vec(filename):
    words = []
    with open(filename,'r',encoding='utf-8') as f:
        lines = f.read().split('\n')
        for line in lines:
            tokens = line.split(' ')
            words.append(tokens)
    
    # 构建模型
    model = models.Word2Vec(words,min_count=2)

    return model
    

def get_similar_words():
    # 创建模型
    model = build_Word2Vec('./data/input.txt')
    word_list = ['人民','总统','市场','艰辛','企业','美国','研究','技术','航空','聪明']
    for word in word_list:
        result = model.most_similar_cosmul(word)[:5]
        print(word,end=': ')
        for res in result:
            print('"%s", '%res[0],end=' ')
        print()
    return 
            
if __name__== '__main__':
    get_similar_words()

