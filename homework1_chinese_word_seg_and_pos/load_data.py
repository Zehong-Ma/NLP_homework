'''
@File    :   load_data.py
@Author  :   Zehong Ma
@Version :   1.0
@Contact :   zehongma@qq.com
@Desc    :   Load people daily data  
'''
import os

def load(filepath):
    file_list = os.listdir(filepath)
    sentences = ""
    words_index = []
    dict_index2type = {}
    dict_index2word = {}
    start = 0
    for filename in file_list:
        sentence = ""
        with open(filepath+filename,'r',encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                elements = line.split(' ')
                for element in elements:
                    if element == '\n' or element == '':
                        continue
                    try:
                        word,word_type = element.split('/')
                        if word[0]=='[':
                            word = word[1:]
                    except:
                        word,word_type,composed_type = element.split('/')
                        word_type = word_type[:-1]
                    sentence+=word

                    end = start + len(word)
                    index = str(start)+str(end-1)
                    words_index.append(index)
                    start = end
                    dict_index2word[index] = word
                    dict_index2type[index] = word_type
            sentences+=sentence     
    return sentences, words_index, dict_index2type

if __name__ == '__main__':
    load('./people-2014/test/0123/')
