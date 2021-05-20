'''
@File    :   evaluation.py
@Author  :   Zehong Ma
@Version :   1.0
@Contact :   zehongma@qq.com
@Desc    :   evaluation on test set: precision, recall, F1
'''
import load_data
import jieba.posseg as pseg
def evaluate(sentences, words_index, type_dict):
    words_seged = pseg.cut(sentences)
    words_seged_index = []
    seged_type_dict = {}
    start=0
    for word, flag in words_seged:
        end = start + len(word)
        index = str(start)+str(end-1)
        words_seged_index.append(index)
        start = end
        seged_type_dict[index] = flag
        # print(word,flag)

    # compute criteria for segmentation
    words_seged_index = set(words_seged_index)
    words_index = set(words_index)
    precision_seg = len(words_index&words_seged_index)/len(words_seged_index)
    recall_seg = len(words_index&words_seged_index)/len(words_index)
    F1_seg = 2*precision_seg*recall_seg/(precision_seg+recall_seg) 
    print("###### seg evaluation ######")
    print("precision: ",precision_seg)
    print("recall: ",recall_seg)
    print("F1: ",F1_seg)

    # compute criteria for pos tagging
    words_candidate = words_index&words_seged_index
    count = 0 
    dict_predict = {}
    dict_truth = {}
    dict_true_positive = {}
    for word_index in words_candidate:

        if type_dict[word_index] in dict_truth.keys():
            dict_truth[type_dict[word_index]] += 1
        else:
            dict_truth[type_dict[word_index]] = 1
        
        if seged_type_dict[word_index] in dict_predict.keys():
            dict_predict[seged_type_dict[word_index]] += 1
        else:
            dict_predict[seged_type_dict[word_index]] = 1
        
        if type_dict[word_index] == seged_type_dict[word_index]:
            count+=1
            if seged_type_dict[word_index] in dict_true_positive.keys():
                dict_true_positive[seged_type_dict[word_index]] += 1
            else:
                dict_true_positive[seged_type_dict[word_index]] = 1
    

    tp = 0
    predict_res = 0
    ground_truth = 0
    print("##### single type pos tagging evaluation #####")
    print("type, precision, recall,  F1 ")
    for predict_type in dict_true_positive.keys():
        precision_single_type = dict_true_positive[predict_type]/dict_predict[predict_type]
        recall_single_type = dict_true_positive[predict_type]/dict_truth[predict_type]
        F1_single_type = 2*precision_single_type*recall_single_type/(precision_single_type+recall_single_type)
        print(" %s:    %.3f,   %.3f,   %.3f"%(predict_type,precision_single_type,recall_single_type,F1_single_type))
        tp += dict_true_positive[predict_type]
        predict_res += dict_predict[predict_type]
        ground_truth += dict_truth[predict_type]
    avg_precision_postag = tp/predict_res
    avg_recall_postag = tp/ground_truth
    F1_postag = 2*avg_precision_postag*avg_recall_postag/(avg_precision_postag+avg_recall_postag)
    print("###### pos taging evaluation ######")
    print("precision: ",avg_precision_postag)
    print("recall: ",avg_recall_postag)
    print("F1: ",F1_postag)

    """print count of each type """
    # comm = {}
    # for key in sorted(dict_true_positive):
    #     comm[key] = dict_truth[key]
    # print(sorted(comm.items(), key = lambda kv:(kv[1], kv[0])))

    return
    
sentences, words_index, dict_index2type = load_data.load('./people-2014/test/0123/')
evaluate(sentences, words_index, dict_index2type)
 