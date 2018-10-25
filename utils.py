import re
import random
import itertools
from inst import *
import numpy as np
import collections
from torch.autograd import Variable
import torch

def read_file(path):
    with open(path,encoding='utf-8') as f:
        sentence_label_list = []
        for line in f.readlines():
            sentence = Sentence_label()
            flag = 1
            sentence.ori = line
            ####sentence all word
            for ele in line:
                if ele == "<":
                    flag = 0
                elif ele == ">":
                    flag = 1
                    continue
                if flag == 1:
                    sentence.words += ele
            ##############

            ####sentence all label
            sentence_label = re.findall(r"<(.*?)>", line)
            for i in sentence_label:
                label = "<" + i + ">"
                sentence.all_label.append(label)
            #######################

            ####sentence entity label
            sentence_entity_label = re.findall(r"<(.?[a]+[0-9])>", line)
            for i in sentence_entity_label:
                label = "<" + i + ">"
                sentence.entity_label.append(label)
            ############################

            ####sentence eval label
            sentence_eval_label = re.findall(r"<(.?[e]+[0-9]+.*?)>", line)
            for i in sentence_eval_label:
                label = "<" + i + ">"
                sentence.eval_label.append(label)
            ############################


            #####sentence explain label
            for i in sentence.all_label:
                if i not in sentence.entity_label and i not in sentence.eval_label:
                    sentence.explain_label.append(i)
            ################################

            sentence_label_list.append(sentence)

        return sentence_label_list


def sentence_extract(sentence_label_list):
    sentence_word_list = []
    for i in sentence_label_list:
        sentence = Sentence_word()
        sentence.ori = i.ori
        sentence.words = i.words
        start = 0
        end = 0
        while end < len(i.entity_label) and start <len(i.entity_label):
            mark = Mark()
            mark.start = i.ori.find(i.entity_label[start])
            end = start + 1
            mark.end = i.ori.find(i.entity_label[end])
            mark.word = i.ori[i.ori.find(i.entity_label[start]) + len(i.entity_label[start]):i.ori.find(i.entity_label[end])]
            mark.label = i.entity_label[start]
            sentence.entity.append(mark)
            start = end + 1

        start = 0
        end = 0
        while end < len(i.eval_label) and start <len(i.eval_label):
            mark = Mark()
            mark.start = i.ori.find(i.eval_label[start])
            end = start + 1
            mark.end = i.ori.find(i.eval_label[end])
            mark.word = i.ori[i.ori.find(i.eval_label[start]) + len(i.eval_label[start]):i.ori.find(i.eval_label[end])]
            mark.sentiment = i.eval_label[start][-2]
            mark.label = i.eval_label[start]
            sentence.eval.append(mark)
            start = end + 1

        start = 0
        end = 0
        while end < len(i.explain_label) and start <len(i.explain_label):
            mark = Mark()
            mark.start = i.ori.find(i.explain_label[start])
            end = start + 1
            mark.end = i.ori.find(i.explain_label[end])
            mark.word = i.ori[i.ori.find(i.explain_label[start]) + len(i.explain_label[start]):i.ori.find(i.explain_label[end])]
            mark.sentiment = i.explain_label[start][-2]
            mark.label = i.explain_label[start]
            sentence.explain.append(mark)
            start = end + 1
        sentence_word_list.append(sentence)
    return sentence_word_list

def exact_feat(sentence_word_list):
    feat_list = []
    candidate_category = []
    for i in sentence_word_list:
        feat = []
        category =[]
        if len(i.entity)> 0 and len(i.eval) > 0 and len(i.explain) > 0:
            for idx in range(len(i.entity)):
                for idy in range(len(i.eval)):
                    list = []
                    list.append(i.entity[idx])
                    list.append(i.eval[idy])
                    if i.eval[idy].label.find(i.entity[idx].label.replace('<', '').replace('>', '')) != -1:
                        category.append('1')
                    else:
                        category.append('0')
                    feat.append(list)
            for idx in range(len(i.entity)):
                for idy in range(len(i.explain)):
                    list =[]
                    list.append(i.entity[idx])
                    list.append(i.explain[idy])
                    if i.explain[idy].label.find(i.entity[idx].label.replace('<', '').replace('>', '')) != -1:
                        category.append('2')
                    else:
                        category.append('0')
                    feat.append(list)
            for idx in range(len(i.eval)):
                for idy in range(len(i.explain)):
                    list = []
                    list.append(i.eval[idx])
                    list.append(i.explain[idy])
                    if i.explain[idy].label.find(i.eval[idx].label.replace('<', '').replace('>', '')[0:2]) != -1:
                        category.append('3')
                    else:
                        category.append('0')
                    feat.append(list)
        elif len(i.entity) > 0 and len(i.eval) > 0 and len(i.explain) == 0:
            for idx in range(len(i.entity)):
                for idy in range(len(i.eval)):
                    list = []
                    list.append(i.entity[idx])
                    list.append(i.eval[idy])
                    if i.eval[idy].label.find(i.entity[idx].label.replace('<', '').replace('>', '')) != -1:
                        category.append('1')
                    else:
                        category.append('0')
                    feat.append(list)

        elif len(i.entity) > 0 and len(i.eval) == 0 and len(i.explain) > 0:
            for idx in range(len(i.entity)):
                for idy in range(len(i.explain)):
                    list =[]
                    list.append(i.entity[idx])
                    list.append(i.explain[idy])
                    if i.explain[idy].label.find(i.entity[idx].label.replace('<', '').replace('>', '')) != -1:
                        category.append('2')
                    else:
                        category.append('0')
                    feat.append(list)

        elif len(i.entity) == 0 and len(i.eval) > 0 and len(i.explain) > 0:
            for idx in range(len(i.eval)):
                for idy in range(len(i.explain)):
                    list = []
                    list.append(i.eval[idx])
                    list.append(i.explain[idy])
                    if i.explain[idy].label.find(i.eval[idx].label.replace('<', '').replace('>', '')[0:2]) != -1:
                        category.append('3')
                    else:
                        category.append('0')
                    feat.append(list)
        else:
            continue
        feat_list.append(feat)
        candidate_category.append(category)
    return feat_list, candidate_category

def cal_distence_feat(feat_list):
    distence_list = []
    for sentence in feat_list:
        sentence_entity_distence = []
        for idx in range(len(sentence)):
            distence = abs(sentence[idx][1].start - sentence[idx][0].end)
            sentence_entity_distence.append(distence)
        distence_list.append(sentence_entity_distence)
    return distence_list



def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/root_sum_square


def load_pretrained_emb_uniform(path, text_field_words_dict, emb_dims, norm=False, set_padding=False):
    padID = text_field_words_dict['<pad>']
    embed_dict, embed_dim = load_pretrained_emb_total(path)
    assert embed_dim == emb_dims
    alphabet_size = len(text_field_words_dict)
    pretrain_emb_size = len(embed_dict)
    # print('The number of words is ' + str(alphabet_size))
    print('The dim of pretrained embedding is ' + str(embed_dim) + '\n')

    pretrain_emb = np.zeros([alphabet_size, embed_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    scale = np.sqrt(3.0 / embed_dim)
    for index, word in enumerate(text_field_words_dict.keys()):
        if word in embed_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embed_dict[word])
            else:
                pretrain_emb[index,:] = embed_dict[word]
            perfect_match += 1
        elif word.lower() in embed_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embed_dict[word.lower()])
            else:
                pretrain_emb[index,:] = embed_dict[word.lower()]
            case_match += 1
        else:
            if set_padding is False or index != padID:
                pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embed_dim])
            not_match += 1
    print("Embedding:\n  pretrain word:%s, alphabet word:%s, prefect match:%s, case match:%s, oov:%s, oov%%:%s"%(pretrain_emb_size, alphabet_size, perfect_match, case_match, not_match, (not_match+0.)/alphabet_size))
    return pretrain_emb


def load_pretrained_emb_avg(path, text_field_words_dict, emb_dims, norm=False, set_padding=True):
    print('Load embedding...')
    padID = text_field_words_dict['<pad>']
    embed_dict, embed_dim = load_pretrained_emb_total(path)
    assert embed_dim == emb_dims
    alphabet_size = len(text_field_words_dict)
    pretrain_emb_size = len(embed_dict)
    print('The dim of pretrained embedding is ' + str(embed_dim) + '\n')

    pretrain_emb = np.zeros([alphabet_size, embed_dim])
    perfect_match = []
    case_match = []
    not_match = []

    for index, word in enumerate(text_field_words_dict.keys()):
        if word in embed_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embed_dict[word])
            else:
                pretrain_emb[index,:] = embed_dict[word]
            perfect_match.append(index)
        elif word.lower() in embed_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embed_dict[word.lower()])
            else:
                pretrain_emb[index,:] = embed_dict[word.lower()]
            case_match.append(index)
        else:
            not_match.append(index)

    sum_col = np.sum(pretrain_emb, axis=0) / (len(perfect_match)+len(case_match))  # 按列求和，再求平均
    for i in not_match:
        if i != padID or set_padding is False:
            pretrain_emb[i] = sum_col
    print("Embedding:\n  pretrain word:%s, alphabet word:%s, prefect match:%s, case match:%s, oov:%s, oov%%:%s"%(pretrain_emb_size, alphabet_size, len(perfect_match), len(case_match), len(not_match), (len(not_match)+0.)/alphabet_size))
    return pretrain_emb


def load_pretrained_emb_zeros(path, text_field_words_dict, emb_dims, norm=False, set_padding=False):
    # padID = text_field_words_dict['<pad>']
    embed_dict, embed_dim = load_pretrained_emb_total(path)
    assert embed_dim == emb_dims
    alphabet_size = len(text_field_words_dict)
    pretrain_emb_size = len(embed_dict)
    # print('The number of words is ' + str(alphabet_size))
    print('The dim of pretrained embedding is ' + str(embed_dim) + '\n')

    pretrain_emb = np.zeros([alphabet_size, embed_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for index, word in enumerate(text_field_words_dict.keys()):
        if word in embed_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embed_dict[word])
            else:
                pretrain_emb[index, :] = embed_dict[word]
            perfect_match += 1
        elif word.lower() in embed_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embed_dict[word.lower()])
            else:
                pretrain_emb[index, :] = embed_dict[word.lower()]
            case_match += 1
        else:
            not_match += 1
    print("Embedding:\n  pretrain word:%s, alphabet word:%s, prefect match:%s, case match:%s, oov:%s, oov%%:%s" % (
    pretrain_emb_size, alphabet_size, perfect_match, case_match, not_match, (not_match + 0.) / alphabet_size))
    return pretrain_emb


def load_pretrained_emb_total(path):
    embed_dim = -1
    embed_dict = collections.OrderedDict()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line_split = line.strip().split(' ')
            if len(line_split) < 3: continue
            if embed_dim < 0: embed_dim = len(line_split) - 1
            else:
                # print('111',embed_dim)
                # print(len(line_split) - 1)
                # print(line)
                assert (embed_dim == len(line_split) - 1)


            embed = np.zeros([1, embed_dim])        # 不直接赋值，也许考虑到python浅赋值的问题
            embed[:] = line_split[1:]
            embed_dict[line_split[0]] = embed
    return embed_dict, embed_dim

def random_inst(inst_list,inst_list_index,labels, labels_index,candidates, candidates_index,sparse_list, sparse_index):
    assert len(inst_list) == len(candidates) ==len(labels)
    insts_num = list(range(0,len(inst_list)))
    random.shuffle(insts_num)

    insts_dict = dict(zip(insts_num,inst_list))
    insts_dict = sorted(insts_dict.items(), key=lambda item: item[0], reverse=False)
    insts_sorted = [ele[1] for ele in insts_dict]

    insts_index_dict = dict(zip(insts_num, inst_list_index))
    insts_index_dict = sorted(insts_index_dict.items(), key=lambda item: item[0], reverse=False)
    insts_index_sorted = [ele[1] for ele in insts_index_dict]

    labels_dict = dict(zip(insts_num,labels))
    labels_dict = sorted(labels_dict.items(), key=lambda item: item[0], reverse=False)
    labels_sorted = [ele[1] for ele in labels_dict]

    labels_index_dict = dict(zip(insts_num,labels_index))
    labels_index_dict = sorted(labels_index_dict.items(), key=lambda item: item[0], reverse=False)
    labels_index_sorted = [ele[1] for ele in labels_index_dict]

    candidates_dict = dict(zip(insts_num, candidates))
    candidates_dict = sorted(candidates_dict.items(), key=lambda item: item[0], reverse=False)
    candidates_sorted = [ele[1] for ele in candidates_dict]

    candidates_index_dict = dict(zip(insts_num, candidates_index))
    candidates_index_dict = sorted(candidates_index_dict.items(), key=lambda item: item[0], reverse=False)
    candidates_index_sorted = [ele[1] for ele in candidates_index_dict]

    sparse_dict = dict(zip(insts_num, sparse_list))
    sparse_dict = sorted(sparse_dict.items(), key=lambda item: item[0], reverse=False)
    sparse_sorted = [ele[1] for ele in sparse_dict]

    sparse_index_dict = dict(zip(insts_num, sparse_index))
    sparse_index_dict = sorted(sparse_index_dict.items(), key=lambda item: item[0], reverse=False)
    sparse_index_sorted = [ele[1] for ele in sparse_index_dict]

    return insts_sorted, insts_index_sorted, labels_sorted, labels_index_sorted, candidates_sorted, candidates_index_sorted, sparse_sorted, sparse_index_sorted



def sorted_instances(batch_sentence,batch_sentences_index, batch_labels, batch_labels_index, batch_candidates,batch_candidates_index, batch_sparse, batch_sparse_index):
    insts_length = [len(inst_index) for inst_index in batch_sentences_index]
    insts_num = list(range(len(batch_sentences_index)))
    assert len(insts_length) == len(insts_num)
    length_dict = dict(zip(insts_num, insts_length))
    length_sorted = sorted(length_dict.items(), key=lambda e: e[1], reverse=True)
    perm_list = [length_sorted[i][0] for i in range(len(length_sorted))]

    insts_dict = dict(zip(insts_num, batch_sentence))
    insts_sorted = [insts_dict.get(i) for i in perm_list]
    insts_index_dict = dict(zip(insts_num, batch_sentences_index))
    insts_index_sorted = [insts_index_dict.get(i) for i in perm_list]

    labels_dict = dict(zip(insts_num, batch_labels))
    labels_sorted = [labels_dict.get(i) for i in perm_list]
    labels_index_dict = dict(zip(insts_num, batch_labels_index))
    labels_index_sorted = [labels_index_dict.get(i) for i in perm_list]

    # candidates_dict = dict(zip(insts_num, batch_candidates))
    # candidates_sorted = [candidates_dict.get(i) for i in perm_list]
    # candidates_index_dict = dict(zip(insts_num, batch_candidates_index))
    # candidates_index_sorted = [candidates_index_dict.get(i) for i in perm_list]

    #################candidates单独处理
    candidates_length = [len(candidates_index) for candidates_index in batch_candidates_index]
    candidates_num = list(range(len(batch_candidates_index)))
    assert len(candidates_length) == len(candidates_num)
    candidates_length_dict = dict(zip(candidates_num, candidates_length))
    candidates_length_sorted = sorted(candidates_length_dict.items(), key=lambda e: e[1], reverse=True)
    candidates_list = [candidates_length_sorted[i][0] for i in range(len(candidates_length_sorted))]

    candidates_dict = dict(zip(candidates_num, batch_candidates))
    candidates_sorted = [candidates_dict.get(i) for i in candidates_list]
    candidates_index_dict = dict(zip(candidates_num, batch_candidates_index))
    candidates_index_sorted = [candidates_index_dict.get(i) for i in candidates_list]
    #################################

    sparse_dict = dict(zip(insts_num, batch_sparse))
    sparse_sorted = [sparse_dict.get(i) for i in perm_list]
    sparse_index_dict = dict(zip(insts_num, batch_sparse_index))
    sparse_index_sorted = [sparse_index_dict.get(i) for i in perm_list]

    return insts_sorted, insts_index_sorted, labels_sorted, labels_index_sorted, candidates_sorted, candidates_index_sorted, sparse_sorted,sparse_index_sorted

def patch_var(train_inputs, train_pairs, train_sparse,train_labels,batch_length, pairs_length):
    sentence_var = Variable(torch.LongTensor(train_inputs[0]))
    sentence_mask_var = Variable(torch.ByteTensor(train_inputs[1]))

    pairs_var = Variable(torch.LongTensor(train_pairs[0]))
    labels_var = Variable(torch.LongTensor(train_labels))

    pairs_mask_var = Variable(torch.ByteTensor(train_pairs[1]))

    sparse_var = Variable(torch.LongTensor(train_sparse))

    batch_length_var = Variable(torch.LongTensor(batch_length))
    pairs_length_var = Variable(torch.LongTensor(pairs_length))

    return sentence_var,sentence_mask_var,pairs_var,pairs_mask_var,sparse_var,labels_var,batch_length_var,pairs_length_var























