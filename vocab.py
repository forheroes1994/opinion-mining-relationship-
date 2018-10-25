import utils
from inst import *
import math

class Data():
    def __init__(self):

        self.word_alphabet = alphabet('word')
        self.label_alphabet = alphabet('label',is_label=True)
        self.sparse_alphabet = alphabet('char')
        self.word_num = 0
        self.label_num = 0
        self.sparse_num = 0
        self.number_normalized = True
        self.norm_word_emb = False
        self.pretrain_word_embedding = None

    def sentence_index(self,path):
        dataset = []
        with open(path, encoding='utf-8')as f:
            sentence_list = []
            label_list = []
            candidate_list = []
            sparse_list = []
            for line in f.readlines():
                sentence_sparse = []
                sentence_candidate = []
                line =line.encode('utf-8').decode('utf-8-sig')
                sentence_list.append(line.split('\t')[0])
                label_list.append(line.split('\t')[3])
                sentence_candidate.append(line.split('\t')[1])
                sentence_candidate.append(line.split('\t')[2])
                candidate_list.append(sentence_candidate)
                sentence_sparse.append(line.split('\t')[4])
                if line.split('\t')[5] == '':
                    sentence_sparse.append('<unk>')
                else:
                    sentence_sparse.append(line.split('\t')[5])
                sentence_sparse.append(line.strip('\n').split('\t')[6])
                sparse_list.append(sentence_sparse)


            if not self.word_alphabet.fix_flag:
                self.build_alphabet(sentence_list,label_list,sparse_list)
                print("重新建字典")

            sentence_index = []
            label_index = []
            candidate_index = []
            sparse_index = []
            for idx in range(len(sentence_list)):
                words_index = [self.word_alphabet.get_index(w) for w in sentence_list[idx]]
                label_index.append(self.label_alphabet.get_index(label_list[idx]))
                sentence_index.append(words_index)
            for idx in range(len(candidate_list)):
                sentence_candidate_index = []
                for w in candidate_list[idx][0]:
                    sentence_candidate_index.append(self.word_alphabet.get_index(w))
                for w in candidate_list[idx][1]:
                    sentence_candidate_index.append(self.word_alphabet.get_index(w))
                # first_candidate_index = [self.word_alphabet.get_index(w) for w in candidate_list[idx][0]]
                # second_candidate_index = [self.word_alphabet.get_index(w) for w in candidate_list[idx][1]]
                # sentence_candidate_index.append(first_candidate_index)
                # sentence_candidate_index.append(second_candidate_index)
                candidate_index.append(sentence_candidate_index)
            for idx in range(len(sparse_list)):
                sentence_sparse_index = []
                distence_feature = self.sparse_alphabet.get_index(sparse_list[idx][0])
                sentiment_feature_1 = self.sparse_alphabet.get_index(sparse_list[idx][1])
                sentiment_feature_2 = self.sparse_alphabet.get_index(sparse_list[idx][2])
                sentence_sparse_index.append(distence_feature)
                sentence_sparse_index.append(sentiment_feature_1)
                sentence_sparse_index.append(sentiment_feature_2)
                sparse_index.append(sentence_sparse_index)
            dataset.append(sentence_list)
            dataset.append(sentence_index)
            dataset.append(label_list)
            dataset.append(label_index)
            dataset.append(candidate_list)
            dataset.append(candidate_index)
            dataset.append(sparse_list)
            dataset.append(sparse_index)
            return dataset

    def build_alphabet(self,sentence_list,label_list,sparse_list):
        assert len(sentence_list) == len(label_list)
        for idx in range(len(sentence_list)):
            for i in sentence_list[idx]:
                self.word_alphabet.add(i)
        for idx in range(len(label_list)):
            self.label_alphabet.add(label_list[idx])
        for idx in range(len(sparse_list)):
            for i in sparse_list[idx]:
                self.sparse_alphabet.add(i)



    def fix_alphabet(self):
        self.word_num = self.word_alphabet.close()
        self.label_num = self.label_alphabet.close()
        self.sparse_num = self.sparse_alphabet.close()
        ####label alphabet ['<pad>','1','2','0','3']

    def load_pretrained_emb_uniform(self,embedding_path, embed_dim):
        self.pretrain_word_embedding = utils.load_pretrained_emb_uniform(embedding_path,self.word_alphabet.word2id,embed_dim)

    def batch_block(self,batch_size, sentences, sentences_index, labels, labels_index, candidates, candidates_index, sparse_list, sparse_index):
        batch_sentence = []
        batch_sentences_index = []
        batch_labels = []
        batch_labels_index = []
        batch_candidates = []
        batch_candidates_index = []
        batch_sparse = []
        batch_sparse_index = []

        batch_num = int(math.ceil(len(sentences_index) / batch_size))
        sentences_index_set = [[[], []] for _ in range(batch_num)]
        candidates_index_set = [[[], []] for _ in range(batch_num)]
        sparse_index_set = [[ ] for _ in range(batch_num)]
        labels_index_set = [[ ] for _ in range(batch_num)]
        for id, inst_index in enumerate(sentences_index):
            idx = id // batch_size
            if id == 0 or id % batch_size != 0:
                batch_sentence.append(sentences[id])
                batch_sentences_index.append(inst_index)

                batch_labels.append(labels[id])
                batch_labels_index.append(labels_index[id])

                batch_candidates.append(candidates[id])
                batch_candidates_index.append(candidates_index[id])

                batch_sparse.append(sparse_list[id])
                batch_sparse_index.append(sparse_index[id])
            elif id % batch_size == 0:
                assert len(batch_sentences_index) == batch_size
                insts_sorted, insts_index_sorted, labels_sorted, labels_index_sorted, candidates_sorted, candidates_index_sorted, sparse_sorted, sparse_index_sorted = utils.sorted_instances(batch_sentence,batch_sentences_index,batch_labels, batch_labels_index, batch_candidates, batch_candidates_index, batch_sparse, batch_sparse_index)
                max_candidates_length = max([len(list_word) for list_word in candidates_index_sorted])
                max_sentence_length = len(insts_sorted[0])
                for idy in range(batch_size):
                    cur_sentence_length = len(insts_sorted[idy])
                    cur_candidate_length = len(candidates_index_sorted[idy])
                    sentences_index_set[idx-1][0].append(insts_index_sorted[idy] + [self.word_alphabet.word2id['<pad>']] * (max_sentence_length - cur_sentence_length))
                    sentences_index_set[idx-1][1].append([1] * cur_sentence_length + [0] * (max_sentence_length - cur_sentence_length))
                    candidates_index_set[idx-1][0].append(candidates_index_sorted[idy] + [self.word_alphabet.word2id['<pad>']] * (max_candidates_length - cur_candidate_length))
                    candidates_index_set[idx-1][1].append([1] * cur_candidate_length + [0] * (max_candidates_length - cur_candidate_length))
                    sparse_index_set[idx-1].append(sparse_index_sorted[idy])

                    labels_index_set[idx-1].append(labels_index_sorted[idy])
                batch_sentence = []
                batch_sentences_index = []
                batch_labels = []
                batch_labels_index = []
                batch_candidates = []
                batch_candidates_index = []
                batch_sparse = []
                batch_sparse_index = []

                batch_sentence.append(sentences[id])
                batch_sentences_index.append(inst_index)

                batch_labels.append(labels[id])
                batch_labels_index.append(labels_index[id])

                batch_candidates.append(candidates[id])
                batch_candidates_index.append(candidates_index[id])

                batch_sparse.append(sparse_list[id])
                batch_sparse_index.append(sparse_index[id])

        if batch_sentences_index != []:
            insts_sorted, insts_index_sorted, labels_sorted, labels_index_sorted, candidates_sorted, candidates_index_sorted, sparse_sorted, sparse_index_sorted = utils.sorted_instances(batch_sentence, batch_sentences_index, batch_labels, batch_labels_index, batch_candidates,batch_candidates_index, batch_sparse, batch_sparse_index)
            max_candidates_length = max([len(list_word) for list_word in candidates_index_sorted])
            max_sentence_length = len(insts_sorted[0])
            for idy in range(len(insts_sorted)):
                cur_sentence_length = len(insts_sorted[idy])
                cur_candidate_length = len(candidates_index_sorted[idy])
                sentences_index_set[batch_num-1][0].append(insts_index_sorted[idy] + [self.word_alphabet.word2id['<pad>']] * (max_sentence_length - cur_sentence_length))
                sentences_index_set[batch_num-1][1].append([1] * cur_sentence_length + [0] * (max_sentence_length - cur_sentence_length))
                candidates_index_set[batch_num-1][0].append(candidates_index_sorted[idy] + [self.word_alphabet.word2id['<pad>']] * (max_candidates_length - cur_candidate_length))
                candidates_index_set[batch_num-1][1].append([1] * cur_candidate_length + [0] * (max_candidates_length - cur_candidate_length))
                sparse_index_set[batch_num-1].append(sparse_index_sorted[idy])

                labels_index_set[batch_num-1].append(labels_index_sorted[idy])
        return sentences_index_set,candidates_index_set,sparse_index_set,labels_index_set




