import utils
from vocab import *
import argparse
import config
import lstm
import train
if __name__ == '__main__':

    data = Data()
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config-file', default=r'..\opining mining\config.cfg')
    argparser.add_argument('--static', default=False, help='fix the embedding')
    argparser.add_argument('--use-cuda', default=False)

    args = argparser.parse_args()
    config = config.Configurable(args.config_file)
    data.static = args.static
    data.use_cuda = args.use_cuda

    #############train data
    train_sentence_label_list = utils.read_file(config.train_file)
    train_sentence_word_list = utils.sentence_extract(train_sentence_label_list)

    train_feat,train_candidate_category = utils.exact_feat(train_sentence_word_list)
    train_distence_feat = utils.cal_distence_feat(train_feat)


    # file = open('train_data.txt', 'w',encoding='utf-8')
    # for idx in range(len(train_feat)):
    #     for idy in range(len(train_feat[idx])):
    #         file.write(str(train_sentence_word_list[idx].words.strip('\n')) + '\t' + str(train_feat[idx][idy][0].word) + '\t' + str(train_feat[idx][idy][1].word) + '\t' + str(train_candidate_category[idx][idy]) + '\t' + str(train_distence_feat[idx][idy]) + '\t' + str(train_feat[idx][idy][0].sentiment) + '\t' + str(train_feat[idx][idy][1].sentiment) + '\n')
    # file.close()

    #############dev data
    dev_sentence_label_list = utils.read_file(config.dev_file)
    dev_sentence_word_list = utils.sentence_extract(dev_sentence_label_list)
    dev_feat,dev_candidate_category = utils.exact_feat(dev_sentence_word_list)
    dev_distence_feat = utils.cal_distence_feat(dev_feat)

    # file = open('dev_data.txt', 'w',encoding='utf-8')
    # for idx in range(len(dev_feat)):
    #     for idy in range(len(dev_feat[idx])):
    #         file.write(str(dev_sentence_word_list[idx].words.strip('\n')) + '\t' + str(dev_feat[idx][idy][0].word) + '\t' + str(dev_feat[idx][idy][1].word) + '\t' + str(dev_candidate_category[idx][idy]) + '\t' + str(dev_distence_feat[idx][idy]) + '\t' + str(dev_feat[idx][idy][0].sentiment) + '\t' + str(dev_feat[idx][idy][1].sentiment) + '\n')
    # file.close()
    #####################

    #############test data
    test_sentence_label_list = utils.read_file(config.test_file)
    test_sentence_word_list = utils.sentence_extract(test_sentence_label_list)
    test_feat,test_candidate_category = utils.exact_feat(test_sentence_word_list)
    test_distence_feat = utils.cal_distence_feat(test_feat)

    # file = open('test_data.txt', 'w',encoding='utf-8')
    # for idx in range(len(test_feat)):
    #     for idy in range(len(test_feat[idx])):
    #         file.write(str(test_sentence_word_list[idx].words.strip('\n')) + '\t' + str(test_feat[idx][idy][0].word) + '\t' + str(test_feat[idx][idy][1].word) + '\t' + str(test_candidate_category[idx][idy]) + '\t' + str(test_distence_feat[idx][idy]) + '\t' + str(test_feat[idx][idy][0].sentiment) + '\t' + str(test_feat[idx][idy][1].sentiment) + '\n')
    # file.close()
    ######################

    # train_sentences, train_sentences_index, train_labels, train_labels_index, train_candidates, train_candidates_index, train_sparse_list, train_sparse_index = data.sentence_index('./train_data.txt')
    # test_sentences, test_sentences_index, test_labels, test_labels_index, test_candidates, test_candidates_index, test_sparse_list, test_sparse_index = data.sentence_index('./test_data.txt')
    train_data_set = data.sentence_index('./train_data.txt')
    if not args.static:
        data.fix_alphabet()
    test_data_set = data.sentence_index('./test_data.txt')
    dev_data_set = data.sentence_index('./dev_data.txt')



    if config.pretrained_wordEmb_file is not'':
        data.load_pretrained_emb_uniform(config.pretrained_wordEmb_file, config.word_dims)

    model = lstm.LSTM(config, data)

    train.train(train_data_set,dev_data_set,test_data_set, model, config, data)

















