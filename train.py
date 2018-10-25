import torch
import utils
import numpy as np
import torch.nn.functional as F
import sys
import os
import shutil

def train(train_data_set,dev_data_set,test_data_set,model, config, data):
    sentences = train_data_set[0]
    sentences_index = train_data_set[1]
    labels = train_data_set[2]
    labels_index = train_data_set[3]
    candidates = train_data_set[4]
    candidates_index = train_data_set[5]
    sparse_list = train_data_set[6]
    sparse_index = train_data_set[7]

    test_sentences = test_data_set[0]
    test_sentences_index = test_data_set[1]
    test_labels = test_data_set[2]
    test_labels_index = test_data_set[3]
    test_candidates = test_data_set[4]
    test_candidates_index = test_data_set[5]
    test_sparse_list = test_data_set[6]
    test_sparse_index = test_data_set[7]

    dev_sentences = dev_data_set[0]
    dev_sentences_index = dev_data_set[1]
    dev_labels = dev_data_set[2]
    dev_labels_index = dev_data_set[3]
    dev_candidates = dev_data_set[4]
    dev_candidates_index = dev_data_set[5]
    dev_sparse_list = dev_data_set[6]
    dev_sparse_index = dev_data_set[7]
    print('Start training....')
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(params=parameters, lr=config.learning_rate, momentum=0.9, weight_decay=config.decay)
    steps = 0
    best_score = .0
    for epoch in range(config.maxIters):
        print('第{}轮迭代，共{}次迭代'.format(epoch,config.maxIters))
        model.train()
        train_sentences,train_sentences_index, labels, labels_index, candidates, candidates_index, sparse_list, sparse_index = utils.random_inst(sentences, sentences_index, labels, labels_index, candidates, candidates_index,sparse_list, sparse_index)
        epoch_loss = 0
        train_inputs, train_pairs, train_sparse, train_labels = data.batch_block(config.batch_size, train_sentences, train_sentences_index, labels, labels_index, candidates, candidates_index, sparse_list, sparse_index)
        for index in range(len(train_inputs)):
            batch_length = np.array([np.sum(mask) for mask in train_inputs[index][1]])
            pairs_length = np.array([np.sum(mask) for mask in train_pairs[index][1]])
            sentence_var, sentence_mask_var, pairs_var, pairs_mask_var, sparse_var, labels_var, batch_length_var, pairs_length_var = utils.patch_var(train_inputs[index], train_pairs[index], train_sparse[index],train_labels[index],batch_length.tolist(), pairs_length.tolist())
            model.zero_grad()
            logit = model.forward(sentence_var, sentence_mask_var, pairs_var, pairs_mask_var, sparse_var, batch_length, pairs_length)

            # print(labels_var)
            loss = F.cross_entropy(logit, labels_var)

            loss.backward()
            optimizer.step()
            steps += 1
            if steps % config.log_interval == 0:
                train_size = len(train_inputs)
                corrects = (torch.max(logit, 1)[1].view(labels_var.size()).data == labels_var.data).sum()
                accuracy = float(corrects) / len(sentence_var) * 100.0
                sys.stdout.write(
                    '\rBatch[{}/{}] - loss: {}  acc: {}%({}/{})'.format(steps,
                                                                     train_size,
                                                                     loss.data[0],
                                                                     accuracy,
                                                                     corrects,
                                                                     len(sentence_var)))
            if steps % config.test_interval == 0:
                # dev_score, dev_recall, dev_F = eval(test_sentences, test_sentences_index, test_labels, test_labels_index, test_candidates, test_candidates_index, test_sparse_list, test_sparse_index, model, dictionaries, config)
                dev_score, dev_recall, dev_F = eval(dev_sentences, dev_sentences_index, dev_labels, dev_labels_index, dev_candidates, dev_candidates_index, dev_sparse_list, dev_sparse_index, model, data, config)
                if dev_F > best_score:
                    if not os.path.isdir(config.save_dir):
                        os.makedirs(config.save_dir)
                    else:
                        shutil.rmtree(config.save_dir)
                    best_score = dev_F
                    print('Best Score of Dev is:{}'.format(dev_F))
                    test_score, test_recall, test_F = eval(test_sentences, test_sentences_index, test_labels, test_labels_index, test_candidates, test_candidates_index, test_sparse_list, test_sparse_index, model,data, config)
                    print(steps)
                    print('Score of Test is:{}'.format(test_score))
                    print('Recall of Test is:{}'.format(test_recall))
                    print('F of Test is:{}'.format(test_F))
                    if not os.path.isdir(config.save_dir): os.makedirs(config.save_dir)
                    save_prefix = os.path.join(config.save_dir, 'snapshot')
                    save_path = '{}_{}steps.pt'.format(save_prefix, steps)
                    torch.save(model.state_dict(), save_path)

def eval(sentences, sentences_index, labels, labels_index, candidates, candidates_index, sparse_list, sparse_index, model, data, config):
    model.eval()
    recall_corrects, recall_count, corrects, avg_loss = 0, 0, 0, 0
    # assert (len(data_words) == len(data_tags))
    # assert (len(data_words) == len(data_sparses))
    # assert (len(data_words) == len(data_pairs))

    # for words, pairs, sparses, tags, in zip(data_words, data_pairs, data_sparses, data_tags):
    #     feature = autograd.Variable(torch.LongTensor(words))
    #     pairs = autograd.Variable(torch.LongTensor(pairs))
    #     sparse = autograd.Variable(torch.LongTensor(sparses))
    #     targets = autograd.Variable(torch.LongTensor(tags).squeeze(1))
    #
    #     if args.is_cuda:
    #         feature = feature.cuda()
    #         pairs = pairs.cuda()
    #         sparse = sparse.cuda()
    #         targets = targets.cuda()
    sentences, sentences_index, labels, labels_index, candidates, candidates_index, sparse_list, sparse_index = utils.random_inst(sentences, sentences_index, labels, labels_index, candidates, candidates_index, sparse_list, sparse_index)
    inputs, pairs, sparse, labels = data.batch_block(config.batch_size, sentences, sentences_index, labels,labels_index, candidates, candidates_index,sparse_list, sparse_index)
    for index in range(len(inputs)):
        batch_length = np.array([np.sum(mask) for mask in inputs[index][1]])
        pairs_length = np.array([np.sum(mask) for mask in pairs[index][1]])
        sentence_var, sentence_mask_var, pairs_var, pairs_mask_var, sparse_var, labels_var, batch_length_var, pairs_length_var = utils.patch_var(inputs[index], pairs[index], sparse[index], labels[index], batch_length.tolist(),pairs_length.tolist())
        logit = model.forward(sentence_var, sentence_mask_var, pairs_var, pairs_mask_var, sparse_var, batch_length,pairs_length)
        loss = F.cross_entropy(logit, labels_var, size_average=False)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(labels_var.size()).data == labels_var.data).sum()

        predict_logit = torch.max(logit, 1)[1].view(labels_var.size())

        for index in range(len(labels_var.data)):
            # print('============================')
            # print(labels_var.data[index])
            # print(data.label_alphabet.word2id['0'])
            # print( data.label_alphabet.word2id[str(labels_var.data[index])])
            if data.label_alphabet.word2id[str(labels_var.data[index])] != '0':
            # if dictionaries['id_to_tag'][labels_var.data[index]] != '0':

                recall_count += 1
            if predict_logit.data[index] == labels_var.data[index] and data.label_alphabet.word2id[str(labels_var.data[index])]  != '0':
                recall_corrects += 1

    # size = sum([len(words) for words in sentences])
    size = len(sentences)
    avg_loss = loss.data[0] / size
    accuracy = float(corrects) / size * 100.0
    recall = float(recall_corrects) / recall_count * 100.0
    F_score = float(2 * accuracy * recall) / (accuracy + recall)

    print('\nEvaluation - loss: {}  acc: {}%({}/{}) \n'.format(avg_loss,
                                                               accuracy,
                                                               corrects,
                                                               size))
    print('\nRecall: {}'.format(recall))
    print('\nF_score: {}'.format(F_score))
    return accuracy, recall, F_score




