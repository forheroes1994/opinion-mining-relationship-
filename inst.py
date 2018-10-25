from collections import OrderedDict
class Sentence_label(object):
    def __init__(self):
        self.ori = ''
        self.words = ''
        self.all_label = []
        self.entity_label = []
        self.eval_label = []
        self.explain_label = []

class Sentence_word(object):
    def __init__(self):
        self.ori = ''
        self.words = ''
        self.entity = []
        self.eval = []
        self.explain = []

class Mark(object):
    def __init__(self):
        self.start = 0
        self.end = 0
        self.word = ''
        self.sentiment = ''
        self.label = ''


class alphabet:
    def __init__(self, word, is_label=False, fix_flag=False):
        self.name = word
        self.id2word = []
        self.word2id = OrderedDict()

        self.is_label = is_label
        self.fix_flag = fix_flag

        self.UNKNOWN = '<unk>'
        self.PAD = '<pad>'
        if not self.is_label: self.add(self.UNKNOWN)
        if not self.is_label: self.add(self.PAD)
        if self.name =='':self.add(self.UNKNOWN)




    def add(self, e):
        if e not in self.id2word:
            self.word2id[e] = self.size()
            self.id2word.append(e)


    def size(self):
        return len(self.id2word)

    def get_index(self, word):
        try:
            return self.word2id[word]
        except KeyError:  # keyerror一般是使用字典里不存在的key产生的错误，避免产生这种错误
            if not self.fix_flag:
                # print('WARNING:Alphabet get_index, unknown instance, add new instance.')
                self.add(word)
                return self.word2id[word]
            else:
                # print('WARNING:Alphabet get_index, unknown instance, return unknown index.')
                return self.word2id[self.UNKNOWN]

    def close(self):
        self.fix_flag = True
        # alphabet_size = self.size()
        return self.size()             #####


