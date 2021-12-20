# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     utils.py
   email:         songdongdong@weidian.com
   Author :       songdongdong
   date：          2021/12/20 09:45
   Description : utils.py
==================================================
"""

import tensorflow as tf
import json,os

def build_vocab(corpus_file_list, vocab_file, tag_file):
    """
    构建 vocab 词典
    :param corpus_file_list: 训练文本
    :param vocab_file: 词典文件
    :param tag_file: tag文件
    :return:
    """
    words = set()
    tags = set()
    for file in corpus_file_list:
        # words = words.union(set([line.strip().split()[0]  for line in open(file, "r", encoding='utf-8').readlines()]))
        # tags = tags.union(set([line.strip().split()[-1] for line in open(file, "r", encoding='utf-8').readlines()]))
        for line in open(file, "r", encoding='utf-8').readlines():
            line = line.strip()
            if line == "end":
                continue
            try:
                w,t = line.split()
                words.add(w)
                tags.add(t)
            except Exception as e:
                print(line.split())
                # raise e

    if not os.path.exists(vocab_file):
        with open(vocab_file,"w",encoding="Utf-8") as f:
            for index,word in enumerate(["<UKN>"]+list(words) ):
                f.write(word+"\n")

    tag_sort = {
        "O": 0,
        "B": 1,
        "I": 2,
        "E": 3,
    }

    tags = sorted(list(tags), #B-LOC
           key=lambda x: (len(x.split("-")), x.split("-")[-1], tag_sort.get(x.split("-")[0], 100))
           )
    if not os.path.exists(tag_file):
        with open(tag_file,"w") as f:
            for index,tag in enumerate(["<UKN>"]+tags):
                f.write(tag+"\n")

# build_vocab(["./data/train.utf8","./data/test.utf8"])


def read_vocab(vocab_file):
    """
    read vocab文件
    :param vocab_file: input 文件
    :return: 返回 vocab2id, id2vocab
    """
    vocab2id = {}
    id2vocab = {}
    for index,line in enumerate([line.strip() for line in open(vocab_file,"r",encoding="utf-8").readlines()]):
        vocab2id[line] = index
        id2vocab[index] = line
    return vocab2id, id2vocab

# print(read_vocab("./data/tags.txt"))



def tokenize(filename,vocab2id,tag2id):
    """
     use list format to load train data
    :param filename: train data
    :param vocab2id:  vocabid
    :param tag2id: tagid
    :return:
    """
    contents = []
    labels = []
    content = []
    label = []
    with open(filename, 'r', encoding='utf-8') as fr:
        for line in [elem.strip() for elem in fr.readlines()][:500000]:
            try:
                if line != "end":
                    w,t = line.split()
                    content.append(vocab2id.get(w,0)) # default id is 0
                    label.append(tag2id.get(t,0))
                else:
                    if content and label:
                        contents.append(content) # use list format
                        labels.append(label) # use list format
                    content = []
                    label = []
            except Exception as e:
                content = []
                label = []
    # use max sentence length padding post
    contents = tf.keras.preprocessing.sequence.pad_sequences(contents, padding='post')
    labels = tf.keras.preprocessing.sequence.pad_sequences(labels, padding='post')
    return contents,labels






tag_check = {
    "I":["B","I"],
    "E":["B","I"],
}


def check_label(front_label,follow_label):
    if not follow_label:
        raise Exception("follow label should not both None")

    if not front_label:
        return True

    if follow_label.startswith("B-"):
        return False

    if (follow_label.startswith("I-") or follow_label.startswith("E-")) and \
        front_label.endswith(follow_label.split("-")[1]) and \
        front_label.split("-")[0] in tag_check[follow_label.split("-")[0]]:
        return True
    return False


def format_result(chars, tags):
    entities = []
    entity = []
    for index, (char, tag) in enumerate(zip(chars, tags)):
        entity_continue = check_label(tags[index - 1] if index > 0 else None, tag)
        if not entity_continue and entity:
            entities.append(entity)
            entity = []
        entity.append([index, char, tag, entity_continue])
    if entity:
        entities.append(entity)

    entities_result = []
    for entity in entities:
        if entity[0][2].startswith("B-"):
            entities_result.append(
                {"begin": entity[0][0] + 1,
                 "end": entity[-1][0] + 1,
                 "words": "".join([char for _, char, _, _ in entity]),
                 "type": entity[0][2].split("-")[1]
                 }
            )

    return entities_result



if __name__ == "__main__":
    text = ['国','家','发','展','计','划','委','员','会','副','主','任','王','春','正']
    tags =  ['B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'E-ORG', 'O', 'O', 'O', 'B-PER', 'I-PER', 'E-PER']
    entities_result= format_result(text,tags)
    print(json.dumps(entities_result, indent=4, ensure_ascii=False))

