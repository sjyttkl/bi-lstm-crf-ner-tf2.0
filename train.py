# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     train.py
   email:         songdongdong@weidian.com
   Author :       songdongdong
   date：          2021/12/20 09:39
   Description : train.py
==================================================
"""

from utils import tokenize, build_vocab, read_vocab
import tensorflow as tf
from model import NerModel
import tensorflow_addons as tf_ad
import os
import math
import numpy as np
from args_help import args
from my_log import logger

if not (os.path.exists(args.vocab_file) and os.path.exists(args.tag_file)):
    logger.info("building vocab file")
    build_vocab([args.train_path], args.vocab_file, args.tag_file)
else:
    logger.info("vocab file exits!!")

vocab2id, id2vocab = read_vocab(args.vocab_file)
tag2id, id2tag = read_vocab(args.tag_file)
text_sequences, label_sequences = tokenize(args.train_path, vocab2id, tag2id)

text_len_all = len(text_sequences)
test_len = math.ceil(text_len_all * 0.2)
train_len = text_len_all - test_len

train_sequences = text_sequences[:train_len]
train_label = label_sequences[:train_len]

test_sequences = text_sequences[train_len + 1:]
test_label = label_sequences[train_len + 1:]

# train dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_sequences, train_label))
train_dataset = train_dataset.shuffle(len(train_dataset)).batch(args.batch_size, drop_remainder=True)
# test_dataset
test_dataset = tf.data.Dataset.from_tensor_slices((test_sequences, test_label))
test_dataset = test_dataset.shuffle(len(test_sequences)).batch(args.batch_size, drop_remainder=True)

logger.info("hidden_num:{}, vocab_size:{}, label_size:{}".format(args.hidden_num, len(vocab2id), len(tag2id)))
model = NerModel(hidden_num=args.hidden_num, vocab_size=len(vocab2id), label_size=len(tag2id),
                 embedding_size=args.embedding_size)
optimizer = tf.keras.optimizers.Adam(args.lr)

ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
ckpt.restore(tf.train.latest_checkpoint(args.output_dir))
ckpt_manager = tf.train.CheckpointManager(ckpt,
                                          args.output_dir,
                                          checkpoint_name='model.ckpt',
                                          max_to_keep=3)


# @tf.function
def train_one_step(text_batch, labels_batch):
    with tf.GradientTape() as tape:
        logits, text_lens, log_likelihood = model(text_batch, labels_batch, training=True)
        loss = - tf.reduce_mean(log_likelihood)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, logits, text_lens


def get_acc_one_step(logits, text_lens, labels_batch):
    """
     compute accuracy in  test
    :param logits: model's output
    :param text_lens: true length  in train/test data
    :param labels_batch: label in train/test data
    :return:
    """
    paths = []
    accuracy = 0
    for logit, text_len, labels in zip(logits, text_lens, labels_batch):
        viterbi_path, _ = tf_ad.text.viterbi_decode(logit[:text_len], model.transition_params)
        paths.append(viterbi_path)
        correct_prediction = tf.equal(
            tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences([viterbi_path], padding='post'),
                                 dtype=tf.int32),
            tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences([labels[:text_len]], padding='post'),
                                 dtype=tf.int32)
        )
        accuracy = accuracy + tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # print(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))
    accuracy = accuracy / len(paths)
    return accuracy


best_acc = 0
step = 0
for epoch in range(args.epoch):
    for _, (text_batch, labels_batch) in enumerate(train_dataset):
        step = step + 1
        loss, logits, text_lens = train_one_step(text_batch, labels_batch)
        if step % 20 == 0:
            accuracy = get_acc_one_step(logits, text_lens, labels_batch)
            logger.info('train --- epoch %d, step %d, loss %.4f , accuracy %.4f' % (epoch, step, loss, accuracy))
            # logger.info(" == test====")
            # loss_test, logits_test, text_lens_test = train_one_step(test_dataset, test_label)
            # accuracy_test = get_acc_one_step(logits, text_lens, labels_batch)
            # logger.info('test --- epoch %d, step %d, loss %.4f , accuracy %.4f' % (epoch, step, loss, accuracy))
            if accuracy > best_acc:
                best_acc = accuracy
                ckpt_manager.save()
                logger.info("model saved")

logger.info("finished")
