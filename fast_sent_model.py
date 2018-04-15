#encoding: utf8

import os
import sys
import tensorflow as tf
import numpy as np
import math

class FastSentModel(object):
    """
    旨在对两个句子对，由第一个预测第二个的方式，训练得第一个句子的句向量。
    具体说：第一句的词向量求和得到句向量，然后softmax方式去预测第二句中的每个词。
            把第二句中每个每个词的交叉熵相加，就是笨句对的loss。
    Note: fastsent 模型是中间句子预测左右两边句子。这里略有不同，不过意思一样
    """

    def __init__(self, dict_size, dict_emb_dize, learning_rate):
        # == 模型参数
        self._dict_size     = dict_size + 1   # 模型输入类别数
        self._dict_emb_dize = dict_emb_dize   # 模型词向量维度

        # == 训练参数
        self._begin_lr = learning_rate # 学习率

        # ============= encoder input data
        self.fd_inputs = tf.placeholder(tf.int32, (None, None), name="input_ids")
        # shape == (batch_size, in_seq_len)

        self.fd_inputs_len = tf.placeholder(tf.int32, (None,), name="input_ids_len")
        # shape == (batch_size,)

        # ============= decoder input data, only used in train
        self.fd_decode_inputs = tf.placeholder(tf.int32, (None,None), name="decode_input") 
        # shape == (time_steps, batch_size)

        self.fd_decode_inputs_seq_len = tf.placeholder(tf.int32, (None,), name="decode_input_len") 
        # shape == (batch_size,)

        self.fd_decode_labels = tf.placeholder(tf.int32, (None,), name="decode_labels") 
        # shape == (batch_size,)

        self.func_gen_input_vec()
        self.func_gen_decode_output()
        self.func_gen_decode_output_1()
        self.func_model_init()

    def func_gen_input_vec(self):
        """
        句子的词向量，累加得到句向量
        """

        init_val = tf.truncated_normal(shape=(self._dict_size, self._dict_emb_dize), 
                                      stddev=1. / math.sqrt(self._dict_size))
        embs = tf.Variable(init_val)

        input_embs = tf.nn.embedding_lookup(embs, self.fd_inputs)
        # input_embs.shape = (batch_size, input_seq_len, emb_size)

        inputs_max_seq_len = tf.shape(self.fd_inputs)[-1]
        # batch内不同句子的长度不一样，所以需要mask，使得按长度正确计算
        mask = tf.sequence_mask(self.fd_inputs_len, maxlen=inputs_max_seq_len)
        # mask.shape = (batch_size, max_seq_len)
        mask = tf.cast(mask, tf.float32)
        mask = tf.expand_dims(mask, -1)
        # mask.shape = (batch_size, input_seq_len, 1)

        inputs_vec = tf.reduce_sum(input_embs*mask, 1)
        # inputs_vec.shape = (batch, emb_size)

        logits, _   = self.FC("fc4softmax", inputs_vec, self._dict_size, act_fun=None)
        # logits.shape == (batch_size, dict_size)
        self._logits = logits

    def FC(self, name, input, dim_out, act_fun=tf.nn.relu, re_use=False):
        with tf.variable_scope("scope_%s" % (name)) as scope:
            if re_use:
                scope.reuse_variables()
            dim_in = input.get_shape().as_list()[-1]
            if act_fun == tf.nn.relu:
                stddev = 1. / math.sqrt(dim_in/2)
            else:
                stddev = math.sqrt(2. / (dim_in + dim_out))
            W = tf.get_variable("w_%s" % (name), shape=(dim_in, dim_out), \
                        initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
            B = tf.get_variable("b_%s" % (name), shape=(dim_out,), \
                        initializer=tf.zeros_initializer(tf.float32))
            if act_fun:
                return act_fun(tf.matmul(input, W) + B), (W, B)
            else:
                return tf.matmul(input, W) + B, (W, B)

    def func_gen_decode_output(self):
        """
        由第一句的句向量，得到预测第二句概率的损失函数
        这里是把第二句所有词放在一个batch里，一次训练。
        另一种方式是，按seq1->seq2_word1, seq1->seq2_word2方式，把第二句按word 拆开分解
                      成多个sample，然后按batch训练的时候，同一句的不同word可能会垮batch
                      self.func_gen_decode_output_1(...)对应此方式
        看了下github上别人的标准fastsent模型的实现，是按第二种方式做的。我只试了本方式的，最后看效果还可以
        """
        softmax_val = -tf.nn.log_softmax(self._logits, -1)
        # softmax_val.shape = (batch_size, dict_size)

        indices = self.fd_decode_inputs #[[0, 1, 2], [2,1,-1]]
        depth = self._dict_size
        indices_one_hot = tf.one_hot(indices, depth)
        # indices_one_hot.shape = (batch_size, decode_input_len, dict_size)
        indices_mask = tf.reduce_sum(indices_one_hot, -2)
        # indices_mask.shape = (batch_size, dict_size)

        # 算一次log_softmax，然后把句子中所有词的值累加, 柔在一个batch的loss里
        loss = tf.reduce_sum(indices_mask * softmax_val, -1)
        # loss.shape = (batch_size,)
        loss = loss / tf.cast(self.fd_decode_inputs_seq_len, tf.float32)
        # loss.shape = (batch_size,)

        self.out_loss = tf.reduce_mean(loss)

    def func_gen_decode_output_1(self):
        """
        详细说明见：self.func_gen_decode_output(..)
        """
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.fd_decode_labels, 
                                                              logits=self._logits)
        self.out_loss_1 = tf.reduce_mean(loss)

    def func_model_init(self):
        self._fd_lr_rate = tf.Variable(self._begin_lr, trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=self._fd_lr_rate)
        # 下面如果loss换成self.out_loss_1，则是另一种方式来训练
        opt = optimizer.minimize(self.out_loss)
        self.opt = opt
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

        self._model_saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
    
    # =============================

    def set_lr(self, cur_lr):
        """ set learning rate """
        self.sess.run(tf.assign(self._fd_lr_rate, cur_lr))

    def save_model(self, path):
        """ save model """
        self._model_saver.save(self.sess, path)

    def restore_model(self, path):
        """ restore model """
        self._model_saver.restore(self.sess, path)

    def gen_train_feed_dict(self, encode_inputs, encode_inputs_seq_len, 
                              decode_inputs, decode_inputs_seq_len):
        """
        得到训练时需要的feed_dict 
        for self.func_gen_decode_output(..)
        """
        feed_data = {
            self.fd_inputs:     encode_inputs,         # shape == (batch_size, in_seq_len)
            self.fd_inputs_len: encode_inputs_seq_len, # shape == (batch_size,)
            self.fd_decode_inputs: decode_inputs,      # shape == (batch_size, max_len)
            self.fd_decode_inputs_seq_len: decode_inputs_seq_len, # shape == (batch_size,)
            }
        return feed_data

    def gen_train_feed_dict_1(self, encode_inputs, encode_inputs_seq_len, decode_inputs):
        """
        得到训练时需要的feed_dict 
        for self.func_gen_decode_output_1(..)
        """
        feed_data = {
            self.fd_inputs:        encode_inputs,         # shape == (batch_size, in_seq_len)
            self.fd_inputs_len:    encode_inputs_seq_len, # shape == (batch_size,)
            self.fd_decode_labels: decode_inputs,         # shape == (batch_size,)
            }
        return feed_data

if __name__ == "__main__":
    model = FastSentModel(dict_size     = 100000,
                          dict_emb_dize = 128,
                          learning_rate = 0.01)

    model.restore_model("./model_fast/fast_1160000")

    from gen_pinyin_and_chineseChar_sample import read_dict

    m_hz = {}
    m_hz_rev = {}
    read_dict("./sim1/segword_freq.dat", 200000, m_hz, m_hz_rev)

    # 导出词向量
    xx = tf.trainable_variables()
    for x in xx:
        #out = model.sess.run(tf.transpose(x))
        out = model.sess.run(x)
        print "---"
        print x, out.shape
        #if "w_fc4softmax" in x.name:
        if "Variable" in x.name:
            print out[0][:5]
        if "Variable" in x.name:
            fp = open("fast_sent.emb", "w")
            fp.write("%d %d\n" % (len(out - 4), 128))
            for i in xrange(len(out)):
                if i <= 3:
                    continue
                key = m_hz[i]
                fp.write("%s " % (key))
                fp.write(" ".join(["%.6f" % x for x in out[i]]))
                fp.write("\n")
                if i % 10000 == 0:
                    print i

#/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
