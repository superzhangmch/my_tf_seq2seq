#encoding: utf8

import os
import sys
import tensorflow as tf
import numpy as np
import math

class Seq2SeqModel(object):

    def __init__(self, \
                 seq1_dict_size, seq1_dict_emb_dize, \
                 seq2_dict_size, seq2_dict_emb_dize, \
                 rnn_layer_cnt, rnn_hidden_size, out_hid_size, beam_size,\
                 learning_rate, use_attention=True, EOS=2, SOS=1):
        # == 模型参数
        self._dict_size       = seq1_dict_size     # 模型输入类别数
        self._dict_emb_dize   = seq1_dict_emb_dize # 模型词向量维度
        self._dict_size_1     = seq2_dict_size     # 模型输入类别数
        self._dict_emb_dize_1 = seq2_dict_emb_dize # 模型词向量维度

        self._rnn_layer_cnt = rnn_layer_cnt   # 模型多层RNN层数
        self._rnn_hid_size  = rnn_hidden_size # 模型 lstm 隐层大小
        self._out_hid_size  = out_hid_size    # 最终输出前的隐层大小
        self._beam_size     = beam_size       # beam search size
        self.EOS = EOS
        self.SOS = SOS
        self.use_attn = use_attention

        # == 训练参数
        self._begin_lr = learning_rate # 学习率

        # ============= encoder input data
        self.fd_encode_inputs = tf.placeholder(tf.int32, (None, None), name="encode_input") 
        # shape == (batch_size, in_seq_len)

        self.fd_encode_inputs_seq_len = tf.placeholder(tf.int32, (None,), name="encode_input_len") 
        # shape == (batch_size,)

        # ============= decoder input data, only used in train
        self.fd_decode_inputs = tf.placeholder(tf.int32, (None,None), name="decode_input") 
        # shape == (time_steps, batch_size)

        self.fd_decode_inputs_seq_len = tf.placeholder(tf.int32, (None,), name="decode_input_len") 
        # shape == (batch_size,)

        self.fd_labels = tf.placeholder(tf.int32, (None, None), name="decode_labels") 
        # shape == (batch_size, in_seq_len)

        self.encode_cell, self.encode_cell_init = self.func_gen_encode_cell()
        self.decode_cell                        = self.func_gen_decode_cell()

        self.encode_seq_output, self.encode_seq_final_state = self.func_gen_encode_output()
        self.func_gen_decode_output()

        self.func_decode_beam_search()

        self.func_model_init()

    def func_gen_encode_cell(self):
        """ rnn cell 
        可以说这里只是声明了 RNN cell，并没有创建cell。所以所谓的参数与网络结构的共享与否的
        各种check都还没发生.  实际上, BasicLSTMCell, MultiRNNCell, EmbeddingWrapper等
        的__init__(..)还没有创建相应的网络结构, 调用__call__(...)才会创建。
        为了处理共享独享所需的scope，需要放在执行__call__(..)的地方, 作为__call__参数传进去。
        如果返回的cell直接塞给了tf.nn.dynamic_rnn, 则是dynamic_rnn内部调用了相应的__call__(..),
        因此如果需要独享一个RNN结果，需要给dynamic_rnn一个独立的scope, 而非这里
        """
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(
                self._rnn_hid_size, forget_bias=1.0, state_is_tuple=True)

        #def dropout_cell():
        #    return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=0.6)

        rnn_cell = lstm_cell
        cell = tf.contrib.rnn.MultiRNNCell([rnn_cell() for _ in range(self._rnn_layer_cnt)])

        w_init = tf.truncated_normal_initializer(stddev=1. / math.sqrt(self._dict_size))
        rnn_cell = tf.contrib.rnn.EmbeddingWrapper(cell, embedding_classes=self._dict_size,
                                embedding_size=self._dict_emb_dize,
                                initializer=w_init)

        batch_size = tf.shape(self.fd_encode_inputs)[0]
        rnn_init_state = rnn_cell.zero_state(batch_size, tf.float32)
        return rnn_cell, rnn_init_state

    def func_gen_decode_cell(self):
        """ rnn cell """
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(
                self._rnn_hid_size, forget_bias=1.0, state_is_tuple=True)

        #def dropout_cell():
        #    return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=0.6)

        rnn_cell = lstm_cell
        cell = tf.contrib.rnn.MultiRNNCell([rnn_cell() for _ in range(self._rnn_layer_cnt)])

        w_init = tf.truncated_normal_initializer(stddev=1. / math.sqrt(self._dict_size_1))
        rnn_cell = tf.contrib.rnn.EmbeddingWrapper(cell, embedding_classes=self._dict_size_1,
                                embedding_size=self._dict_emb_dize_1,
                                initializer=w_init)

        return rnn_cell

    def func_gen_encode_output(self):
        with tf.variable_scope("scope_encode_rnn") as scope:
            # 希望encode独享完整的RNN结构，所以需要指定variable_scope
            # emb_inputs.shape: (batch_size, in_seq_len) => (batch_size, in_seq_len, 1)
            emb_inputs = tf.expand_dims(self.fd_encode_inputs, -1)

            # == rnn outputs
            rnn_outputs, final_rnn_state = tf.nn.dynamic_rnn(cell=self.encode_cell,
                                               inputs=emb_inputs, dtype="float",
                                               sequence_length=self.fd_encode_inputs_seq_len,
                                               initial_state=self.encode_cell_init)
            # rnn_outputs.shape == (batch_size, in_seq_len, rnn_hid_size)
            # type(final_rnn_state) == (LSTMStateTuple.{c,h}, LSTMStateTuple.{c,h})
            # final_rnn_state.c.shape == (batch_size, rnn_hid_size)
            return rnn_outputs, final_rnn_state

    def gen_attention_context(self, decode_1step_out, encode_seq_output, seq_len):
        """
        encode_seq_output.shape == (batch_size, in_seq_len, rnn_hid_size)
        decode_1step_out.shape  == (batch_size, rnn_hid_size)
        seq_len.shape == (batch_size,)
        这里实现点乘形式的 attention, 还有其他形式的attention
        """
        # encode_seq_output = tf.constant([[[1.1, 2.1, 3.1], [0.91, 1.92, 1.93], [1.5, 2.2, 1.], [0.7, 0.71, 0.73]],
        #                                  [[1., 1.1, 1.3],  [3., 3.1, 3.3], [.5, 0.51, 0.53],  [0.91, 0.92, 0.94]]])
        # decode_1step_out = tf.constant([[3., 1., 4.], [5., 9., 2.]])
        # seq_len = tf.constant([2, 3])

        encode_seq_output_shape = tf.shape(encode_seq_output)
        #batch_size   = encode_seq_output_shape[0] # batch size
        seq_max_len  = encode_seq_output_shape[1] # in a batch, multi seq are padded to the same seq len)
        #rnn_hid_size = encode_seq_output_shape[2] # rnn hidden size

        # == 1. calc dot product
        # reshape decode_1step_out to [batch_size, 1, rnn_hid_size]
        #decode_1step_out_1 = tf.reshape(decode_1step_out, [-1, 1, rnn_hid_size])
        # expand_dims is more convenient
        decode_1step_out_1 = tf.expand_dims(decode_1step_out, 1)

        # now we can do point-wise product within each batch
        # note: pointwise_product.shape == encode_seq_output.shape == (batch_size, in_seq_len, rnn_hid_size)
        pointwise_product = decode_1step_out_1 * encode_seq_output

        dot_product = tf.reduce_sum(pointwise_product, axis=2)
        # dot_product.shape == [batch_size, in_seq_len]

        # == 2. calc mask for each seq in the same batch
        mask = tf.sequence_mask(seq_len, maxlen=seq_max_len)
        mask = tf.cast(mask, tf.float32) #得到0,1表示的mask矩阵
        out_len_mask = 1 - mask #得到0,1表示的mask矩阵
        # mask.shape == [batch_size, in_seq_len]
        # mask = [[1, 1, .., 1, 0, .. 0], [..], .. ]

        # == 3. 得到对应encode_seq_output每个向量的 softmax 权重
        # 超过长度的，令取值tf.float32.min, 如此tf.exp(.) == 0., 算出的softmax自然忽略超长度的
        dot_product_masked = dot_product * mask + out_len_mask * tf.float32.min
        # dot_product_masked.shape == [batch_size, in_seq_len]

        attn_weight = tf.nn.softmax(dot_product_masked)
        # attn_weight.shape == [batch_size, in_seq_len]

        # == 4. 把 encode_seq_output 内的向量，加权求和，得到最终结果
        attn_weight_1 = tf.expand_dims(attn_weight, -1)
        # 下面做的是把权重w和相应的encode 向量相乘
        #   attn_weight_1.shape == [batch_size, in_seq_len, 1]
        #   encode_seq_output.shape == (batch_size, in_seq_len, rnn_hid_size)
        #   所以可以相乘
        attn_context_0 = encode_seq_output * attn_weight_1
        # attn_context_0.shape == (batch_size, rnn_hid_size, in_seq_len)
        # 下面做的是把乘权后的encode 向量加起来
        attn_context_1 = tf.reduce_sum(attn_context_0, 1)
        # attn_context_1.shape = (batch_size, rnn_hid_size)

        #sess = tf.Session()
        #print "pointwise_product", sess.run(pointwise_product)
        #print "dot_product", sess.run(dot_product)
        #print "mask", sess.run(mask)
        #print "out_len_mask", sess.run(out_len_mask)
        #print "xx1", sess.run( encode_seq_output)
        #print "dot_product_masked", sess.run(dot_product_masked)
        #print 'attn_weight', sess.run(attn_weight)
        #print 'attn_weight_1', sess.run(attn_weight_1)
        #print 'encode_seq_output_1', sess.run(encode_seq_output)
        #print "att_0", sess.run(attn_context_0)
        #print "att_1", sess.run(attn_context_1)

        return attn_context_1

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
        基于 tf.while_loop 的序列生成。不用它发现生成很慢
        """
        decode_inputs_max_seq_len = tf.shape(self.fd_decode_inputs)[0]

        idx = tf.constant(0)
        attn_dim = self._rnn_hid_size * 2 if self.use_attn else self._rnn_hid_size
        # whileLoop 也可以传递python list 类型数据，从而把rnn序列结果收集起来。
        #   这里是用的tensor逐个拼接方式
        batch_size = tf.shape(self.fd_encode_inputs)[0]
        z = tf.Variable(0, tf.int32)
        decode_rnn_out = tf.zeros((z, batch_size, attn_dim))

        def cond(idx, cell_state, decode_rnn_out):
            return idx < decode_inputs_max_seq_len

        def body(idx, cell_state, decode_rnn_out):
            with tf.variable_scope("rnn_decode") as scope:
                # 希望 decoder 所需要的RNN结构与encoder不共享(Note: seq2seq一般不共享), 
                # 所以这里指定和encoder处不一样的 variable_scope
                # decoder的网络结构是通过self.decode_cell.__call__(..., scope=scope)构建出来的
                cur_out, cur_state = self.decode_cell(self.fd_decode_inputs[idx], 
                                                                cell_state, scope=scope)
                # cur_out.shape == (batch_size, self._rnn_hid_size)
            context_vec = self.gen_attention_context(cur_out, self.encode_seq_output, self.fd_encode_inputs_seq_len)
            # context_vec.shape == (batch_size, self._rnn_hid_size)

            out = tf.concat([context_vec, cur_out], 1) if self.use_attn else cur_out
            # out.shape == (batch_size, self._rnn_hid_size*2)

            rnn_out = tf.concat([decode_rnn_out, tf.expand_dims(out, 0)], axis=0)
            return idx+1, cur_state, rnn_out

        #print self.encode_seq_final_state, len(self.encode_seq_final_state), 'xxxxxxxxxxx'
        # self.encode_seq_final_state format:
        # list of LSTMStateTuple(c=<tf.Tensor (?, 128) float32>, h=<tf.Tensor (?, 128) float32>)
        # with count equal to rnn layer cnt
        loop_vars=[idx, self.encode_seq_final_state, decode_rnn_out]
        decode_rnn_loop = tf.while_loop(cond, body, loop_vars=loop_vars)

        decode_loop_out = tf.transpose(decode_rnn_loop[2], (1, 0, 2))
        # decode_loop_out.shape == (batch_size, time_steps, rnn_hid_size*2)

        fc_in = tf.reshape(decode_loop_out, [-1, attn_dim])
        attn_out, _   = self.FC("fc1", fc_in, self._out_hid_size, tf.nn.relu)
        attn_out_1, _ = self.FC("fc2", attn_out, self._dict_size_1)

        logits_out = tf.reshape(attn_out_1, (-1, decode_inputs_max_seq_len, self._dict_size_1))
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.fd_labels, logits=logits_out)

        mask = tf.sequence_mask(self.fd_decode_inputs_seq_len, maxlen=decode_inputs_max_seq_len)
        mask = tf.cast(mask, tf.float32)
        loss = tf.reshape(loss, tf.shape(mask))
        loss = loss * mask
        self.out_loss_ori = loss
        self.out_decode_prob = tf.reduce_sum(loss, -1)
        loss = tf.reduce_sum(loss) / tf.cast(tf.reduce_sum(self.fd_decode_inputs_seq_len), tf.float32)
        self.out_loss = loss

    def func_decode_beam_search(self):
        """
        beam search decoder
        """

        beam_size = self._beam_size
        attn_dim = self._rnn_hid_size * 2 if self.use_attn else self._rnn_hid_size

        self.fd_seqgen_max_len = tf.placeholder(tf.int32)

        idx = tf.constant(0)
        z = tf.Variable(0, tf.int32)

        out_last_idx    = tf.zeros((z, beam_size), dtype=tf.int32)
        out_ids         = tf.zeros((z, beam_size), dtype=tf.int32)
        out_probs       = tf.zeros((z, beam_size), dtype=tf.float32)
        out_total_probs = tf.zeros((z, beam_size), dtype=tf.float32)

        def cond(idx, cell_state, in_ids,
                 out_last_idx, out_ids, out_probs, out_total_probs, can_break):
            # 当beam 宽度内解出的已经全结束后，就可以令can_break=1，从而退出循环了
            # 如果一直没有全end of sentence, 则循环够最大长度后退出
            return tf.logical_and(idx < self.fd_seqgen_max_len, tf.equal(can_break, 0))
            #return idx < self.fd_seqgen_max_len

        def body(idx, cell_state, in_ids,
                 out_last_idx, out_ids, out_probs, out_total_probs, can_break):
            with tf.variable_scope("rnn_decode") as scope:
                # 希望 decoder 所需要的RNN结构与encoder不共享(Note: seq2seq一般不共享), 
                # 所以这里指定和encoder处不一样的 variable_scope
                # 另外需要和 func_gen_decode_output() 处实现参数共享，所以下面用 reuse_variables()
                tf.get_variable_scope().reuse_variables()
                cur_out, cur_state = self.decode_cell(in_ids, cell_state, scope=scope)
                # cur_out.shape == (batch_size, self._rnn_hid_size)

            bt_sz = tf.shape(in_ids)[0]
            encode_seq_output = tf.tile(tf.expand_dims(self.encode_seq_output[0], 0), [bt_sz, 1, 1])
            encode_inputs_seq_len = tf.tile(tf.expand_dims(self.fd_encode_inputs_seq_len[0], 0), [bt_sz])
            context_vec = self.gen_attention_context(cur_out, encode_seq_output, encode_inputs_seq_len)
            # context_vec.shape == (batch_size, self._rnn_hid_size)

            out = tf.concat([context_vec, cur_out], 1) if self.use_attn else cur_out
            # out.shape == (batch_size, self._rnn_hid_size*2)

            fc_in = out
            attn_out, _   = self.FC("fc1", fc_in, self._out_hid_size, tf.nn.relu, re_use=True)
            attn_out_1, _ = self.FC("fc2", attn_out, self._dict_size_1, re_use=True)

            out_log_softmax = tf.nn.log_softmax(attn_out_1)
            # above.shape == (batch_size, dict_size)
            out_top_k = tf.nn.top_k(out_log_softmax, beam_size)
            # above.format: [values=(batch_size, beam_size), idx=(..)]

            def beam_choose(cur_state, top_k_prob, top_k_idx,
                            out_last_idx, out_ids, out_probs, out_total_probs):

                last_out_ids         = []
                last_out_total_probs = []
                last_last_idx        = []
                last_out_probs       = []

                if len(out_ids) > 0:
                    last_out_ids         = out_ids[-1]
                    last_out_total_probs = out_total_probs[-1]
                    last_last_idx        = out_last_idx[-1]
                    last_out_probs       = out_probs[-1]

                arr = []
                for batch_i in xrange(len(top_k_idx)):
                    last_idx = batch_i
                    last_total_prob = 0. if len(last_out_total_probs) == 0 else last_out_total_probs[last_idx]
                    for word_j in xrange(len(top_k_idx[batch_i])):
                        this_prob = top_k_prob[batch_i][word_j]
                        this_id   = top_k_idx [batch_i][word_j]
                        prob_till_now = this_prob + last_total_prob
                        if this_id == self.EOS:
                            arr.append([last_idx, this_id, this_prob, prob_till_now, -1])
                        else:
                            arr.append([last_idx, this_id, this_prob, prob_till_now, batch_i])

                if len(last_out_ids) > 0:
                    for i in xrange(len(top_k_idx), len(last_out_ids), 1):
                        #arr.append([last_last_idx[i], last_out_ids[i], last_out_probs[i], 
                        arr.append([i, last_out_ids[i], last_out_probs[i], 
                                    last_out_total_probs[i], -2])
                arr = sorted(arr, key=lambda x: x[3], reverse=True)
                arr_1 = arr[:beam_size]

                arr_not_finished = [a for a in arr_1 if a[-1] >= 0]
                arr_finished = [a for a in arr_1 if a[-1] < 0]

                arr_finally_choosed = arr_not_finished + arr_finished
                #if len(out_ids) == 6:
                #    print "xxxxxxxxxxxxxxxxxxxxxxxxx", len(out_ids)
                #    print arr_not_finished, "not finish"
                #    print arr_finished, "finished"
                #    print arr_finally_choosed, "all"
                #    print arr, "arr"
                #    print arr_1, "arr_1"
                #    print out_last_idx, out_ids, out_probs, out_total_probs, "vvv"
                cur_last_idx    = [a[0] for a in arr_finally_choosed]
                cur_out_ids     = [a[1] for a in arr_finally_choosed]
                cur_out_probs   = [a[2] for a in arr_finally_choosed]
                cur_total_probs = [a[3] for a in arr_finally_choosed]
                can_break = 0
                if len(arr_not_finished) == 0:
                    can_break = 1

                next_ids = [a[1] for a in arr_not_finished]
                next_state_idx = [a[-1] for a in arr_not_finished]
                next_state = cur_state[:,:,next_state_idx,:]

                return np.float32(next_state), np.int32(next_ids), \
                       np.int32(cur_last_idx), np.int32(cur_out_ids), \
                       np.float32(cur_out_probs), np.float32(cur_total_probs), np.int32(can_break)

            next_state, next_ids, \
                        cur_last_idx, cur_out_ids, cur_out_probs, cur_total_probs, can_break_1\
                        = tf.py_func(beam_choose, 
                                     [cur_state, out_top_k[0], out_top_k[1], 
                                      out_last_idx, out_ids, out_probs, out_total_probs],
                                     [tf.float32, tf.int32,
                                      tf.int32, tf.int32, tf.float32, tf.float32, tf.int32])

            out_last_idx_1    = tf.concat([out_last_idx,    tf.expand_dims(cur_last_idx,    0)], axis=0)
            out_ids_1         = tf.concat([out_ids,         tf.expand_dims(cur_out_ids,     0)], axis=0)
            out_probs_1       = tf.concat([out_probs,       tf.expand_dims(cur_out_probs,   0)], axis=0)
            out_total_probs_1 = tf.concat([out_total_probs, tf.expand_dims(cur_total_probs, 0)], axis=0)

            next_state_1 = tf.reshape(next_state, (len(cell_state) * 2, -1, self._rnn_hid_size))
            ns = tf.unstack(next_state_1)
            next_state_transed = tuple([tf.contrib.rnn.LSTMStateTuple(ns[i], ns[i+1]) for i in xrange(0, len(ns), 2)])

            next_ids.set_shape((None,))
            can_break_1.set_shape(())
            return idx+1, next_state_transed, next_ids, \
                          out_last_idx_1, out_ids_1, out_probs_1, out_total_probs_1, can_break_1

        # while_loop 内部此处对应的参数是变成的，所以只好用placeholder使得初始输入也是变成的
        self.fd_SOS_start = tf.placeholder(tf.int32, (None,), name="SOS_start")

        loop_vars=[idx, self.encode_seq_final_state, 
                   self.fd_SOS_start, 
                   out_last_idx, out_ids, out_probs, out_total_probs, 0]

        ab = self.encode_seq_final_state
        encode_last_state_shape = tuple([tf.contrib.rnn.LSTMStateTuple(ab[i][0].get_shape(), \
                                                                       ab[i][1].get_shape()) \
                                                                       for i in xrange(len(ab))])

        decode_rnn_loop = tf.while_loop(cond, body, loop_vars=loop_vars
                        ,shape_invariants=[idx.get_shape(), 
                                          encode_last_state_shape,
                                          tf.TensorShape([None]),
                                          out_last_idx.get_shape(), out_ids.get_shape(),
                                          out_probs.get_shape(), out_total_probs.get_shape(),
                                          idx.get_shape(),
                                         ]
                        )
        _, _, _, out_last_idx, out_ids, out_probs, out_total_probs, _ = decode_rnn_loop
        self.beam_search_out = [out_last_idx, out_ids, out_probs, out_total_probs]

    def func_model_init(self):
        self._fd_lr_rate = tf.Variable(self._begin_lr, trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=self._fd_lr_rate)
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
                              decode_inputs, decode_inputs_seq_len, decode_labels):
        """
        得到训练时需要的feed_dict 
        """
        feed_data = {
            self.fd_encode_inputs:         encode_inputs,         # shape == (batch_size, in_seq_len)
            self.fd_encode_inputs_seq_len: encode_inputs_seq_len, # shape == (batch_size,)
            self.fd_decode_inputs:         np.array(decode_inputs).transpose(),  
                                                                  # shape == (time_steps, batch_size)
            self.fd_decode_inputs_seq_len: decode_inputs_seq_len, # shape == (batch_size,)
            self.fd_labels:                decode_labels,         # shape == (batch_size, in_seq_len)
            }
        return feed_data

    def gen_decode_prob_feed_dict(self, in_seq, out_seq):
        EOS = self.EOS
        SOS = self.SOS

        in_seq_len = [len(in_seq)] * len(out_seq)
        in_seq_1 = [in_seq] * len(out_seq)
        out_len = [len(s) + 1 for s in out_seq]

        max_len = max(out_len)
        out_seq_1 = [([SOS] + s + ([0] * (max_len - 1 - len(s)))) for s in out_seq]
        out_seq_2 = [(s + [EOS] + ([0] * (max_len - 1 - len(s)))) for s in out_seq]
        return self.gen_train_feed_dict(in_seq_1, in_seq_len, out_seq_1, out_len, out_seq_2)

    def gen_beam_search_feed_dict(self, encode_inputs, decode_max_len=20):
        """
        得到训练时需要的feed_dict 
        """
        feed_data = {
            self.fd_encode_inputs:         [encode_inputs],
            self.fd_encode_inputs_seq_len: [len(encode_inputs)],
            self.fd_SOS_start: [1],
            self.fd_seqgen_max_len: decode_max_len
            }
        return feed_data

    def _find_beam_path(self, out_last_idx, out_ids, out_probs, out_total_probs, topK_idx, length_idx):
        def find_path(out_last_idx, out_ids, out_probs, out_total_probs, topK_idx, length_idx, out_arr):
            arr = [ out_ids[length_idx][topK_idx], 
                    out_last_idx[length_idx][topK_idx],
                    out_probs[length_idx][topK_idx],
                    out_total_probs[length_idx][topK_idx]
                    ]
            out_arr.append(arr)
            if length_idx == 0:
                return
            topK_idx = out_last_idx[length_idx][topK_idx]
            length_idx -= 1
            find_path(out_last_idx, out_ids, out_probs, out_total_probs, topK_idx, length_idx, out_arr)
        out_arr = []
        find_path(out_last_idx, out_ids, out_probs, out_total_probs, topK_idx, length_idx, out_arr)
        out_arr.reverse()
        out = []
        for a in out_arr:
            out.append(a)
            if a[0] == self.EOS:
                break
        return out

    def find_beam_sarch_path(self, out_last_idx, out_ids, out_probs, out_total_probs):
        beam_res = []
        for i in xrange(len(out_ids[-1])):
            out_arr = self._find_beam_path(out_last_idx, out_ids, out_probs, out_total_probs, i, len(out_ids) - 1)
            beam_res.append(out_arr)
        return beam_res

    def print_beam_sarch_result(self, in_seq, expect_seq, m_dict, beam_ori_res, add_space=False):
        out_last_idx, out_ids, out_probs, out_total_probs = beam_ori_res
        beam_res = self.find_beam_sarch_path(out_last_idx, out_ids, out_probs, out_total_probs)
        sep = "" if not add_space else " "
        print "Input:", sep.join([m_dict[s] for s in in_seq]), in_seq
        if expect_seq:
            print "Expect:", sep.join([m_dict[s] for s in expect_seq]), expect_seq
        for i in xrange(len(beam_res)):
            b = beam_res[i]
            score = b[-1][-1]
            b = b[:-1] # 跳过末尾的EOS
            try:
                print "%d\t%.3f\t%s" % (i, score, sep.join([m_dict[bb[0]] for bb in b]))
            except:
                print "%d\t%.3f\t%s" % (i, score, str(b))

    def seq2seq_gen(self, in_seq, expect_seq, m_dict, add_space=False):
        beam_res = self.sess.run(self.beam_search_out, 
                              feed_dict=self.gen_beam_search_feed_dict(in_seq))
    
        self.print_beam_sarch_result(in_seq, expect_seq, m_dict, beam_res, add_space)


if __name__ == "__main__":
    model = Seq2SeqModel(seq1_dict_size       = 3000,
                         seq1_dict_emb_dize   = 64,
                         seq2_dict_size       = 3000,
                         seq2_dict_emb_dize   = 64,
                         rnn_layer_cnt   = 1,
                         rnn_hidden_size = 64,
                         out_hid_size    = 64,
                         beam_size       = 5,
                         learning_rate   = 0.001)


    model.restore_model("./model/seq2seq")

    #seq = [22,4,7,9,11]
    #model.seq2seq_gen(seq)

    while True:
        print ">",
        input = raw_input().strip()
        if not input:
            continue
        if input.lower() in ["exit", "quit"]:
            break
        arr = input.split(" ")
        arr = [a for a in arr if a.strip()]
        if len(arr) == 1:
            in_seq, _ = gen_sample_from_txt(input, "gbk")
            model.seq2seq_gen(in_seq, None, m_dict)
        else:
            in_seq_1, _ = gen_sample_from_txt(arr[0], "gbk")
            in_seq_2 = []
            for i in xrange(len(arr[1:])):
                in_seq, _ = gen_sample_from_txt(arr[i+1], "gbk")
                in_seq_2.append(in_seq)
            out = model.sess.run(model.out_decode_prob, 
                        feed_dict=model.gen_decode_prob_feed_dict(in_seq_1, in_seq_2))
            out = -out
            print out
#/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
