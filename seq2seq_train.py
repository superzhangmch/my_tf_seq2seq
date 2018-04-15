#encoding: utf8
import os
import sys
import tensorflow as tf
import math
import random
import gc
import time

from seq2seq_model import Seq2SeqModel
from gen_sample import read_freq_dict, gen_batch_train_data, read_samples, print_sample

if __name__ == "__main__":
    batch_size = 64
    cur_lr = 0.001
    model = Seq2SeqModel(seq1_dict_size       = 5000,
                         seq1_dict_emb_dize   = 64,
                         seq2_dict_size       = 6000,
                         seq2_dict_emb_dize   = 128,
                         rnn_layer_cnt   = 2,
                         rnn_hidden_size = 128,
                         out_hid_size    = 128,
                         beam_size       = 10,
                         learning_rate   = cur_lr)
    out_freq_file = "./freq.dat"
    m_word, m_word_rev = read_freq_dict(out_freq_file, 5000)
    j = 0
    per_batch_cnt_for_stat = 100
    for step in xrange(5):
        for file_idx in xrange(10):
            read_lines = -1
            samples = read_samples("./seg.sample", read_cnt=read_lines)
            batched_data = gen_batch_train_data(samples, batch_size, do_shuffle=True)
            del samples
            gc.collect()

            total_loss = 0.
            cnt = 0
            tm = time.time()
            for i in xrange(len(batched_data)):
                batch = batched_data[i]

                j += 1

                _, loss = model.sess.run([model.opt, model.out_loss], feed_dict=model.gen_train_feed_dict(*batch))
                cnt += 1
                total_loss += loss

                if j % per_batch_cnt_for_stat == 0:
                    mean_loss = total_loss / cnt 
                    print ("step=%d file=%d/%d: %d/%d=%.2f%%: mean_loss=%.4f " +\
                          "cur_loss=%.4f tm=%.2fs lr=%.9f") % (step, file_idx, j, i, \
                                    len(batched_data), 100. * i / len(batched_data), \
                                    mean_loss, loss, time.time() - tm, cur_lr)
                    sys.stdout.flush()
                    if j % (5*per_batch_cnt_for_stat) == 0:
                        try:
                            cur_lr = float(open("lr.txt").read())
                            model.set_lr(cur_lr)
                        except:
                            print "read lr.txt error"
                            pass

                    total_loss = 0.
                    tm = time.time()
                    cnt = 0
                    if j % 500 == 0:

                        idx = random.randint(0, len(batch[0]) - 1)
                        seq   = batch[0][idx][:batch[1][idx]]
                        seq_1 = batch[2][idx][:batch[3][idx]][1:]
                        model.seq2seq_gen(seq, seq_1, m_word)

                if j % 10000 == 0:
                    model.save_model("./model_py1/seq2seq_%d" % (j))

#/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
