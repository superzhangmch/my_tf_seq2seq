#encoding: utf8
import os
import sys
import tensorflow as tf
import math
import random
import gc
import time

from fast_sent_model import FastSentModel

from gen_sample import read_freq_dict, gen_batch_train_data, read_samples, print_sample

if __name__ == "__main__":
    batch_size = 64
    cur_lr = 0.01
    model = FastSentModel(dict_size       = 5000,
                          dict_emb_dize   = 128,
                          learning_rate   = cur_lr)

    
    #model.restore_model("./model_fast/fast_310000")
    out_freq_file = "./freq.dat"
    m_word, m_word_rev = read_freq_dict(out_freq_file, 5000)

    j = 0
    #for step in xrange(5):
    for step in xrange(1):
        #for file_idx in xrange(10):
        for file_idx in xrange(40):
            read_lines = -1

            samples = read_samples("./seg.sample", read_cnt=read_lines)
            batched_data = gen_batch_train_data(samples, batch_size, 
                                                do_shuffle=True, for_seq2seq=False)
            del samples
            gc.collect()

            total_loss = 0.
            cnt = 0
            tm = time.time()
            for i in xrange(len(batched_data)):
                batch = batched_data[i]

                j += 1

                _, loss = model.sess.run([model.opt, model.out_loss], 
                                          feed_dict=model.gen_train_feed_dict(*batch[:-1]))
                cnt += 1
                total_loss += loss

                if j % 100 == 0:
                    try:
                        cur_lr = float(open("lr.txt").read())
                        model.set_lr(cur_lr)
                    except:
                        pass
                    mean_loss = total_loss / cnt
                    print ("step=%d file=%d/%d: %d/%d=%.2f%%: mean_loss=%.4f " \
                            +"cur_loss=%.4f tm=%.2fs lr=%f") % (step, file_idx, j, i, len(batched_data), 
                            100. * i / len(batched_data), mean_loss, loss, \
                            time.time() - tm, \
                            cur_lr)
                    sys.stdout.flush()

                    total_loss = 0.
                    tm = time.time()
                    cnt = 0
                if j  % 10000 == 0:
                    model.save_model("./model_fast/fast_%d" % (j))

#/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
