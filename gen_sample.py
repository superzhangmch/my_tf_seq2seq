#encoding: utf8

import random

def gen_freq_dict(sample_file, out_freq_file, field=-1):
    """
    sample_file: Tab分割的序列对文件.前两列是序列对。序列内用空格分割
            格式：id1 id2 .. id_N \t Id1 Id2 .. Id_M \t ...
    out_freq_file: 输出的词频文件
    field: 从sample_file 的哪些列统计得词频文件
        -1: 前两列. seq->seq的前后两个序列可以共享词汇的时候，可以选用这个
        0: 第一列
        1：第二列
    """
    def up_word_m(ch, word_m):
        if ch in ["SOS", "EOS", "UNKNOW"]:
            return
        if ch not in word_m:
            word_m[ch] = 0
        word_m[ch] += 1
    
    def write_file(m, file_name):
        arr = [(k, m[k]) for k in m]
        arr = sorted(arr, key=lambda x:x[1], reverse=True)
        
        fp = open(file_name, "w")
        fp.write("0\t%s\t%d\n" % ("UNKNOW", 0))
        fp.write("1\t%s\t%d\n" % ("SOS", 0))
        fp.write("2\t%s\t%d\n" % ("EOS", 0))
        i = 3
        for k, cnt in arr:
            fp.write("%d\t%s\t%d\n" % (i, k, cnt))
            i += 1

    word_m = {}
    for line in  open(sample_file):
        line = line.strip()
        LL = line.split("\t")
        if len(LL) < 2:
            continue
        if field == 0 or field == -1:
            arr1 = LL[0].split(" ")
            for a in arr1:
                up_word_m(a, word_m)
        if field == 1 or field == -1:
            arr2 = LL[1].split(" ")
            for a in arr2:
                up_word_m(a, word_m)
            
    write_file(word_m, out_freq_file)


def read_freq_dict(file, line_cnt):
    """
    读取词表文件. 词表文件第一列是词，第二列是频次，截取line_cnt行，放入m与m_rev
    m: idx -> word
    m_rev: word -> idx
    """
    i = 0
    m = {}
    m_rev = {}
    for line in open(file):
        LL = line.strip().split("\t")
        if len(LL) < 3:
            continue
        idx = int(LL[0])
        ch = LL[1]
        cnt = int(LL[2])
        assert idx == i, "idx=%d != i=%d" % (idx, i)
        if i >= line_cnt:
            break
        m[idx] = ch
        m_rev[ch] = idx
        i += 1
    return m, m_rev

def print_sample(arr_word_ids, m_word):
    arr1 = [m_word[id] for id in arr_word_ids]
    print "".join(arr1)

def gen_sample_from_seq_str(seq_str, m_word_rev):
    """
    对于原始字符串形式的序列，返回数字编号的序列
    """
    def gen_sample(arr, m_word_rev):
        """
        对于原始序列数组，换成数字编号的序列
        """
        sample_word_arr = []
        for a in arr:
            id_word = m_word_rev["UNKNOW"] if a not in m_word_rev else m_word_rev[a]
            sample_word_arr.append(id_word)
        return sample_word_arr

    arr = seq_str.split(" ")
    word_sample = gen_sample(arr, m_word_rev)
    return word_sample

def gen_sample_file(ori_sample_file, save_sample_file, m_word_rev_1, m_word_rev_2):
    fp_out = open(save_sample_file, "w")

    def num_arr_2_str(arr):
        arr = [str(a) for a in arr]
        return ",".join(arr)

    i = 0
    for line in open(ori_sample_file):
        line = line.strip()
        LL = line.split("\t")
        if len(LL) < 2:
            continue
        id_word_0 = gen_sample_from_seq_str(LL[0], m_word_rev_1)
        id_word_1 = gen_sample_from_seq_str(LL[1], m_word_rev_2)
        fp_out.write("%s\t%s\n" % (num_arr_2_str(id_word_0), num_arr_2_str(id_word_1)))
        i += 1
        if i % 10000 == 0:
            print i

def read_samples(sample_file, read_cnt=-1, max_seq_len=25):
    """
    读取样本数据
    sample_file: gen_sample_file(...)的输出文件
    """
    samples = []
    i = 0
    for line in open(sample_file):
        LL = line.strip().split("\t")
        if len(LL) < 2:
            continue

        arr = [[int(L) for L in LL[0].split(",")], [int(L) for L in LL[1].split(",")]]
        if len(arr[0]) > max_seq_len or len(arr[1]) > max_seq_len:
            continue

        samples.append(arr)
        i += 1
        if i % 100000 == 0:
            print "read %s line %d" % (sample_file, i)
        if read_cnt > 0 and i > read_cnt:
            break
    return samples


def gen_batch_train_data(samples, batch_size, do_shuffle=False, for_seq2seq=True):
    """
    samples: read_samples(..)的返回，或类似数据结果
    batch_size: 按多大batch size 来组合训练样本数据
    for_seq2seq: 是否用来训练seq2seq，如果是，则需要补充SOS、EOS特殊token，
                 且按shift1方式生成decode数据
    """
    SOS = 1
    EOS = 2
    if do_shuffle:
        random.shuffle(samples)
    batched_data = []
    cnt = len(samples) / batch_size
    if len(samples) % batch_size != 0:
        cnt += 1

    for i in xrange(cnt):
        d = samples[i*batch_size: i*batch_size+batch_size]

        encode_inputs_seq_len = [len(a[0]) for a in d]

        len_adjust = 0
        if for_seq2seq:
            len_adjust = 1
        decode_inputs_seq_len = [len(a[1]) + len_adjust for a in d]

        max_len_encode = max(encode_inputs_seq_len)
        max_len_decode = max(decode_inputs_seq_len)

        encode_inputs = [(a[0] + [0]*(max_len_encode-len(a[0]))) for a in d]
        if for_seq2seq:
            decode_inputs = [([SOS] + a[1] + [0]*(max_len_decode-(len(a[1])+1))) for a in d]
            decode_labels = [(a[1] + [EOS] + [0]*(max_len_decode-(len(a[1])+1))) for a in d]
        else:
            decode_inputs = [(a[1] + [0]*(max_len_decode-len(a[1]))) for a in d]
            decode_labels = None
        batched_data.append([encode_inputs, encode_inputs_seq_len,
                             decode_inputs, decode_inputs_seq_len, decode_labels])
    return batched_data

if __name__ == "__main__":

    sample_file = "seg"
    out_freq_file = "freq.dat"
    gen_freq_dict(sample_file, out_freq_file)
    m_word, m_word_rev = read_freq_dict(out_freq_file, 100)
