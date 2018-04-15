#encoding: utf8

import sys
from gen_sample import gen_freq_dict, read_freq_dict, gen_sample_file

if __name__ == "__main__":

    if len(sys.argv) < 2 or sys.argv[1] not in ["gen_freq_dict", "gen_sample"]:
        print "Usage:"
        print "    python %s gen_freq_dict $in_sample_file $out_freq_file"
        print "    python %s gen_sample $in_freq_file $word_cnt $in_ori_sample_file $out_ID_sample_file"
        sys.exit(-1)

    if sys.argv[1] == "gen_freq_dict":
        sample_file = sys.argv[2]
        out_freq_file = sys.argv[3]
        gen_freq_dict(sample_file, out_freq_file)
    else:
        # 词频文件, 格式: word_idx \t word \t freq
        # 按第3列排好序了. 前三行是 UNKNOW, SOS, EOS 特殊word的
        word_freq_dict_file = sys.argv[2]
        # 从词频文件截取前多少高频词
        word_count          = int(sys.argv[3])
        # 原始pair样本文件，每行一个pair，tag分割，前两列构成一个pair
        ori_sample_file     = sys.argv[4]
        # 生成的样本文件, word已经转化为了数字index形式
        save_sample_file    = sys.argv[5]

        m_word, m_word_rev = read_freq_dict(word_freq_dict_file, word_count)

        gen_sample_file(ori_sample_file, save_sample_file, m_word_rev, m_word_rev)
