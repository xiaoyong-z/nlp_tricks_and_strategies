# -*- coding: utf-8 -*-
# base
import csv
import pandas as pd
from string import punctuation
# my
from config import Config

# conf-my
myconf = Config()
vocab = [int(i) for i in myconf.label.vocab.split(',')]
fields = myconf.data.fields.split(',')


def get_stopwords():
    with open("dict/stopwords.txt") as f:
        return [line.strip() for line in f.readlines()]

punctuation_zh = ".、，。°？！：；“”’‘～…【】『』［］（）()《》｛｝×―－·•→℃〈〉"
spec = [' ', '\xa0', '\n', '\ufeff', '\r']
# Combination of Chinese and English punctuation.
punctuation_all = punctuation + punctuation_zh
# 表情
emoticon = myconf.data.emoticon
# 句尾语气词
tone_words = "的了呢吧吗啊啦呀么嘛哒哼"

# char level
# 1.不去
filter_characters = spec
# 2.去掉 标点+停用词+表情符号
# filter_characters = list(set(punctuation_all) | set(get_stopwords()) | set(emoticon)) + spec
# 3.去掉 标点+表情符号
# filter_characters = list(set(punctuation_all) | set(emoticon)) + spec
# 4.去掉 标点
# filter_characters = list(set(punctuation_all)) + spec


def get_chars(text):
    res = [c for c in text if c not in filter_characters]
    return res


def process_csv_dict(srcfile=None, desfile=None, func=None, fields=None):
    """按行以字典形式读取 csv 文件进行自定义处理，将结果保存到目标文件。
    直到遇到空行或者到达文件末尾为止。
    """
    n = 0
    with open(srcfile, 'r', encoding='utf-8-sig') as src, open(desfile, 'w', encoding='utf-8') as des:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(des, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            writer.writerow(func(row, fields))
            n += 1
            if n % 100 == 0:
                print(n)
    print("process_csv_dict complete!")


def process_pred(srcfile=None, desfile=None, func=None, fields=None):
    n = 0
    with open(desfile, 'w', encoding='UTF-8') as des:
        if fields:
            des.write(','.join(fields) + '\n') # header
        with open(srcfile, 'r', encoding='UTF-8') as src:
            for line in iter(src.readline, ''):
                des.write(','.join(func(line)) + '\n')
                n += 1
                if n % 100 == 0:
                    print(n)
    print("process_pred complete!")


def generate(pfile=None, desfile=None, k=None, default=None):
    """用粒度 k 的预测结果更新测试文件中相应的列。
    """
    assert pfile is not None, "pfile can not be None"
    assert desfile is not None, "desfile can not be None"
    assert k is not None, "k can not be None"

    test = pd.read_csv(desfile)
    ddata = test.copy()
    pdata = pd.read_csv(pfile)
    # ddata[k] = pdata[k].astype(int)
    ddata[k] = pdata[k].fillna(default).astype(int)
    with open(desfile, mode='w', newline='\n', encoding='UTF-8') as f:
        ddata.to_csv(f, index=False)