# coding: utf-8
import numpy as np
import pandas as pd
from collections import Counter, OrderedDict


def f1(pred, label, vocab):
	"""计算单一粒度的 f1
	usage:
		vocab = [1, 0, -1, -2]
		pred = [1, 0, 1, -2, -1, 0, -1, 1, -2]
		label = [1, -2, -1, -2, -1, 0, 1, 1, 1]
		print(f1_score(pred, label, vocab))
	"""
	assert vocab is not None, "vocab can not be None!"
	assert len(pred) == len(label), "len(pred) must be equal to len(label)!"
	tmp = []
	for v in vocab:
		tp, fp, tn, fn = 0, 0, 0, 0
		precision, recall = 0, 0
		for p, l in zip(pred, label):
			if p == v:
				if p == l:
					tp += 1
				else:
					fp += 1
			else:
				if l == v:
					fn += 1
				else:
					tn += 1
		# print(v, tp, fp, tn, fn)
		precision = tp / (tp + fp + 0.01)
		recall = tp / (tp + fn + 0.01)
		f1_v = 2 * precision * recall / (precision + recall + 0.01)
		tmp.append(f1_v)
	# print(tmp)
	return np.mean(tmp)


def f1_mltc(pfile=None, lfile=None, fields=None, vocab=None):
	"""Computing f1_score of Multi Label Task Classification
	"""
	assert pfile is not None, "pfile can not be None"
	assert lfile is not None, "lfile can not be None"
	assert fields is not None, "fields can not be None"
	assert vocab is not None, "vocab can not be None"

	pdata = pd.read_csv(pfile)
	ldata = pd.read_csv(lfile)
	tmp = []
	for k in fields:
		f1_k = f1(pdata[k], ldata[k], vocab)
		tmp.append(f1_k)
		print(k, f1_k)
	return np.mean(tmp)


def count_label(filepath=None, fields=None):
    """统计指定粒度的标签。
    """
    assert filepath is not None, "filepath can not be None"
    data = pd.read_csv(filepath)
    mat = '{:50}'
    res = {}
    for k in fields:
        tmp = Counter(data[k].astype(int))
        counter = OrderedDict(sorted(tmp.items(), key=lambda t: t[0]))
        print(mat.format(k), counter, '\n')
        res[k] = counter
    return res

def get_maxlen(srcfile=None):
    """获取训练集单个序列的最大长度
    """
    maxlen = 0
    assert srcfile is not None, "srcfile can not be None"
    with open(srcfile, 'r', encoding='UTF-8') as src:
        for line in iter(src.readline, ''):
            maxlen = max(maxlen, len(line.split()))
    return maxlen

def get_avglen(srcfile=None):
    """获取训练集单个序列的平均长度
    """
    assert srcfile is not None, "srcfile can not be None"
    with open(srcfile, 'r', encoding='UTF-8') as src:
        tmp = [len(line.split()) for line in iter(src.readline, '')]
        counter = Counter(tmp)
        res = OrderedDict(sorted(counter.items(), key=lambda t: t[0]))
        return np.mean(tmp), res
    return None


if __name__ == '__main__':
	print("maxlen:", get_maxlen(srcfile="preprocess/base/chars.txt"))
	print("avglen:", get_avglen(srcfile="preprocess/base/chars.txt"))