# coding: utf-8

# 训练词向量 word2vec

# import word2vec
from gensim.models import word2vec

sentences = []
with open('chars.txt', 'r', encoding='UTF-8') as f:
    for line in iter(f.readline, ''):
        sentences.append(line.split())
model = word2vec.Word2Vec(sentences, size=100, window=5, min_count=2, workers=10, iter=15)
# # 打印单词 '很多' 的词向量
# print(model.wv.word_vec('很多'))
# # 打印和 '很多' 相似的前2个单词
# print(model.wv.most_similar('很多', topn=2))
# 保存模型到文件
model.wv.save_word2vec_format("gensim/chars.vector", binary=True)
# model.save('gensim/chars.bin')

# word2vec.word2vec('src.txt', 'vectors_100d.bin', size=100, verbose=True)