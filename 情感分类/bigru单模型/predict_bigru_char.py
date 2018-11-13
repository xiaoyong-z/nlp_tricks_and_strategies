from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
import gc
import pandas as pd
import pickle
import numpy as np
np.random.seed(16)
from tensorflow import set_random_seed
set_random_seed(16)
from keras.layers import *
from keras.preprocessing import sequence
from gensim.models.keyedvectors import KeyedVectors
from classifier_bigru import TextClassifier


def getClassification(arr):
    arr = list(arr)
    if arr.index(max(arr)) == 0:
        return -2
    elif arr.index(max(arr)) == 1:
        return -1
    elif arr.index(max(arr)) == 2:
        return 0
    else:
        return 1


if __name__ == "__main__":
    with open('tokenizer_char.pickle', 'rb') as handle:
        maxlen = 1200
        model_dir = "model_bigru_char/"
        tokenizer = pickle.load(handle)
        word_index = tokenizer.word_index
        validation = pd.read_csv("preprocess/test_char.csv")
        validation["content"] = validation.apply(lambda x: eval(x[1]), axis=1)
        X_test = validation["content"].values
        list_tokenized_validation = tokenizer.texts_to_sequences(X_test)
        input_validation = sequence.pad_sequences(list_tokenized_validation, maxlen=maxlen)
        w2_model = KeyedVectors.load_word2vec_format("word2vec/chars.vector", binary=True, encoding='utf8',
                                                     unicode_errors='ignore')
        embeddings_index = {}
        embeddings_matrix = np.zeros((len(word_index) + 1, w2_model.vector_size))
        word2idx = {"_PAD": 0}
        vocab_list = [(k, w2_model.wv[k]) for k, v in w2_model.wv.vocab.items()]
        for word, i in word_index.items():
            if word in w2_model:
                embedding_vector = w2_model[word]
            else:
                embedding_vector = None
            if embedding_vector is not None:
                embeddings_matrix[i] = embedding_vector

        submit = pd.read_csv("ai_challenger_sentiment_analysis_testa_20180816/sentiment_analysis_testa.csv")
        submit_prob = pd.read_csv("ai_challenger_sentiment_analysis_testa_20180816/sentiment_analysis_testa.csv")


        model = TextClassifier().model(embeddings_matrix, maxlen, word_index)
        model.load_weights(model_dir + "model_13.hdf5")
        predict = model.predict(input_validation)
        submit["location_traffic_convenience"] = list(map(getClassification, predict[0]))
        submit["location_distance_from_business_district"] = list(map(getClassification, predict[1]))
        submit["location_easy_to_find"] = list(map(getClassification, predict[2]))
        submit["service_wait_time"] = list(map(getClassification, predict[3]))
        submit["service_waiters_attitude"] = list(map(getClassification, predict[4]))
        submit["service_parking_convenience"] = list(map(getClassification, predict[5]))
        submit["service_serving_speed"] = list(map(getClassification, predict[6]))
        submit["price_level"] = list(map(getClassification, predict[7]))
        submit["price_cost_effective"] = list(map(getClassification, predict[8]))
        submit["price_discount"] = list(map(getClassification, predict[9]))
        submit["environment_decoration"] = list(map(getClassification, predict[10]))
        submit["environment_noise"] = list(map(getClassification, predict[11]))
        submit["environment_space"] = list(map(getClassification, predict[12]))
        submit["environment_cleaness"] = list(map(getClassification, predict[13]))
        submit["dish_portion"] = list(map(getClassification, predict[14]))
        submit["dish_taste"] = list(map(getClassification, predict[15]))
        submit["dish_look"] = list(map(getClassification, predict[16]))
        submit["dish_recommendation"] = list(map(getClassification, predict[17]))
        submit["others_overall_experience"] = list(map(getClassification, predict[18]))
        submit["others_willing_to_consume_again"] = list(map(getClassification, predict[19]))

        submit_prob["location_traffic_convenience"] = list(predict[0])
        submit_prob["location_distance_from_business_district"] = list(predict[1])
        submit_prob["location_easy_to_find"] = list(predict[2])
        submit_prob["service_wait_time"] = list(predict[3])
        submit_prob["service_waiters_attitude"] = list(predict[4])
        submit_prob["service_parking_convenience"] = list(predict[5])
        submit_prob["service_serving_speed"] = list(predict[6])
        submit_prob["price_level"] = list(predict[7])
        submit_prob["price_cost_effective"] = list(predict[8])
        submit_prob["price_discount"] = list(predict[9])
        submit_prob["environment_decoration"] = list(predict[10])
        submit_prob["environment_noise"] = list(predict[11])
        submit_prob["environment_space"] = list(predict[12])
        submit_prob["environment_cleaness"] = list(predict[13])
        submit_prob["dish_portion"] = list(predict[14])
        submit_prob["dish_taste"] = list(predict[15])
        submit_prob["dish_look"] = list(predict[16])
        submit_prob["dish_recommendation"] = list(predict[17])
        submit_prob["others_overall_experience"] = list(predict[18])
        submit_prob["others_willing_to_consume_again"] = list(predict[19])

        submit.to_csv("baseline_bigru_char.csv", index=None)
        submit_prob.to_csv("baseline_bigru_char_prob.csv", index=None)
