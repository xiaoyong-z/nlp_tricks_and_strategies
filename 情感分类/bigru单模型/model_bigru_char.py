from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
import random
random.seed = 42
import pandas as pd
from tensorflow import set_random_seed
set_random_seed(42)
from keras.preprocessing import text, sequence
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report
from keras.layers import *
from classifier_bigru import TextClassifier
from gensim.models.keyedvectors import KeyedVectors
import pickle
import gc
import numpy as np

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


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = np.zeros((20, 100))

    def on_epoch_end(self, epoch, logs={}):
        val_predict = self.model.predict(self.validation_data[0])
        val_targ = self.validation_data
        all_max_loss = []
        current_loss = []
        for i in range(20):
            print("\n\nlabel : ", i+1)
            print("\n")
            val_predict_single_label = list(map(getClassification, val_predict[i]))
            val_targ_single_label = list(map(getClassification, val_targ[i+1]))
            _val_f1 = f1_score(val_targ_single_label, val_predict_single_label, average="macro")
            print(classification_report(val_targ_single_label, val_predict_single_label))
            self.val_f1s[i][epoch] = _val_f1
            current_loss.append(_val_f1)
            print(_val_f1)
            print("max f1")
            max_loss = max(self.val_f1s[i])
            all_max_loss.append(max_loss)
            print(max_loss)
        print("\n ___________________________________Average_loss___________________________________\n", np.average(current_loss))
        return


data = pd.read_csv("preprocess/train_char.csv")
data["content"] = data.apply(lambda x: eval(x[1]), axis=1)

validation = pd.read_csv("preprocess/validation_char.csv")
validation["content"] = validation.apply(lambda x: eval(x[1]), axis=1)

model_dir = "model_bigru_char/"
maxlen = 1200
max_features = 20000
batch_size = 128
epochs = 200
tokenizer = text.Tokenizer(num_words=None)
tokenizer.fit_on_texts(data["content"].values)
with open('tokenizer_char.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


word_index = tokenizer.word_index
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

X_train = data["content"].values
Y_train_ltc = pd.get_dummies(data["location_traffic_convenience"])[[-2, -1, 0, 1]].values
Y_train_ldfbd = pd.get_dummies(data["location_distance_from_business_district"])[[-2, -1, 0, 1]].values
Y_train_letf = pd.get_dummies(data["location_easy_to_find"])[[-2, -1, 0, 1]].values
Y_train_swt = pd.get_dummies(data["service_wait_time"])[[-2, -1, 0, 1]].values
Y_train_swa = pd.get_dummies(data["service_waiters_attitude"])[[-2, -1, 0, 1]].values
Y_train_spc = pd.get_dummies(data["service_parking_convenience"])[[-2, -1, 0, 1]].values
Y_train_ssp = pd.get_dummies(data["service_serving_speed"])[[-2, -1, 0, 1]].values
Y_train_pl = pd.get_dummies(data["price_level"])[[-2, -1, 0, 1]].values
Y_train_pce = pd.get_dummies(data["price_cost_effective"])[[-2, -1, 0, 1]].values
Y_train_pd = pd.get_dummies(data["price_discount"])[[-2, -1, 0, 1]].values
Y_train_ed = pd.get_dummies(data["environment_decoration"])[[-2, -1, 0, 1]].values
Y_train_en = pd.get_dummies(data["environment_noise"])[[-2, -1, 0, 1]].values
Y_train_es = pd.get_dummies(data["environment_space"])[[-2, -1, 0, 1]].values
Y_train_ec = pd.get_dummies(data["environment_cleaness"])[[-2, -1, 0, 1]].values
Y_train_dp = pd.get_dummies(data["dish_portion"])[[-2, -1, 0, 1]].values
Y_train_dt = pd.get_dummies(data["dish_taste"])[[-2, -1, 0, 1]].values
Y_train_dl = pd.get_dummies(data["dish_look"])[[-2, -1, 0, 1]].values
Y_train_dr = pd.get_dummies(data["dish_recommendation"])[[-2, -1, 0, 1]].values
Y_train_ooe = pd.get_dummies(data["others_overall_experience"])[[-2, -1, 0, 1]].values
Y_train_owta = pd.get_dummies(data["others_willing_to_consume_again"])[[-2, -1, 0, 1]].values
Y_train_total = [Y_train_ltc, Y_train_ldfbd, Y_train_letf, Y_train_swt, Y_train_swa, Y_train_spc, Y_train_ssp,
                          Y_train_pl, Y_train_pce, Y_train_pd, Y_train_ed, Y_train_en, Y_train_es, Y_train_ec,
                          Y_train_dp, Y_train_dt, Y_train_dl, Y_train_dr, Y_train_ooe, Y_train_owta
                          ]

X_validation = validation["content"].values
Y_validation_ltc = pd.get_dummies(validation["location_traffic_convenience"])[[-2, -1, 0, 1]].values
Y_validation_ldfbd = pd.get_dummies(validation["location_distance_from_business_district"])[[-2, -1, 0, 1]].values
Y_validation_letf = pd.get_dummies(validation["location_easy_to_find"])[[-2, -1, 0, 1]].values
Y_validation_swt = pd.get_dummies(validation["service_wait_time"])[[-2, -1, 0, 1]].values
Y_validation_swa = pd.get_dummies(validation["service_waiters_attitude"])[[-2, -1, 0, 1]].values
Y_validation_spc = pd.get_dummies(validation["service_parking_convenience"])[[-2, -1, 0, 1]].values
Y_validation_ssp = pd.get_dummies(validation["service_serving_speed"])[[-2, -1, 0, 1]].values
Y_validation_pl = pd.get_dummies(validation["price_level"])[[-2, -1, 0, 1]].values
Y_validation_pce = pd.get_dummies(validation["price_cost_effective"])[[-2, -1, 0, 1]].values
Y_validation_pd = pd.get_dummies(validation["price_discount"])[[-2, -1, 0, 1]].values
Y_validation_ed = pd.get_dummies(validation["environment_decoration"])[[-2, -1, 0, 1]].values
Y_validation_en = pd.get_dummies(validation["environment_noise"])[[-2, -1, 0, 1]].values
Y_validation_es = pd.get_dummies(validation["environment_space"])[[-2, -1, 0, 1]].values
Y_validation_ec = pd.get_dummies(validation["environment_cleaness"])[[-2, -1, 0, 1]].values
Y_validation_dp = pd.get_dummies(validation["dish_portion"])[[-2, -1, 0, 1]].values
Y_validation_dt = pd.get_dummies(validation["dish_taste"])[[-2, -1, 0, 1]].values
Y_validation_dl = pd.get_dummies(validation["dish_look"])[[-2, -1, 0, 1]].values
Y_validation_dr = pd.get_dummies(validation["dish_recommendation"])[[-2, -1, 0, 1]].values
Y_validation_ooe = pd.get_dummies(validation["others_overall_experience"])[[-2, -1, 0, 1]].values
Y_validation_owta = pd.get_dummies(validation["others_willing_to_consume_again"])[[-2, -1, 0, 1]].values
Y_val_total = [Y_validation_ltc, Y_validation_ldfbd, Y_validation_letf, Y_validation_swt, Y_validation_swa, Y_validation_spc, Y_validation_ssp,
                          Y_validation_pl, Y_validation_pce, Y_validation_pd, Y_validation_ed, Y_validation_en, Y_validation_es, Y_validation_ec,
                          Y_validation_dp, Y_validation_dt, Y_validation_dl, Y_validation_dr, Y_validation_ooe, Y_validation_owta]


list_tokenized_train = tokenizer.texts_to_sequences(X_train)
input_train = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)

list_tokenized_validation = tokenizer.texts_to_sequences(X_validation)
input_validation = sequence.pad_sequences(list_tokenized_validation, maxlen=maxlen)

print("model")
model = TextClassifier().model(embeddings_matrix, maxlen, word_index)
file_path = model_dir + "model_{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(file_path, verbose=2, save_weights_only=True)
metrics = Metrics()
callbacks_list = [checkpoint, metrics]
history = model.fit(input_train, Y_train_total, batch_size=batch_size, epochs=epochs,
                     validation_data=[input_validation, Y_val_total], callbacks=callbacks_list, verbose=2)
del model
del history
gc.collect()
K.clear_session()
