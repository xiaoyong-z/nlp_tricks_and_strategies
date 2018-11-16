import keras
from keras import Model
from keras.layers import *
from JoinAttLayer import Attention
from keras.losses import categorical_crossentropy

hidden_size_class = 512

def mycrossentropy(y_true, y_pred):
    # loss = 0
    # for i in range(20):
    #     y_true_part = y_true[i]
    #     y_pred_part = y_pred[i]
    loss = categorical_crossentropy(y_true, y_pred)
    return loss

class TextClassifier():

    def model(self, embeddings_matrix, maxlen, word_index):
        inp = Input(shape=(maxlen,))
        encode = Bidirectional(CuDNNGRU(300, return_sequences=True))
        encode2 = Bidirectional(CuDNNGRU(300, return_sequences=True))
        attention = Attention(maxlen)
        x_4 = Embedding(len(word_index) + 1,
                        embeddings_matrix.shape[1],
                        weights=[embeddings_matrix],
                        input_length=maxlen,
                        trainable=True)(inp)
        x_3 = SpatialDropout1D(0.2)(x_4)
        x_3 = encode(x_3)
        x_3 = Dropout(0.2)(x_3)
        x_3 = encode2(x_3)
        x_3 = Dropout(0.2)(x_3)
        avg_pool_3 = GlobalAveragePooling1D()(x_3)
        max_pool_3 = GlobalMaxPooling1D()(x_3)
        attention_3 = attention(x_3)
        x = keras.layers.concatenate([avg_pool_3, max_pool_3, attention_3], name="fc")

        output = []

        x_location = Dense(units=hidden_size_class, activation="tanh", )(x)
        x_location = Dropout(0.2)(x_location)
        x_service = Dense(units=hidden_size_class, activation="tanh")(x)
        x_service = Dropout(0.2)(x_service)
        x_price = Dense(units=hidden_size_class, activation="tanh")(x)
        x_price = Dropout(0.2)(x_price)
        x_environment = Dense(units=hidden_size_class, activation="tanh")(x)
        x_environment = Dropout(0.2)(x_environment)
        x_dish = Dense(units=hidden_size_class, activation="tanh")(x)
        x_dish = Dropout(0.2)(x_dish)
        x_others = Dense(units=hidden_size_class, activation="tanh")(x)
        x_others = Dropout(0.2)(x_others)

        for i in range(3):
            output.append(Dense(4, activation="sigmoid")(x_location))

        for i in range(3, 7):
            output.append(Dense(4, activation="sigmoid")(x_service))

        for i in range(7, 10):
            output.append(Dense(4, activation="sigmoid")(x_price))

        for i in range(10, 14):
            output.append(Dense(4, activation="sigmoid")(x_environment))

        for i in range(14, 18):
            output.append(Dense(4, activation="sigmoid")(x_dish))

        for i in range(18, 20):
            output.append(Dense(4, activation="sigmoid")(x_others))
        adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,amsgrad=True)
        model = Model(inputs=inp, outputs=output)
        model.compile(
            loss=mycrossentropy,
            optimizer=adam)
        return model
