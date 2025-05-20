import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Layer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import warnings
warnings.simplefilter("ignore", FutureWarning)


input_texts = [
    "Hello.", "How are you?",
    "I am learning machine translation.",
    "What is your name?", "I love programming."
]
target_texts = [
    "Hola.", "¿Cómo estás?",
    "Estoy aprendiendo traducción automática.",
    "¿Cuál es tu nombre?", "Me encanta programar."
]
target_texts = ["startseq " + t + " endseq" for t in target_texts]


input_tok = Tokenizer()
output_tok = Tokenizer()

input_tok.fit_on_texts(input_texts)
output_tok.fit_on_texts(target_texts)

enc_seqs = input_tok.texts_to_sequences(input_texts)
dec_seqs = output_tok.texts_to_sequences(target_texts)

max_enc_len = max(len(s) for s in enc_seqs)
max_dec_len = max(len(s) for s in dec_seqs)

enc_seqs = pad_sequences(enc_seqs, maxlen=max_enc_len, padding="post")
dec_seqs = pad_sequences(dec_seqs, maxlen=max_dec_len, padding="post")

enc_vocab = len(input_tok.word_index) + 1
dec_vocab = len(output_tok.word_index) + 1


dec_in  = dec_seqs[:, :-1]
dec_out = dec_seqs[:, 1:]
dec_out = np.array([np.eye(dec_vocab)[row] for row in dec_out])


class SelfAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        d = input_shape[-1]

        self.Wq = self.add_weight(
            name="Wq",
            shape=(d, d),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.Wk = self.add_weight(
            name="Wk",
            shape=(d, d),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.Wv = self.add_weight(
            name="Wv",
            shape=(d, d),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        q = K.dot(x, self.Wq)
        k = K.dot(x, self.Wk)
        v = K.dot(x, self.Wv)
        scale = K.sqrt(K.cast(K.shape(k)[-1], K.floatx()))
        scores = K.batch_dot(q, k, axes=[2, 2]) / scale
        w = K.softmax(scores, axis=-1)
        return K.batch_dot(w, v)

    def compute_output_shape(self, shape):
        return shape


def build_model(init="glorot_uniform", optimizer="adam"):
    # Encoder
    enc_in   = Input(shape=(max_enc_len,))
    enc_emb  = Embedding(enc_vocab, 256)(enc_in)
    enc_lstm = LSTM(256, return_sequences=True, return_state=True,
                    kernel_initializer=init)
    enc_out, h, c = enc_lstm(enc_emb)
    enc_states = [h, c]

    enc_out = SelfAttention()(enc_out)

    dec_inp = Input(shape=(max_dec_len - 1,))
    dec_emb = Embedding(dec_vocab, 256)(dec_inp)
    dec_lstm = LSTM(256, return_sequences=True, return_state=True,
                    kernel_initializer=init)
    dec_out, _, _ = dec_lstm(dec_emb, initial_state=enc_states)
    dec_out = SelfAttention()(dec_out)
    dec_out = Dense(dec_vocab, activation="softmax")(dec_out)

    model = Model([enc_in, dec_inp], dec_out)
    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def main():
    model_glorot = build_model(init="glorot_uniform", optimizer="adam")
    hist_g = model_glorot.fit([enc_seqs, dec_in], dec_out,
                              epochs=100, batch_size=16, verbose=0)

    model_he = build_model(init="he_uniform", optimizer="adam")
    hist_h = model_he.fit([enc_seqs, dec_in], dec_out,
                          epochs=100, batch_size=16, verbose=0)

    plt.plot(hist_g.history["loss"], label="Glorot + Adam")
    plt.plot(hist_h.history["loss"], label="He + Adam")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Training complete.")


if __name__ == "__main__":
    main()
