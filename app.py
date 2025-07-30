#to run--> streamlit run app.py

import streamlit as st
from PIL import Image
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from keras.utils import register_keras_serializable
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import mixed_precision

# --- 1. Bahdanau Attention Layer ---

class BahdanauAttention(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)
        self.units = units
        self.W1 = keras.layers.Dense(units)     # For image features
        self.W2 = keras.layers.Dense(units)     # For decoder hidden state
        self.V = keras.layers.Dense(1)          # For score

    def call(self, features, hidden):
        # features shape: [batch_size, num_features, embedding_dim] - from CNN
        # hidden shape: [batch_size, rnn_units] - from LSTM previous hidden state

        # Expand hidden state to apply W2
        hidden_with_time_axis = tf.expand_dims(hidden, 1) # [batch_size, 1, rnn_units]

        # [batch, num_features, units] + [batch, 1, units] = [batch_size, num_features, units]
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        # Attention weights shape: [batch_size, num_features, 1]
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # [batch, num_features, embedding_dim]
        context_vector = attention_weights * features # element-wise multiplication --> Tensorflow Broadcasting

        # Context vector shape: [batch_size, embedding_dim]
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

# --- Full Image Caption Generator Model ---

@register_keras_serializable(package="Custom", name='ImageCaptionGenerator')
class ImageCaptionGenerator(tf.keras.Model):
    def __init__(self, vocab_size, max_seq_length, embedding_dim, rnn_units, attention_units, **kwargs):
        super().__init__(**kwargs)

        # Encoder: EfficientNetB3 (bottom-frozen)
        self.cnn_base = keras.applications.EfficientNetB3(include_top=False, weights='imagenet', pooling=None)
        for layer in self.cnn_base.layers[:-30]:
            layer.trainable = False
        for layer in self.cnn_base.layers[-30:]:
            layer.trainable = True
        self.pooling = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='Same')
        self.final_conv = keras.layers.Conv2D(100, (3, 3), padding='same', activation='leaky_relu')

        # Decoder Components
        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.embedding_dropout = keras.layers.Dropout(0.5)      # Dropout after embedding

        self.ln_pre_lstm = keras.layers.LayerNormalization()    # LayerNorm before LSTM
        self.lstm = keras.layers.LSTM(rnn_units, return_sequences=True, return_state=True,
                                      dropout=0.4, recurrent_dropout=0.4)
        self.ln_post_lstm = keras.layers.LayerNormalization()   # LayerNorm after LSTM
        self.lstm_dropout = keras.layers.Dropout(0.5)           # Dropout after LSTM

        self.attention = BahdanauAttention(attention_units)

        self.projection = keras.layers.Dense(128, activation='relu')
        self.projection_dropout = keras.layers.Dropout(0.5)
        self.output_dense = keras.layers.Dense(vocab_size, activation='softmax', kernel_regularizer=keras.regularizers.l2(0.005))

        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.rnn_units = rnn_units
        self.attention_units = attention_units
        self.vocab_size = vocab_size

    @tf.function
    def call(self, inputs):
        # image_input : [batch, height, width, channel]  //  caption_input : [batch, max_seq]
        image_input, caption_input = inputs

        # --- Encoder ---
        cnn_features = self.cnn_base(image_input)               # [batch, h, w, c]
        image_features_inter = self.pooling(cnn_features)       # [batch, h/2, w/2, c]
        image_features = self.final_conv(image_features_inter)  # [batch, h/2, w/2, 128]
        b, h, w, c = tf.unstack(tf.shape(image_features))
        features = tf.reshape(image_features, (b, h * w, c))    # [batch, num_regions, c]

        # --- Decoder ---
        # caption_input : [batch, max_seq]
        embedded_caption0 = self.embedding(caption_input)       # [batch, max_seq, emb_dim]
        embedded_caption = self.embedding_dropout(embedded_caption0)

        batch_size = tf.shape(image_input)[0]
        h_state = tf.zeros((batch_size, self.rnn_units))
        c_state = tf.zeros((batch_size, self.rnn_units))

        outputs = []

        for t in range(self.max_seq_length):
            current_word_embedding = tf.expand_dims(embedded_caption[:, t, :], 1)

            # Attention
            context_vector, attention_weights = self.attention(features, h_state)
            context_vector = tf.expand_dims(context_vector, 1)

            # LSTM Input
            decoder_input0 = tf.concat([context_vector, current_word_embedding], axis=-1)
            decoder_input = self.ln_pre_lstm(decoder_input0)

            output0, h_state, c_state = self.lstm(decoder_input, initial_state=[h_state, c_state])
            output1 = self.ln_post_lstm(output0)
            output2 = self.lstm_dropout(output1)

            # Optional projection before final Dense
            output3 = self.projection(output2)                  # [batch, 1, 256]
            output = self.projection_dropout(output3)           # Apply dropout

            output = tf.reshape(output, (-1, output.shape[2]))
            predicted_word = self.output_dense(output)          # [batch, vocab_size]

            outputs.append(predicted_word)

        final_output = tf.stack(outputs, axis=1)                # [batch, seq_len, vocab_size]
        return final_output

    def predict_caption(self, image_tensor, tokenizer, start_token='start', end_token='end'):
        start_idx = tokenizer.word_index[start_token]
        end_idx = tokenizer.word_index[end_token]

        # 1. Encode the image just once
        cnn_features = self.cnn_base(image_tensor, training=False)
        image_features_inter = self.pooling(cnn_features)
        image_features = self.final_conv(image_features_inter)
        b, h, w, c = tf.unstack(tf.shape(image_features))
        features = tf.reshape(image_features, (b, h * w, c))

        # 2. Initialize the decoder state
        h_state = tf.zeros((1, self.rnn_units))
        c_state = tf.zeros((1, self.rnn_units))

        # 3. Start the sequence with the <start> token
        # The shape should be [batch_size, 1] for the embedding layer
        next_word_input = tf.expand_dims([start_idx], 0)

        result_ids = []

        for _ in range(self.max_seq_length):
            # 4. Embed the last word and get attention context
            # Input shape: [1, 1] -> Output shape: [1, 1, embedding_dim]
            word_embedding = self.embedding(next_word_input, training=False)

            context_vector, _ = self.attention(features, h_state)
            context_vector = tf.expand_dims(context_vector, 1)

            # 5. Run one step of the decoder
            decoder_input = tf.concat([context_vector, word_embedding], axis=-1)
            # Pass training=False to all layers with this argument
            decoder_input = self.ln_pre_lstm(decoder_input, training=False)

            output, h_state, c_state = self.lstm(decoder_input, initial_state=[h_state, c_state], training=False)

            output = self.ln_post_lstm(output, training=False)
            output = self.lstm_dropout(output, training=False)
            output = self.projection(output)
            output = self.projection_dropout(output, training=False)

            # 6. Predict the next word ID
            predicted_logits = self.output_dense(output)
            predicted_id = tf.argmax(predicted_logits, axis=2)[0, 0].numpy()

            if predicted_id == end_idx:
                break

            result_ids.append(predicted_id)

            # 7. Use the predicted ID as the input for the next iteration
            next_word_input = tf.expand_dims([predicted_id], 0)

        # 8. Convert token IDs to words
        inv_word_index = {v: k for k, v in tokenizer.word_index.items()}
        result_text = [inv_word_index.get(idx, '') for idx in result_ids]

        return ' '.join(result_text)

    def get_config(self):
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'max_seq_length': self.max_seq_length,
            'embedding_dim': self.embedding_dim,
            'rnn_units': self.rnn_units,
            'attention_units': self.attention_units
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Fix mixed precision policy if it's a string
        if "dtype" in config and isinstance(config["dtype"], str):
            config["dtype"] = mixed_precision.Policy(config["dtype"])
        return cls(**config)

# Load model and tokenizer
@st.cache_resource
def load_model_tokenizer():
    model = load_model("model3.keras", custom_objects={
            "ImageCaptionGenerator": ImageCaptionGenerator,
            "BahdanauAttention": BahdanauAttention
        } # , compile=False  # 🔥 Critical fix
    )   
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model_tokenizer()

def preprocess_image(pil_img, target_size=(299, 299)):
    pil_img = pil_img.resize(target_size)
    img_array = img_to_array(pil_img)                     # shape: (299, 299, 3)
    img_array = np.expand_dims(img_array, axis=0)         # shape: (1, 299, 299, 3)
    img_array = preprocess_input(img_array)               # normalized input
    return tf.convert_to_tensor(img_array, dtype=tf.float32)


st.title("🖼️ Image Caption Generator")
st.write("Upload an image to get its AI-generated caption.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image")

    if st.button("Generate Caption"):
        image_tensor = preprocess_image(img)  # Now resized to (299, 299)
        caption = model.predict_caption(image_tensor, tokenizer)
        st.success("📝 Caption: " + caption)

