# imports
import gc
import json
import math
import os
import re
import time


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tqdm.auto import tqdm

from src.utils import EpochTracker, ModelTracker, TokenManager

# Initiate Token Manager
token_manager = TokenManager()

def run_transformer(MODEL_NAME, MAX_VOCAB_SIZE, D_MODEL, N_LAYERS, FFN_UNITS, N_HEADS, DROPOUT_RATE, ACTIVATION, USE_POSITIONAL, NUM_SAMPLES = 80000, MAX_LENGTH=15, EPOCHS = 10, LANG="SPA"):
    # Parameters for our model
    INPUT_COLUMN = "input"
    TARGET_COLUMN = "target"
    BATCH_SIZE = 64  # Batch size for training.
    
    if LANG == "DE":
        DATASET_FILENAME = "deu-eng.zip"
        data_folder_name = "datasets/eng_de_translations"
        train_filename = "deu.txt"
    else:
        DATASET_FILENAME = "spa-eng.zip"
        data_folder_name = "datasets/eng_spa_translations"
        train_filename = "spa.txt"
    print(f"{DATASET_FILENAME=}")

    # Global parameters
    root_folder = "."
    checkpoint_folder = f"ckpt-{MODEL_NAME}/"
    metrics_folder = "metrics/"
    epoch_filename = f"epoch-{MODEL_NAME}.json"
    hyperparameter_filename = f"hyperparameters-{MODEL_NAME}.json"

    # Variable for data directory
    DATA_PATH = os.path.abspath(os.path.join(root_folder, data_folder_name))
    train_filenamepath = os.path.abspath(os.path.join(DATA_PATH, train_filename))
    checkpoint_path = os.path.abspath(os.path.join(root_folder, checkpoint_folder))
    metrics_path = os.path.abspath(os.path.join(root_folder, metrics_folder))

    # epoch metrics and counter
    epoch_filenamepath = os.path.abspath(os.path.join(metrics_folder, epoch_filename))

    # hyperparameter file
    hyperparameter_filenamepath = os.path.abspath(os.path.join(metrics_folder, hyperparameter_filename))

    # Both train and test set are in the root data directory
    train_path = DATA_PATH

    DATASET_PATH = os.path.abspath(os.path.join(DATA_PATH, DATASET_FILENAME))

    # Store hyperparameters in dict
    hyperparameters_ = {"MODEL NAME": MODEL_NAME, "MAX VOCAB SIZE": MAX_VOCAB_SIZE, "D_MODEL": D_MODEL, "N_LAYERS": N_LAYERS, "FFN_UNITS": FFN_UNITS, "N_HEADS": N_HEADS, "DROPOUT_RATE": DROPOUT_RATE, "ACTIVATION": ACTIVATION, "USE_POSITIONAL": USE_POSITIONAL, "MAX_LENGTH": MAX_LENGTH, "NUM_SAMPLES": NUM_SAMPLES, "EPOCHS": EPOCHS}
    
    # Write hyperparameters to file
    tracker = ModelTracker(hyperparameter_filenamepath, hyperparameters_)
    tracker.writer()

    # Print hyperparameters
    for key, value in hyperparameters_.items():
        print(f"{key}: {value}")

    # # The dataset and text processing
    # The text sentences are almost clean, they are simple plain text, so we only need to remove dots that are not a end of sentence symbol and duplicated white spaces.
    # The following functions will apply the cleaning mentioned previously:
    def preprocess_text_nonbreaking(corpus, non_breaking_prefixes):
        corpus_cleaned = corpus
        # Add the string $$$ before the non breaking prefixes
        # To avoid remove dots from some words
        for prefix in non_breaking_prefixes:
            corpus_cleaned = corpus_cleaned.replace(prefix, prefix + "$$$")
        # Remove dots not at the end of a sentence
        corpus_cleaned = re.sub(r"\.(?=[0-9]|[a-z]|[A-Z])", ".$$$", corpus_cleaned)
        # Remove the $$$ mark
        corpus_cleaned = re.sub(r"\.\$\$\$", "", corpus_cleaned)
        # Rmove multiple white spaces
        corpus_cleaned = re.sub(r"  +", " ", corpus_cleaned)

        return corpus_cleaned


    # ## Loading the dataset
    # Loading the list of non breaking prefixes for the english and the spanish sentences


    with open(DATA_PATH + "/nonbreaking_prefix.en", mode="r", encoding="utf-8") as f:
        non_breaking_prefix_en = f.read()
    if LANG == "DE":
        with open(DATA_PATH + "/nonbreaking_prefix.de", mode="r", encoding="utf-8") as f:
            non_breaking_prefix_es = f.read()
    else:
        with open(DATA_PATH + "/nonbreaking_prefix.es", mode="r", encoding="utf-8") as f:
            non_breaking_prefix_es = f.read()

    non_breaking_prefix_en = non_breaking_prefix_en.split("\n")
    non_breaking_prefix_en = [" " + pref + "." for pref in non_breaking_prefix_en]
    non_breaking_prefix_es = non_breaking_prefix_es.split("\n")
    non_breaking_prefix_es = [" " + pref + "." for pref in non_breaking_prefix_es]


    # Load the dataset into a pandas dataframe and apply the preprocess function to the input and target columns.
    # Load the dataset: sentence in english, sentence in spanish
    df = pd.read_csv(
        train_filenamepath, sep="\t", header=None, names=[INPUT_COLUMN, TARGET_COLUMN], usecols=[0, 1], nrows=NUM_SAMPLES
    )
    # Preprocess the input data
    input_data = df[INPUT_COLUMN].apply(lambda x: preprocess_text_nonbreaking(x, non_breaking_prefix_en)).tolist()
    # Preprocess and include the end of sentence token to the target text
    target_data = df[TARGET_COLUMN].apply(lambda x: preprocess_text_nonbreaking(x, non_breaking_prefix_es)).tolist()

    print("Number of sentences: ", len(input_data))

    # Delete the dataframe and release the memory (if it is possible)
    del df
    gc.collect()

    # Tokenize and pad the sequences
    token_manager.generate_tokens(input_data, target_data, MAX_LENGTH, MAX_VOCAB_SIZE, LANG)
    
    # Input sequences
    encoder_inputs, tokenizer_inputs, num_words_inputs, sos_token_input, eos_token_input, del_idx_inputs = token_manager.encoder_inputs, token_manager.tokenizer_inputs, token_manager.num_words_inputs, token_manager.sos_token_input, token_manager.eos_token_input, token_manager.del_idx_inputs

    decoder_outputs, tokenizer_outputs, num_words_output, sos_token_output, eos_token_output, del_idx_outputs = token_manager.decoder_outputs, token_manager.tokenizer_outputs, token_manager.num_words_output, token_manager.sos_token_output, token_manager.eos_token_output, token_manager.del_idx_outputs

    print("Size of Input Vocabulary: ", num_words_inputs)
    print("Size of Output Vocabulary: ", num_words_output)


    # # Create the batch data generator
    # Define a dataset
    dataset = tf.data.Dataset.from_tensor_slices((encoder_inputs, decoder_outputs))
    dataset = dataset.shuffle(len(input_data), reshuffle_each_iteration=True).batch(BATCH_SIZE, drop_remainder=True)

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # # Building a Transformer
    def scaled_dot_product_attention(queries, keys, values, mask):
        # Calculate the dot product, QK_transpose
        product = tf.matmul(queries, keys, transpose_b=True)
        # Get the scale factor
        keys_dim = tf.cast(tf.shape(keys)[-1], tf.float32)
        # Apply the scale factor to the dot product
        scaled_product = product / tf.math.sqrt(keys_dim)
        # Apply masking when it is requiered
        if mask is not None:
            scaled_product += mask * -1e9
        # dot product with Values
        attention = tf.matmul(tf.nn.softmax(scaled_product, axis=-1), values)

        return attention


    # ## Multi-Head Attention
    class MultiHeadAttention(layers.Layer):
        def __init__(self, n_heads):
            super(MultiHeadAttention, self).__init__()
            self.n_heads = n_heads

        def build(self, input_shape):
            self.d_model = input_shape[-1]
            assert self.d_model % self.n_heads == 0
            # Calculate the dimension of every head or projection
            self.d_head = self.d_model // self.n_heads
            # Set the weight matrices for Q, K and V
            self.query_lin = layers.Dense(units=self.d_model)
            self.key_lin = layers.Dense(units=self.d_model)
            self.value_lin = layers.Dense(units=self.d_model)
            # Set the weight matrix for the output of the multi-head attention W0
            self.final_lin = layers.Dense(units=self.d_model)

        def split_proj(self, inputs, batch_size):  # inputs: (batch_size, seq_length, d_model)
            # Set the dimension of the projections
            shape = (batch_size, -1, self.n_heads, self.d_head)
            # Split the input vectors
            splited_inputs = tf.reshape(inputs, shape=shape)  # (batch_size, seq_length, nb_proj, d_proj)
            return tf.transpose(splited_inputs, perm=[0, 2, 1, 3])  # (batch_size, nb_proj, seq_length, d_proj)

        def call(self, queries, keys, values, mask):
            # Get the batch size
            batch_size = tf.shape(queries)[0]
            # Set the Query, Key and Value matrices
            queries = self.query_lin(queries)
            keys = self.key_lin(keys)
            values = self.value_lin(values)
            # Split Q, K y V between the heads or projections
            queries = self.split_proj(queries, batch_size)
            keys = self.split_proj(keys, batch_size)
            values = self.split_proj(values, batch_size)
            # Apply the scaled dot product
            attention = scaled_dot_product_attention(queries, keys, values, mask)
            # Get the attention scores
            attention = tf.transpose(attention, perm=[0, 2, 1, 3])
            # Concat the h heads or projections
            concat_attention = tf.reshape(attention, shape=(batch_size, -1, self.d_model))
            # Apply W0 to get the output of the multi-head attention
            outputs = self.final_lin(concat_attention)

            return outputs


    # # Positional Encoding
    class PositionalEncoding(layers.Layer):
        def __init__(self):
            super(PositionalEncoding, self).__init__()

        def get_angles(self, pos, i, d_model):  # pos: (seq_length, 1) i: (1, d_model)
            angles = 1 / np.power(10000.0, (2 * (i // 2)) / np.float32(d_model))
            return pos * angles  # (seq_length, d_model)

        def call(self, inputs):
            # input shape batch_size, seq_length, d_model
            seq_length = inputs.shape.as_list()[-2]
            d_model = inputs.shape.as_list()[-1]
            # Calculate the angles given the input
            angles = self.get_angles(np.arange(seq_length)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
            # Calculate the positional encodings
            angles[:, 0::2] = np.sin(angles[:, 0::2])
            angles[:, 1::2] = np.cos(angles[:, 1::2])
            # Expand the encodings with a new dimension
            pos_encoding = angles[np.newaxis, ...]

            return inputs + tf.cast(pos_encoding, tf.float32)


    # # The Encoder
    class EncoderLayer(layers.Layer):
        def __init__(self, FFN_units, n_heads, dropout_rate, activation):
            super(EncoderLayer, self).__init__()
            # Hidden units of the feed forward component
            self.FFN_units = FFN_units
            # Set the number of projectios or heads
            self.n_heads = n_heads
            # Dropout rate
            self.dropout_rate = dropout_rate
            # activation layer
            self.activation = activation

        def build(self, input_shape):
            self.d_model = input_shape[-1]
            # Build the multihead layer
            self.multi_head_attention = MultiHeadAttention(self.n_heads)
            self.dropout_1 = layers.Dropout(rate=self.dropout_rate)
            # Layer Normalization
            self.norm_1 = layers.LayerNormalization(epsilon=1e-6)
            # Fully connected feed forward layer
            self.ffn1_relu = layers.Dense(units=self.FFN_units, activation=self.activation)
            self.ffn2 = layers.Dense(units=self.d_model)
            self.dropout_2 = layers.Dropout(rate=self.dropout_rate)
            # Layer normalization
            self.norm_2 = layers.LayerNormalization(epsilon=1e-6)

        def call(self, inputs, mask, training):
            # Forward pass of the multi-head attention
            attention = self.multi_head_attention(inputs, inputs, inputs, mask)
            attention = self.dropout_1(attention, training=training)
            # Call to the residual connection and layer normalization
            attention = self.norm_1(attention + inputs)
            # Call to the FC layer
            outputs = self.ffn1_relu(attention)
            outputs = self.ffn2(outputs)
            outputs = self.dropout_2(outputs, training=training)
            # Call to residual connection and the layer normalization
            outputs = self.norm_2(outputs + attention)

            return outputs


    class Encoder(layers.Layer):
        def __init__(self, n_layers, FFN_units, n_heads, dropout_rate, vocab_size, d_model, activation, name="encoder", use_positional = True):
            super(Encoder, self).__init__(name=name)
            self.n_layers = n_layers
            self.d_model = d_model
            # The embedding layer
            self.embedding = layers.Embedding(vocab_size, d_model)
            # Positional encoding layer
            self.dropout = layers.Dropout(rate=dropout_rate)
            # Stack of n layers of multi-head attention and FC
            self.enc_layers = [EncoderLayer(FFN_units, n_heads, dropout_rate, activation=activation) for _ in range(n_layers)]
            self.use_positional = use_positional
            if self.use_positional: 
                self.pos_encoding = PositionalEncoding()
            else:
                print('Skipping positional encoder in Encoder Layer')
                self.pos_encoding = layers.Lambda(lambda x: x)

        def call(self, inputs, mask, training):
            # Get the embedding vectors
            outputs = self.embedding(inputs)
            # Scale the embeddings by sqrt of d_model
            outputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
            # Positional encodding
            outputs = self.pos_encoding(outputs)
            outputs = self.dropout(outputs, training)
            # Call the stacked layers
            for i in range(self.n_layers):
                outputs = self.enc_layers[i](outputs, mask, training)

            return outputs


    # # The Decoder
    class DecoderLayer(layers.Layer):
        def __init__(self, FFN_units, n_heads, dropout_rate, activation):
            super(DecoderLayer, self).__init__()
            self.FFN_units = FFN_units
            self.n_heads = n_heads
            self.dropout_rate = dropout_rate
            self.activation = activation

        def build(self, input_shape):
            self.d_model = input_shape[-1]

            # Self multi head attention, causal attention
            self.multi_head_causal_attention = MultiHeadAttention(self.n_heads)
            self.dropout_1 = layers.Dropout(rate=self.dropout_rate)
            self.norm_1 = layers.LayerNormalization(epsilon=1e-6)

            # Multi head attention, encoder-decoder attention
            self.multi_head_enc_dec_attention = MultiHeadAttention(self.n_heads)
            self.dropout_2 = layers.Dropout(rate=self.dropout_rate)
            self.norm_2 = layers.LayerNormalization(epsilon=1e-6)

            # Feed foward
            self.ffn1_relu = layers.Dense(units=self.FFN_units, activation=self.activation)
            self.ffn2 = layers.Dense(units=self.d_model)
            self.dropout_3 = layers.Dropout(rate=self.dropout_rate)
            self.norm_3 = layers.LayerNormalization(epsilon=1e-6)

        def call(self, inputs, enc_outputs, mask_1, mask_2, training):
            # Call the masked causal attention
            attention = self.multi_head_causal_attention(inputs, inputs, inputs, mask_1)
            attention = self.dropout_1(attention, training)
            # Residual connection and layer normalization
            attention = self.norm_1(attention + inputs)
            # Call the encoder-decoder attention
            attention_2 = self.multi_head_enc_dec_attention(attention, enc_outputs, enc_outputs, mask_2)
            attention_2 = self.dropout_2(attention_2, training)
            # Residual connection and layer normalization
            attention_2 = self.norm_2(attention_2 + attention)
            # Call the Feed forward
            outputs = self.ffn1_relu(attention_2)
            outputs = self.ffn2(outputs)
            outputs = self.dropout_3(outputs, training)
            # Residual connection and layer normalization
            outputs = self.norm_3(outputs + attention_2)

            return outputs


    class Decoder(layers.Layer):
        def __init__(self, n_layers, FFN_units, n_heads, dropout_rate, vocab_size, d_model, activation, name="decoder", use_positional=True):
            super(Decoder, self).__init__(name=name)
            self.d_model = d_model
            self.n_layers = n_layers
            # Embedding layer
            self.embedding = layers.Embedding(vocab_size, d_model)
            self.dropout = layers.Dropout(rate=dropout_rate)
            # Stacked layers of multi-head attention and feed forward
            self.dec_layers = [DecoderLayer(FFN_units, n_heads, dropout_rate, activation=activation) for _ in range(n_layers)]
            
            self.use_positional = use_positional
            # Positional encoding layer
            if self.use_positional: 
                self.pos_encoding = PositionalEncoding()
            else:
                print('Skipping positional encoder in Decoder Layer')
                self.pos_encoding = layers.Lambda(lambda x: x)

        def call(self, inputs, enc_outputs, mask_1, mask_2, training):
            # Get the embedding vectors
            outputs = self.embedding(inputs)
            # Scale by sqrt of d_model
            outputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
            # Positional encodding
            outputs = self.pos_encoding(outputs)
            outputs = self.dropout(outputs, training)
            # Call the stacked layers
            for i in range(self.n_layers):
                outputs = self.dec_layers[i](outputs, enc_outputs, mask_1, mask_2, training)

            return outputs


    # # Transformer
    class Transformer(tf.keras.Model):
        def __init__(
            self, vocab_size_enc, vocab_size_dec, d_model, n_layers, FFN_units, n_heads, dropout_rate, activation = 'relu',name="transformer", use_positional = True
        ):
            super(Transformer, self).__init__(name=name)
            # Build the encoder
            self.encoder = Encoder(n_layers, FFN_units, n_heads, dropout_rate, vocab_size_enc, d_model, activation=activation, use_positional=use_positional)
            # Build the decoder
            self.decoder = Decoder(n_layers, FFN_units, n_heads, dropout_rate, vocab_size_dec, d_model, activation=activation, use_positional=use_positional)
            # build the linear transformation and softmax function
            self.last_linear = layers.Dense(units=vocab_size_dec, name="lin_ouput")

        def create_padding_mask(self, seq):  # seq: (batch_size, seq_length)
            # Create the mask for padding
            mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
            return mask[:, tf.newaxis, tf.newaxis, :]

        def create_look_ahead_mask(self, seq):
            # Create the mask for the causal attention
            seq_len = tf.shape(seq)[1]
            look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
            return look_ahead_mask

        def call(self, enc_inputs, dec_inputs, training):
            # Create the padding mask for the encoder
            enc_mask = self.create_padding_mask(enc_inputs)
            # Create the mask for the causal attention
            dec_mask_1 = tf.maximum(self.create_padding_mask(dec_inputs), self.create_look_ahead_mask(dec_inputs))
            # Create the mask for the encoder-decoder attention
            dec_mask_2 = self.create_padding_mask(enc_inputs)
            # Call the encoder
            enc_outputs = self.encoder(enc_inputs, enc_mask, training)
            # Call the decoder
            dec_outputs = self.decoder(dec_inputs, enc_outputs, dec_mask_1, dec_mask_2, training)
            # Call the Linear and Softmax functions
            outputs = self.last_linear(dec_outputs)

            return outputs


    # # Training the Transformer model
    # $lrate = d_{model}^{-0.5}*min(step\_num^{-0.5}, step\_num*warmup\_steps^{-1.5})$
    def loss_function(target, pred):
        mask = tf.math.logical_not(tf.math.equal(target, 0))
        loss_ = loss_object(target, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)


    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, d_model, warmup_steps=4000):
            super(CustomSchedule, self).__init__()

            self.d_model = tf.cast(d_model, tf.float32)
            self.warmup_steps = warmup_steps

        def __call__(self, step):
            arg1 = tf.math.rsqrt(step)
            arg2 = step * (self.warmup_steps ** -1.5)

            return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


    # Train function
    # - For every iteration on the batch generator that produce batch size inputs and outputs
    # - Get the input sequence from 0 to length-1 and the actual outputs from 1 to length, the next word expected at every sequence step.
    # - Call the transformer to get the predictions
    # - Calculate the loss function between the real outputs and the predictions
    # - Apply the gradients to update the weights in the model
    # - Calculate the mean loss and the accuracy for the batch data
    # - Show some results and save the model in every epoch


    def main_train(dataset, transformer, n_epochs, epoch_tracker: EpochTracker, print_every=50):
        """Train the transformer model for n_epochs using the data generator dataset"""
        losses = []
        accuracies = []
        epoch_times = {}
        # In every epoch
        for epoch in range(epoch_tracker.last_epoch, n_epochs):
            
            print(f"{epoch_tracker.last_epoch=}, {n_epochs=}")
            print("Starting epoch {}".format(epoch + 1))
            start = time.time()
            # Reset the losss and accuracy calculations
            train_loss.reset_states()
            train_accuracy.reset_states()
            # Get a batch of inputs and targets
            batch_times = []
            batch_t0 = time.time()
            tqdm_ = tqdm(dataset, position=0, leave=True)
            dataset_len = len(tqdm_)
            for (batch, (enc_inputs, targets)) in enumerate(tqdm_):    
                # Set the decoder inputs
                dec_inputs = targets[:, :-1]
                # Set the target outputs, right shifted
                dec_outputs_real = targets[:, 1:]
                with tf.GradientTape() as tape:
                    # Call the transformer and get the predicted output
                    predictions = transformer(enc_inputs, dec_inputs, True)
                    # Calculate the loss
                    loss = loss_function(dec_outputs_real, predictions)
                # Update the weights and optimizer
                gradients = tape.gradient(loss, transformer.trainable_variables)
                optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
                # Save and store the metrics
                train_loss(loss)
                train_accuracy(dec_outputs_real, predictions)

                if batch % print_every == 0 or batch == (dataset_len - 1):
                    losses.append(train_loss.result())
                    accuracies.append(train_accuracy.result())
                    # Stop batch time counter, append to list
                    batch_t1 = time.time() - batch_t0
                    batch_t0 = time.time()
                    batch_times.append(batch_t1)
                    print(
                        "Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}".format(
                            epoch + 1, batch, train_loss.result(), train_accuracy.result()
                        ),
                        flush=True,
                    )

            epoch_t1 = time.time() - start
            epoch_times[epoch_t1] = batch_times
            # update epoch_tracker info
            epoch_tracker.epoch_data[epoch + 1] = {
                "losses": [float(i) for i in losses],
                "accuracies": [float(i) for i in accuracies],
                "epoch_times": epoch_times,
            }
            epoch_tracker.writer()  # write data to file

            # Checkpoint the model on every epoch
            ckpt_save_path = ckpt_manager.save()
            print("Saving checkpoint for epoch {} in {}".format(epoch + 1, ckpt_save_path))
            print("Time for 1 epoch: {} secs\n".format(epoch_t1))

        return losses, accuracies


    # Clean the session
    tf.keras.backend.clear_session()
    # Create the Transformer model
    transformer = Transformer(
        vocab_size_enc=num_words_inputs,
        vocab_size_dec=num_words_output,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        FFN_units=FFN_UNITS,
        n_heads=N_HEADS,
        dropout_rate=DROPOUT_RATE,
        activation = ACTIVATION,
        use_positional=USE_POSITIONAL
    )

    # Define a categorical cross entropy loss
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
    # Define a metric to store the mean loss of every epoch
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    # Define a matric to save the accuracy in every epoch
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
    # Create the scheduler for learning rate decay
    leaning_rate = CustomSchedule(D_MODEL)
    # Create the Adam optimizer
    optimizer = tf.keras.optimizers.Adam(leaning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)


    # Create the Checkpoint
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Last checkpoint restored.")

    # Initiate Epoch Tracker
    epoch_tracker = EpochTracker(epoch_filenamepath)

    # Train the model
    main_train(dataset, transformer, EPOCHS, epoch_tracker, 100)

    # plot
    metrics = epoch_tracker.epoch_data
    max_epoch = max([int(k) for k in metrics.keys()])

    losses, accuracies = metrics[max_epoch]['losses'], metrics[max_epoch]['accuracies']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    # plot some data
    ax1.plot(losses, label="loss")
    # plt.plot(results.history['val_loss'], label='val_loss')
    ax1.set_title("Training Loss")
    ax1.legend()
    # accuracies
    ax2.plot(accuracies, label="acc")
    # plt.plot(results.history['val_accuracy_fn'], label='val_acc')
    ax2.set_title("Training Accuracy")
    ax2.legend()
    plt.show()
    plt.savefig(f"images/fig-{MODEL_NAME}.png")


    # # Make predictions
    def predict(inp_sentence, tokenizer_in, tokenizer_out, target_max_len):
        # Tokenize the input sequence using the tokenizer_in
        inp_sentence = sos_token_input + tokenizer_in.encode(inp_sentence) + eos_token_input
        enc_input = tf.expand_dims(inp_sentence, axis=0)

        # Set the initial output sentence to sos
        out_sentence = sos_token_output
        # Reshape the output
        output = tf.expand_dims(out_sentence, axis=0)

        # For max target len tokens
        for _ in range(target_max_len):
            # Call the transformer and get the logits
            predictions = transformer(enc_input, output, False)  # (1, seq_length, VOCAB_SIZE_ES)
            # Extract the logists of the next word
            prediction = predictions[:, -1:, :]
            # The highest probability is taken
            predicted_id = tf.cast(tf.argmax(prediction, axis=-1), tf.int32)
            # Check if it is the eos token
            if predicted_id == eos_token_output:
                return tf.squeeze(output, axis=0)
            # Concat the predicted word to the output sequence
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0)

    def translate(sentence):
        # Get the predicted sequence for the input sentence
        output = predict(sentence, tokenizer_inputs, tokenizer_outputs, MAX_LENGTH).numpy()
        # Transform the sequence of tokens to a sentence
        predicted_sentence = tokenizer_outputs.decode([i for i in output if i < sos_token_output])

        return predicted_sentence

    translations = {}
    # Show some translations
    sentence = "you should pay for it."
    print("Input sentence: {}".format(sentence))
    predicted_sentence = translate(sentence)
    print("Output sentence: {}".format(predicted_sentence))
    translations[sentence] = predicted_sentence

    # Show some translations
    sentence = "we have no extra money."
    print("Input sentence: {}".format(sentence))
    predicted_sentence = translate(sentence)
    print("Output sentence: {}".format(predicted_sentence))
    translations[sentence] = predicted_sentence

    # Next, let's predict some new sentences on diferent topics:
    # Show some translations
    sentence = "This is a problem to deal with."
    print("Input sentence: {}".format(sentence))
    predicted_sentence = translate(sentence)
    print("Output sentence: {}".format(predicted_sentence))
    translations[sentence] = predicted_sentence


    # Show some translations
    sentence = "This is a really powerful method!"
    print("Input sentence: {}".format(sentence))
    predicted_sentence = translate(sentence)
    print("Output sentence: {}".format(predicted_sentence))
    translations[sentence] = predicted_sentence

    # Show some translations
    sentence = "This is an interesting course about Natural Language Processing"
    print("Input sentence: {}".format(sentence))
    predicted_sentence = translate(sentence)
    print("Output sentence: {}".format(predicted_sentence))
    translations[sentence] = predicted_sentence

    # Show some translations
    sentence = "Jerry liked to look at paintings while eating garlic ice cream."
    print("Input sentence: {}".format(sentence))
    predicted_sentence = translate(sentence)
    print("Output sentence: {}".format(predicted_sentence))
    translations[sentence] = predicted_sentence

    # Show some translations
    sentence = "The irony of the situation wasn't lost on anyone in the room."
    print("Input sentence: {}".format(sentence))
    predicted_sentence = translate(sentence)
    print("Output sentence: {}".format(predicted_sentence))
    translations[sentence] = predicted_sentence

    # Show some translations
    sentence = "Facebook plans to make a dramatic break with its past by rebranding the company next week, according to a report."
    print("Input sentence: {}".format(sentence))
    predicted_sentence = translate(sentence)
    print("Output sentence: {}".format(predicted_sentence))
    translations[sentence] = predicted_sentence
    
      # Show some translations
    sentence = 'iPhone assembler Foxconn has revealed three prototype electric vehicles as part of its effort to become a major player in the automotive industry. "Our biggest challenge is we don’t know how to make cars," Foxconn chairman Young Liu said at the event held Monday.'
    print("Input sentence: {}".format(sentence))
    predicted_sentence = translate(sentence)
    print("Output sentence: {}".format(predicted_sentence))
    translations[sentence] = predicted_sentence

    # create json object from dictionary
    json_ = json.dumps(translations)
    with open(f"translations/file-{MODEL_NAME}.json", 'w') as f:
        f.write(json_)