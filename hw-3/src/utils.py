import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tqdm.auto import tqdm


class EpochTracker:
    def __init__(self, epoch_filenamepath) -> None:
        super().__init__()
        self.epoch_filenamepath = epoch_filenamepath
        self.epoch_data = dict()
        self.last_epoch, self.epoch_data = self.load_data()

    def load_data(self) -> int:

        """
        Loads epoch metrics and files from specified epoch filename
        self.epoch_filenamepath: define path for the epoch metrics and files
        """

        if os.path.exists(self.epoch_filenamepath):
            with open(self.epoch_filenamepath, "r") as f:
                try:
                    file_content = json.load(f)
                # in case file is empty
                except ValueError:
                    return 0, {}
                
                # check if file is correctly formatted
                if (type(file_content) != dict):
                    return 0, {}
                
                # check if there are any keys in json file, choose highest key value
                # else make starting epoch = 0
                if len(file_content.keys()) == 0:
                    return 0, {}
                else:
                    content_with_keys_as_int = {int(k): v for k,v in file_content.items()}
                    starting_epoch = max(content_with_keys_as_int)
                    return starting_epoch, content_with_keys_as_int
        else:
            path = Path(self.epoch_filenamepath)
            os.makedirs(path.parent.absolute(), exist_ok=True)
            return 0, {}        
    
    def writer(self):
        with open(self.epoch_filenamepath, 'w') as f:
            json.dump(self.epoch_data, f, indent=4)

class ModelTracker:
    def __init__(self, model_params_filenamepath, model_params) -> None:
        super().__init__()
        self.model_params_filenamepath = model_params_filenamepath 
        self.model_params = model_params

    def writer(self):
        with open(self.model_params_filenamepath, 'w') as f:
            json.dump(self.model_params, f, indent=4)

class TokenManager:
    def __init__(self):
        self.input_data = None
        self.target_data = None
        self.max_length = None
        self.vocab_size = None
        
        # Have the tokens been built before?
        self.is_built = False

        # Input Tokens
        self.encoder_inputs = None
        self.tokenizer_inputs = None
        self.num_words_inputs = None
        self.sos_token_input = None
        self.eos_token_input = None
        self.del_idx_inputs = None
        
        # Output Tokens
        self.decoder_outputs = None
        self.tokenizer_outputs = None
        self.num_words_output = None
        self.sos_token_output = None
        self.eos_token_output = None
        self.del_idx_outputs = None
        

        # # Tokenize the text data
    def subword_tokenize(self, corpus, vocab_size, max_length):
        # Create the vocabulary using Subword tokenization
        tokenizer_corpus = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(corpus, target_vocab_size=vocab_size)
        # Get the final vocab size, adding the eos and sos tokens
        num_words = tokenizer_corpus.vocab_size + 2
        # Set eos and sos token
        sos_token = [num_words - 2]
        eos_token = [num_words - 1]
        # Tokenize the corpus
        sentences = [sos_token + tokenizer_corpus.encode(sentence) + eos_token for sentence in corpus]
        # Identify the index of the sentences longer than max length
        idx_to_remove = [count for count, sent in enumerate(sentences) if len(sent) > max_length]
        # Pad the sentences
        sentences = tf.keras.preprocessing.sequence.pad_sequences(sentences, value=0, padding="post", maxlen=max_length)

        return sentences, tokenizer_corpus, num_words, sos_token, eos_token, idx_to_remove

    def generate_tokens(self, input_data, target_data, max_length, vocab_size):
        if (self.is_built == True):
            print("Model has been built previously")
            if(self.vocab_size == vocab_size) or (self.vocab_size is not None):
                print("Vocab Size exists and is the same as previous")
                if (self.max_length == max_length) or (self.max_length is not None):
                    print('Max length exists and is the same as previous')
            print("Token building passed ... everything looks OK")
            time.sleep(2)
        else:
            print("Wait...Building new tokens...")
            # input tokens
            self.encoder_inputs, self.tokenizer_inputs, self.num_words_inputs, self.sos_token_input, self.eos_token_input, self.del_idx_inputs = self.subword_tokenize(input_data, vocab_size, max_length)
            # output tokens
            self.decoder_outputs, self.tokenizer_outputs, self.num_words_output, self.sos_token_output, self.eos_token_output, self.del_idx_outputs = self.subword_tokenize(target_data, vocab_size, max_length)

            self.is_built = True
            self.input_data = input_data
            self.target_data = target_data
            self.vocab_size = vocab_size
            self.max_length = max_length