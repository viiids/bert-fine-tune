import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizer


class TweetDataset(Dataset):
    def __init__(self, data, maxlen):

        # Store the contents of the file in a pandas dataframe
        self.df = data

        # Initialize the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.maxlen = maxlen

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """
        Returns the token_ids_tensors, attn_mask for the item and text denoting the sentiment.

        :param index:
        :return:
        """

        # Selecting the sentence and label at the specified index in the data frame
        orig_sentence = self.df.iloc[index]['text']
        sentiment = self.df.iloc[index]['sentiment']
        selected_text = self.df.iloc[index]['selected_text']

        # Preprocessing the text to be suitable for BERT

        # Encode the sentence. Does the following:
        # 1. Inserting the CLS and SEP token in the beginning and end of the sentence
        # 2. Generates attention mask
        # 3. Generate token_type_ids used to differentiate first part of the sentence from the second
        encoded_dict = self.tokenizer.encode_plus(
            sentiment,
            orig_sentence,
            max_length=self.maxlen,
            truncation_strategy='only_second',
            pad_to_max_length=True,
            return_tensors='pt',
            return_token_type_ids=True,
            return_attention_mask=True
        )
        tokens = encoded_dict['input_ids']
        token_type_ids = encoded_dict['token_type_ids']
        attn_mask = encoded_dict['attention_mask']

        # Determine the beginning and end of the sentence
        def phrase_start_finder(sentence, phrase):
            if phrase not in sentence:
                raise ValueError('s2 not substring of s1')
            start = sentence.find(phrase)
            return len(sentence[:start].strip().split(' '))

        def phrase_end_finder(sentence, phrase):
            if phrase not in sentence:
                raise ValueError('s2 not substring of s1')
            return phrase_start_finder(sentence, phrase) + len(phrase.strip().split(' ')) - 1

        start = phrase_start_finder(orig_sentence, selected_text)
        end = phrase_end_finder(orig_sentence, selected_text)

        return {
            'tokens': tokens,
            'attention_mask': attn_mask,
            'token_type_ids': token_type_ids,
            'start': float(start),
            'end': float(end),
            'sentence': orig_sentence,
            'selected_text': selected_text,
            'sentiment': sentiment
        }
