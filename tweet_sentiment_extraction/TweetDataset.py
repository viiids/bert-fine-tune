import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizer


class TweetDataset(Dataset):
    def __init__(self, filename, maxlen):

        # Store the contents of the file in a pandas dataframe
        self.df = pd.read_csv(filename)

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

        sentence = sentiment + ': ' + orig_sentence

        # Preprocessing the text to be suitable for BERT

        # Tokenize the sentence
        tokens = self.tokenizer.tokenize(sentence)

        # Inserting the CLS and SEP token in the beginning and end of the sentence
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))]  # Padding sentences
        else:
            tokens = tokens[:self.maxlen - 1] + ['[SEP]']  # Prunning the list to be of specified max length

        # Obtaining the indices of the tokens in the BERT Vocabulary
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids_tensor = torch.tensor(tokens_ids)  # Converting the list to a pytorch tensor

        # Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_mask = (tokens_ids_tensor != 0).long()

        attn_mask = attn_mask.view(1, -1)
        tokens_ids_tensor = tokens_ids_tensor.view(1, -1)
        print(attn_mask.shape)
        print(tokens_ids_tensor.shape)

        return tokens_ids_tensor, attn_mask, selected_text, orig_sentence, sentiment
