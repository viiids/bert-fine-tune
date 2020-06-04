from torch import nn

from tweet_sentiment_extraction.TweetDataset import TweetDataset
import random

# Testing the Dataset
from tweet_sentiment_extraction.TweetExtractorModel import TweetExtractorModel

dataset = TweetDataset('data/train.csv', 50)

# Testing the model
model = TweetExtractorModel()
num = random.randint(0, dataset.__len__()-1)
for i in range(num, num + 5):
    tokens, attn_masks, selected_text, orig_sentence, sentiment = dataset.__getitem__(i)
    print(orig_sentence + ': ' + selected_text + ': ' + sentiment)
    start, end = model.forward(tokens, attn_masks)
    print(start)
    print(end)



def loss_fn(start_logits, end_logits, start_positions, end_positions):
    """
    Adding up loss of specifying the start and end positions.
    TODO(viman): Could be better, by actually calculating the iou
    """
    loss_fct = nn.CrossEntropyLoss()
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss = (start_loss + end_loss)
    return total_loss




