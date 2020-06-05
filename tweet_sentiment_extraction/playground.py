import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch import utils
import pandas as pd

from tweet_sentiment_extraction.TweetDataset import TweetDataset
import random

# Testing the Dataset
from tweet_sentiment_extraction.TweetExtractorModel import TweetExtractorModel

train_data = pd.read_csv('data/train.csv')
train_data, val_data = train_test_split(train_data, test_size=0.15)
dataset = TweetDataset(train_data, 50)

# Testing one forward pass
model = TweetExtractorModel()

# num = random.randint(0, dataset.__len__()-1)
# parameters = dataset.__getitem__(num)
# tokens = parameters['tokens']
# attn_mask = parameters['attention_mask']
# token_type_ids = parameters['token_type_ids']
# start = parameters['start']
# end = parameters['end']
# sentence = parameters['sentence']
# selected_text = parameters['selected_text']
# sentiment = parameters['sentiment']

# pred_start, pred_end = model.forward(tokens, attn_mask, token_type_ids)
# print(sentence + ': ' + selected_text + ': ' + sentiment)
# print(start)
# print(end)
# print(pred_start)
# print(pred_end)


def loss_fn(start_logits, end_logits, start_positions, end_positions):
    """
    Simple loss function based on logit loss of start and end.
    TODO(viman): Could be better, by actually calculating the iou
    """
    loss_fct = nn.CrossEntropyLoss()
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss = (start_loss + end_loss) / 2
    return total_loss


trainloader = utils.data.DataLoader(
    dataset,
    batch_size=64,
    shuffle=True
)

criterion = loss_fn
optimizer = optim.Adam(model.parameters(), amsgrad=True)

def train():
    device = torch.device("cpu")
    EPOCH_COUNT = 2
    for epoch in range(EPOCH_COUNT):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            start_label = data['start']
            end_label = data['end']
            input_ids = data['tokens']
            attn_mask = data['attention_mask']
            token_type_ids = data['token_type_ids']

            optimizer.zero_grad()

            # Reshaping since reading a batch introduces a "1" in between
            input_ids = torch.reshape(input_ids, (input_ids.shape[0], input_ids.shape[2]))
            attn_mask = torch.reshape(attn_mask, (attn_mask.shape[0], attn_mask.shape[2]))
            token_type_ids = torch.reshape(token_type_ids, (token_type_ids.shape[0], token_type_ids.shape[2]))

            outputs = model(input_ids, attn_mask, token_type_ids)
            start_logits = outputs[0]
            end_logits = outputs[1]

            loss = criterion(start_logits, end_logits, start_label, end_label)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 0: # Print after every 1000 minibatches
                print('Epoch: %d, iteration: %d loss: %.3f' % (epoch, i, running_loss))
                running_loss = 0

    print('Finished Training')

train()









