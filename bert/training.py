import argparse
from datetime import time

import torch

from bert.DataLoader import load_data
from bert.SentimentClassifier import SentimentClassifier
import torch.nn as nn
import torch.optim as optim

net = SentimentClassifier(freeze_bert=True)

criterion = nn.BCEWithLogitsLoss()
opti = optim.Adam(net.parameters(), lr=2e-5)


def get_accuracy_from_logits(logits, labels):
    probs = torch.sigmoid(logits.unsqueeze(-1))
    soft_probs = (probs > 0.5).long()
    acc = (soft_probs.squeeze() == labels).float().mean()
    return acc


def evaluate(net, criterion, dataloader, args):
    net.eval()
    mean_acc, mean_loss = 0, 0
    count = 0

    with torch.no_grad():
        for seq, attn_masks, labels in dataloader:
            # seq, attn_masks, labels = seq.cuda(args.gpu), attn_masks.cuda(args.gpu), labels.cuda(args.gpu)
            logits = net(seq, attn_masks)
            mean_loss += criterion(logits.squeeze(-1), labels.float()).item()
            mean_acc += get_accuracy_from_logits(logits, labels)
            count += 1

    return mean_acc / count, mean_loss / count


def train(net, criterion, opti, train_loader, val_loader, args):
    best_acc = 0
    for ep in range(args.max_eps):
        print(ep)
        ctr = 1
        for it, (seq, attn_masks, labels) in enumerate(train_loader):
            # Clear gradients
            opti.zero_grad()
            # Converting these to cuda tensors
            # seq, attn_masks, labels = seq.cuda(args.gpu), attn_masks.cuda(args.gpu), labels.cuda(args.gpu)

            # Obtaining the logits from the model
            logits = net(seq, attn_masks)

            # Computing loss
            loss = criterion(logits.squeeze(-1), labels.float())

            # Backpropagating the gradients
            loss.backward()

            # Optimization step
            opti.step()

            if (it + 1) % args.print_every == 0:
                acc = get_accuracy_from_logits(logits, labels)
                print("Iteration {} of epoch {} complete. Loss : {} Accuracy : {}"
                      .format(it + 1, ep + 1, loss.item(), acc))
            print('Processed {} batches'.format(ctr))
            ctr = ctr + 1

        val_acc, val_loss = evaluate(net, criterion, val_loader, args)
        print("Epoch {} complete! Validation Accuracy : {}, Validation Loss : {}".format(ep, val_acc, val_loss))
        if val_acc > best_acc:
            print("Best validation accuracy improved from {} to {}, saving model...".format(best_acc, val_acc))
            best_acc = val_acc
            # TODO(viman): Enable this when ready to save
            # torch.save(net.state_dict(), 'Models/sstcls_{}_freeze_{}.dat'.format(ep, args.freeze_bert))

parser = argparse.ArgumentParser()
parser.add_argument('-gpu', type=int, default=0)
parser.add_argument('-freeze_bert', action='store_true')
parser.add_argument('-maxlen', type=int, default=25)
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-lr', type=float, default=2e-5)
parser.add_argument('-print_every', type=int, default=100)
parser.add_argument('-max_eps', type=int, default=2)
args = parser.parse_args()

print("Let the training begin")
train_loader, val_loader = load_data()
train(net, criterion, opti, train_loader, val_loader, args)
# TODO(viman): Look how to evaluate
# net.eval()
print("Done")
