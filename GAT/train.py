import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import load_data, accuracy
from models import GAT

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--data_path',type=string, default='./cora/',help='cora data file')
parser.add_argument('--cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
#parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--log_interval', type=int, default=100, help='interval to test the model and log accuracy')
parser.add_argument('--save_interval', type=int, default=100, help='interval to save the model data')
#parser.add_argument('--log_interval', type=int, default=40, help='interval to save loss and judge the model')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=10, help='Patience')


args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

model = GAT(nfeat=features.shape[1], 
            nhid=args.hidden, 
            nclass=int(labels.max()) + 1, 
            dropout=args.dropout, 
            nheads=args.nb_heads, 
            alpha=args.alpha)
optimizer = optim.Adam(model.parameters(), 
                    lr=args.lr, 
                    weight_decay=args.weight_decay)
if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

features, adj, labels = Variable(features), Variable(adj), Variable(labels)

def train(epoch):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    loss_data=loss_train.data.item()
    acc_data=acc_train.data.item()
    return loss_data, acc_data

def compute_test(epoch):
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    loss_data=loss_test.data.item()
    acc_data=acc_test.data.item()
    print("epoch {:>5d}||Test set results:".format(),
          "loss= {:.4f}".format(loss_data),
          "accuracy= {:.4f}".format(acc_data))
    return loss_data



if __name__=='__main__':
    t_total = time.time()
    loss_values = []
    bad_counter = 0
    best = args.epochs + 1
    best_epoch = 0
    for epoch in range(args.epochs):
        ifepoch and (epoch%args.log_interval==0):
            loss_values.append(train(epoch)[0])
            if loss_values[-1] < best:
                best = loss_values[-1]
                best_epoch = epoch
                bad_counter = 0
            else:
                bad_counter += 1
            if bad_counter == args.patience:
                break
            compute_test(epoch)
    #save the model data

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Restore best model
    print('Loading {}th epoch'.format(best_epoch))
    model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

    # Testing
    compute_test(epoch)