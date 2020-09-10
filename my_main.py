# This model make prediction only based on title.
import os
import traceback
from tqdm import tqdm
from dataset import MOSTPortal
import json
from dataloader import TripletDataGenerator
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.nn as nn
import constants as c
from model import AttentiveMultiView

if torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'

print("using cuda:", torch.cuda.is_available())

with open("word_dict.json", "r") as fin:
    word_dict = json.load(fin)
# word_dict = None

ds = MOSTPortal("QN_fixed_posts.csv", "QN_fixed_trans.csv", train_test_set="train_test_set.json", is_testing=False)
training_model = AttentiveMultiView(len(word_dict))
saved_ckpt = torch.load("./checkpoints/init.pt")
training_model = saved_ckpt['model']
training_model.to(device)

def fit(model, saved_ckpt=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    error = nn.BCELoss()
    if saved_ckpt is not None:
        model = saved_ckpt['model']
        optimizer.load_state_dict(saved_ckpt['optimizer_state_dict'])
    model.train()
    for epoch in range(0, c.EPOCHS):
        torch_ds = TripletDataGenerator(ds, "/home/nghiatd/workspace/Content-based/ws/QNPortal/posts", 
                                        word_dict=word_dict, batch_size=c.MINIBATCH_SIZE, npratio=c.npratio)
        train_loader = DataLoader(torch_ds, shuffle=False, num_workers=8)
        print(f"EPOCH {epoch+1}/{c.EPOCHS}")
        correct = 0
        total_counter = 0
        with tqdm(total=torch_ds.datasize) as pbar:
            for batch_idx, batch_sample in enumerate(train_loader):
                
                browsed_news, candidate_news, batch_labels = batch_sample
                
                y_batch = batch_labels
                var_browsed_news = Variable(browsed_news[0]).to(device)
                var_candidate_news = Variable(candidate_news[0]).to(device)
                var_y_batch = Variable(y_batch[0]).float().to(device)

                optimizer.zero_grad()
                output = model(var_browsed_news, var_candidate_news)
                
                loss = error(output, var_y_batch)
                loss.backward()
                optimizer.step()

                # Total correct predictions
                predicted = torch.max(output.data.cpu(), 1)[1]
                target = torch.max(y_batch[0], 1)[1]
                
                correct += (predicted == target).sum()
                real_batch_size = len(y_batch[0])
                total_counter += real_batch_size
                pbar.update(real_batch_size)
                if batch_idx % 10 == 0 or batch_idx == len(train_loader)-1:
                    display_string = 'Loss: {:.6f} | Accuracy:{:.3f}%'.format(loss.data.item(), float(correct*100) / total_counter)
                    pbar.set_description(display_string)
                    pbar.refresh()
        # Saving checkpoint
        torch.save({
            'model': model,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, f"checkpoints/{epoch+1}.pt")
fit(training_model, saved_ckpt)