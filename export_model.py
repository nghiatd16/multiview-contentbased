import os
import torch
if os.path.isfile("state_dict.pt"):
    os.remove("state_dict.pt")
saved_ckpt = torch.load("./checkpoints/40.pt")
# saved_ckpt = None
training_model = saved_ckpt['model']
training_model.to("cpu")
torch.save(training_model.state_dict(), "state_dict.pt")