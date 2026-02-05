import os
import torch


def save_model( model, optimizer, epoch,file_name):
    model_dir = os.path.dirname(file_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir,exist_ok=True)
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
    torch.save(state, file_name)


