# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# Modified by Sudeep Dasari
# --------------------------------------------------------
from __future__ import print_function

import torch
import numpy as np
import pickle
import utils
from voc_dataset import VOCDataset
from torch.utils.tensorboard import SummaryWriter
import notebook_util
import os
notebook_util.pick_gpu_lowest_memory()
def save_this_epoch(args, epoch):
        # TODO: Q2 check if model should be saved this epoch
    return epoch%args.save_freq==0


def save_model(epoch, model_name, model):
    # TODO: Q2 Implement code for model saving   
    PATH=os.path.join('.',model_name,'models',str(epoch)+'_.pt')
#     torch.save(model,PATH )
    #for parameters only
    torch.save(model.state_dict(), PATH)

# def save_conv1_output(epoch,model,outputs):
#     model.conv1.register_forward_hook()
    
#     filters = model.layers[1]
#     outputs.append()


def train(args, model, optimizer, scheduler=None, model_name='model'):
    # TODO: Q1.5 Initialize your visualizer here!
    path = model.__class__.__name__
    writer = SummaryWriter(os.path.join(path,'runs')) 
    # TODO: Q1.2 complete your dataloader in voc_dataset.py
    train_loader = utils.get_data_loader('voc', train=True, batch_size=args.batch_size, split='trainval')
    test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test')

    # Ensure model is in correct mode and on right device
    model.train()
    model = model.to(args.device)

    # TODO: Q1.4 Implement model training code!
    cnt = 0
    loss_criterion = torch.nn.BCEWithLogitsLoss()
    conv1_output={}
    for epoch in range(args.epochs):
        for batch_idx, (data, target, wgt) in enumerate(train_loader):
            # Get a batch of data
            data, target, wgt = data.to(args.device), target.to(args.device), wgt.to(args.device)
            optimizer.zero_grad()
            # Forward pass
            
            output = model(data)
            # Calculate the loss
            # TODO: your loss for multi-label clf?
            
            loss = loss_criterion(output,target)
#             loss = 0
            # Calculate gradient w.r.t the loss
            loss.backward()
            # Optimizer takes one step
            optimizer.step()
            # Log info
            if cnt % args.log_every == 0:
                # todo: add your visualization code
                print('Train Epoch: {} [{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, cnt, 100. * batch_idx / len(train_loader), loss.item()))
                writer.add_scalar('Loss/train', loss, cnt)
                for param in model.parameters():
                    writer.add_histogram('Histogram of Gradients', param.grad, cnt)
                writer.add_scalar('Learning Rate',scheduler.get_last_lr()[0],cnt)

                
            # Validation iteration
            if cnt % args.val_every == 0:
                model.eval()
                ap, map_ = utils.eval_dataset_map(model, args.device, test_loader)
                print('Test Epoch: {} [{} ({:.0f}%)]\tmAP: {:.6f}'.format(
                    epoch, cnt, 100. * batch_idx / len(train_loader), map_))
                writer.add_scalar('map/test', map_, cnt)
                model.train()
            
            cnt += 1
        
        # Save model
        if(save_this_epoch(args,epoch)):
            save_model(epoch,model.__class__.__name__,model)
        if scheduler is not None:
            scheduler.step()
    
    ## Save conv1 features
#     pickle.dump(model.conv_features,os.path.join(path,'conv1_features.pkl'))
    
    # Validation iteration
    test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test')
    ap, map_ = utils.eval_dataset_map(model, args.device, test_loader)
    writer.add_scalar('map/test', map_, cnt)
    
    #Save model
    if(args.save_at_end):
        save_model(epoch,model.__class__.__name__,model)
    
    ## Close the summary writer
    writer.close()
    
    return ap, map_
