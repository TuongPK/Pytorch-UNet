import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim

from eval import eval_net
from unet import UNet

from torchvision.transforms import RandomChoice
from torch.utils import data
from tqdm import tqdm

from CubiCasa5k.floortrans.aleads_loaders.augmentations import (RandomCropToSizeTorch,
                                              ResizePaddedTorch,
                                              Compose,
                                              DictToTensor,
                                              ColorJitterTorch,
                                              RandomRotations)
from floortrans.aleads_loaders.aleads_svg_loader import FloorplanSVG

def train_net(net, args):
    
    aug = Compose([
        RandomChoice([
            RandomCropToSizeTorch(data_format='dict', size=(args.image_size, args.image_size)),
            ResizePaddedTorch((0, 0), data_format='dict', size=(args.image_size, args.image_size))
        ])
    ])
                       #DictToTensor()])
                       #ColorJitterTorch()])
    
    train_set = FloorplanSVG(args.data_path, 'train.txt', format='lmdb',
                            augmentations=aug)
    val_set = FloorplanSVG(args.data_path, 'val.txt', format='lmdb',
                           augmentations=None)

    
    trainloader = data.DataLoader(train_set, batch_size=args.batch_size,
                                  num_workers=8, shuffle=True, pin_memory=True)

    valloader = data.DataLoader(val_set, batch_size=1,
                                num_workers=8, pin_memory=True)



    optimizer = optim.SGD(net.parameters(),
                          lr=args.lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    criterion = nn.BCELoss()
    best_val_loss = np.inf

    for epoch in range(args.epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, args.epochs))

        # Train
        net.train()
        epoch_loss = 0.0
        for i, samples in tqdm(enumerate(trainloader), total=len(trainloader), ncols=80, leave=False):
            images = samples['image'].cuda(non_blocking=True)
            labels = samples['label'][0].cuda(non_blocking=True)
            
            outputs = net(images)

            outputs_flat = outputs.view(-1)
            labels_flat = labels.view(-1)

            loss = criterion(outputs_flat, labels_flat)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch finished ! Loss: {}'.format(epoch_loss / i))
        
        # Validate
        val_loss = eval_net(net, valloader, gpu=True)
        print('Validation loss: {}'.format(val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(),
                       'checkpoints/' + 'CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))



def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batch_size', default=10,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-i', '--size', dest='image_size', type='int',
                      default=256, help='image size')
    parser.add_option('-D', '--data-path', dest='data_path', type='string',
                      default='data/cubicasa5k', help='data path')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=1, help='downscaling factor of the images')

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()
    
    net = UNet(n_channels=3, n_classes=1)
        
    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()
        cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net=net, args=args)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
