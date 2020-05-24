import argparse
import os
import torch
from train_config import Trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--model', type=str, default='U-net', choices=['U-net', 'DeepLab-ResNet50', 'DeepLab-AttResNet'],
                        help='segmentation model (default: U-net)')
    parser.add_argument('--out-stride', type=int, default=8, help='network output stride (default: 8)')
    parser.add_argument('--num_classes', type=int, default=5, help='the number of classes (default: 5)')
    parser.add_argument('--base-size', type=int, default=224, help='base image size')
    parser.add_argument('--loss_type', type=str, default='ce', choices=['ce', 'focal'], help='loss func (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 100)')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N', help='batch size for training (default: 8)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--lr_scheduler', type=str, default='poly', choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, metavar='M', help='w-decay (default: 1e-4)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

    parser.add_argument('--checkname', type=str, default=None, help='set the checkpoint name')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Total Epoches:', trainer.args.epochs)

    for epoch in range(0, trainer.args.epochs):
        print('*' * 50)
        trainer.train(epoch)
        trainer.valid(epoch)
        trainer.lr_scheduler.step()
