import os
import numpy as np
import torch
import shutil
from collections import OrderedDict
import glob

from SegModel import *
from SegLoss import *
from dataSet import *


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class, self.num_class))

    def kappa(self):
        po = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        pc = 0.0
        for i in range(0, self.num_class):
            pc += np.sum(self.confusion_matrix[:, i]) * np.sum(self.confusion_matrix[i, :])
        pc = pc / np.sum(self.confusion_matrix)**2
        kappa = (po - pc) / (1.0 - pc)
        return kappa

    def overall_acc(self):
        oa = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return oa

    def user_acc(self):
        ua = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        return ua

    def mIOU(self):
        m = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0)
                                              - np.diag(self.confusion_matrix))
        m = np.nanmean(m)
        return m

    def _generate_matrix(self, target_image, pred_image):
        mask = (target_image >= 0) & (target_image < self.num_class)
        label = self.num_class * target_image[mask].astype('int') + pred_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, target_image, pred_image):
        self.confusion_matrix += self._generate_matrix(target_image, pred_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class, self.num_class))


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.num_classes = args.num_classes
        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        # Define Dataloader
        self.train_loader, self.val_loader = load_seg_data(args, root_dir="SegDataset")

        # Define segmentation model
        # model', type=str, default='U-net', choices=['U-net', 'DeepLab-ResNet50', 'DeepLab-AttResNet']
        if args.model == "U-net":
            self.model = LUSegUNet(num_channel=512, num_classes=5)
        elif args.model == "DeepLab-ResNet50":
            self.model = LUSegDeepLab("ResNet50", num_plane=2048, output_stride=16, num_classes=5,
                                      pretrained_backbone="pretrained_SEResAttentionNet.pt")
        elif args.model == "DeepLab-AttResNet":
            self.model = LUSegDeepLab("AttResNet", num_plane=2048, output_stride=16, num_classes=5,
                                      pretrained_backbone="pretrained_ResNet50.pt")
        else:
            raise NotImplementedError

        # Define Optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        # Define Criterion
        # whether to use class balanced weights
        self.criterion = SegmentationLosses(ignore_index=5, cuda=args.cuda).build_loss(mode=args.loss_type)

        # Define Evaluator
        self.evaluator = Evaluator(self.num_classes)
        # Define lr scheduler
        if args.lr_scheduler == "step":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif args.lr_scheduler == "poly":
            poly_lr = lambda epoch: (1.0-epoch/args.epochs)**0.9
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=poly_lr)
        elif args.lr_scheduler == "cos":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(self.train_loader.dataset), eta_min=0)

        # Using cuda
        if self.args.cuda:
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0

    def train(self, epoch):
        self.model.train()
        self.evaluator.reset()
        train_loss = 0.0
        for i, sample in enumerate(self.train_loader):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()*image.size(0)
            pred = output.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            target = target.cpu().numpy()
            self.evaluator.add_batch(target, pred)

        kappa = self.evaluator.kappa()
        over_acc = self.evaluator.overall_acc()
        user_acc = self.evaluator.user_acc()
        mIoU = self.evaluator.mIOU()
        print('-----Training-----')
        print('Training Loss at Epoch ', epoch, ': %.4f' % train_loss)
        print('Training Overall Accuracy at Epoch ', epoch, ': %.4f' % over_acc)
        print('Training Kappa at Epoch ', epoch, ': %.4f' % kappa)
        print('Training mIoU at Epoch ', epoch, ': %.4f' % mIoU)
        print('Training User Accuracy at Epoch ', epoch, ': ', " ".join(map(lambda x: "{:.4f}".format(x), user_acc)))

    def valid(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        valid_loss = 0.0
        for i, sample in enumerate(self.val_loader):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)

            valid_loss += loss.item()*image.size(0)
            pred = output.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            target = target.cpu().numpy()
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        kappa = self.evaluator.kappa()
        over_acc = self.evaluator.overall_acc()
        user_acc = self.evaluator.user_acc()
        mIoU = self.evaluator.mIOU()
        print('-----Validation-----')
        print('Validation Loss at Epoch ', epoch, ': %.4f' % valid_loss)
        print('Validation Overall Accuracy at Epoch ', epoch, ': %.4f' % over_acc)
        print('Validation Kappa at Epoch ', epoch, ': %.4f' % kappa)
        print('Validation mIoU at Epoch ', epoch, ': %.4f' % mIoU)
        print('Validation User Accuracy at Epoch ', epoch, ': ', " ".join(map(lambda x: "{:.4f}".format(x), user_acc)))

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


class Saver(object):
    def __init__(self, args):
        self.args = args
        self.directory = os.path.join('run', args.checkname)
        # check the experiments have been run so far
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))
            if self.runs:
                previous_miou = [0.0]
                for run in self.runs:
                    run_id = run.split('_')[-1]
                    path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            miou = float(f.readline())
                            previous_miou.append(miou)
                    else:
                        continue
                max_miou = max(previous_miou)
                if best_pred > max_miou:
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()
        p['model'] = self.args.model
        p['out_stride'] = self.args.out_stride
        p['lr'] = self.args.lr
        p['lr_scheduler'] = self.args.lr_scheduler
        p['loss_type'] = self.args.loss_type
        p['epoch'] = self.args.epochs
        p['base_size'] = self.args.base_size

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()
