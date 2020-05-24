import torch
import torch.nn as nn
import torch.optim as optim
import dataSet
import pretrainedModel
from basicModule import *
import sys
import os

## Note that: here we provide a basic solution for training and validation.
## You can directly change it if you find something wrong or not good enough.


def train_model(model, train_loader, valid_loader, criterion, optimizer, lr_scheduler, num_epochs, save_name):

    def train(model, train_loader, optimizer, criterion):
        model.train(True)
        total_loss = 0.0
        total_correct = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # ---clear the accumulated gradients
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)
            # ---backward propagation
            loss.backward()
            # ---parameter update
            optimizer.step()
            # ---report loss and accuracy
            total_loss += loss.item() * inputs.size(0)
            total_correct += torch.sum(predictions == labels.data)

        epoch_loss = total_loss / len(train_loader.dataset)
        epoch_acc = total_correct.double() / len(train_loader.dataset)
        return epoch_loss, epoch_acc.item()

    def valid(model, valid_loader,criterion):
        model.train(False)
        total_loss = 0.0
        total_correct = 0
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)
            total_loss += loss.item() * inputs.size(0)
            total_correct += torch.sum(predictions == labels.data)
        epoch_loss = total_loss / len(valid_loader.dataset)
        epoch_acc = total_correct.double() / len(valid_loader.dataset)
        return epoch_loss, epoch_acc.item()

    best_acc = 0.0
    for epoch in range(num_epochs):
        print('epoch:{:d}/{:d}'.format(epoch, num_epochs))
        print('*' * 100)
        train_loss, train_acc = train(model, train_loader,optimizer,criterion)
        print("training: {:.4f}, {:.4f}".format(train_loss, train_acc))
        valid_loss, valid_acc = valid(model, valid_loader,criterion)
        print("validation: {:.4f}, {:.4f}".format(valid_loss, valid_acc))
        lr_scheduler.step()
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = model
            torch.save(best_model, save_name)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    ## about model
    num_classes = 5

    ## about data
    data_dir = "./dataset/"
    inupt_size = 224
    batch_size = 6

    ## about training
    num_epochs = 100
    lr = 0.1

    ## model initialization
    if sys.argv[1] == "SEResAttentionNet":
        model = pretrainedModel.SEResAttentionNet(pic_length=224, output_stride=8, block=ResBottleneck, in_plane=64, expansion=4, num_classes=5)
    elif sys.argv[1] == "ResNet101":
        model = pretrainedModel.ResNet101(output_stride=8, block=ResBottleneck, in_plane=64, expansion=4, num_classes=5)
    else:
        raise NotImplementedError
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ## data preparation
    train_loader, valid_loader = dataSet.load_pretrained_data(data_dir=data_dir, batch_size=batch_size)

    ## optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
    # learning rate strategy
    lr_strategy = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    ## loss function
    # criterion = models.LabelSmoothingCEloss()
    criterion = nn.CrossEntropyLoss()
    train_model(model, train_loader, valid_loader, criterion, optimizer, lr_strategy, num_epochs=num_epochs, save_name="pretrained_model.pt")