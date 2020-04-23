import os
import re
import numpy as np
import matplotlib.pyplot as plt
import torch
import data


def get_eval(file_name):
    train = []
    valid = []
    file = open(file_name)
    file_lines = file.readlines()
    for l in file_lines:
        if "training" in l:
            train_data = re.findall(r"\s+(\d+.\d+)", l)
            train.append(float(train_data[1]))
        elif "validation" in l:
            valid_data = re.findall(r"\s+(\d+.\d+)", l)
            valid.append(float(valid_data[1]))
    return train, valid


# training and validation curve plot
'''
color_dict = ["red", "firebrick", "royalblue", "navy", "limegreen", "forestgreen"]

resNet_pretrained = os.path.join("test_res", "model_A.txt")

resNet = os.path.join("test_res", "model_B.txt")
resNet_cpu = os.path.join("test_res", "model_B_SGD.txt")

attResNet = os.path.join("test_res", "model_C_attRes_GPU.txt")
attResNet_cpu = os.path.join("test_res", "model_C_attRes_CPU.txt")
attSENet = os.path.join("test_res", "model_C_attSERes_GPU.txt")
attSEResNet_cpu = os.path.join("test_res", "model_C_attSERes_CPU.txt")

resNet_pretrained_train, resNet_pretrained_valid = get_eval(resNet_pretrained)
resNet_train, resNet_valid = get_eval(resNet)
resNetCPU_train, resNetCPU_valid = get_eval(resNet_cpu)
attResNet_train, attResNet_valid = get_eval(attResNet)
attResNetCPU_train, attResNetCPU_valid = get_eval(attResNet_cpu)
attSENet_train, attSENet_valid = get_eval(attSENet)
attSENetCPU_train, attSENetCPU_valid = get_eval(attSEResNet_cpu)

print("ResNet_pretrained", "\tMax train acc: ", max(resNet_pretrained_train), "\tMax valid acc: ", max(resNet_pretrained_valid))
print("ResNet", "\tMax train acc: ", max(resNet_train), "\tMax valid acc: ", max(resNet_valid))
print("AttResNet", "\tMax train acc: ", max(attResNet_train), "\tMax valid acc: ", max(attResNet_valid))
print("AttResNet_CPU", "\tMax train acc: ", max(attResNetCPU_train), "\tMax valid acc: ", max(attResNetCPU_valid))
print("AttSENet", "\tMax train acc: ", max(attSENet_train), "\tMax valid acc: ", max(attSENet_valid))
print("AttSENet_CPU", "\tMax train acc: ", max(attSENetCPU_train), "\tMax valid acc: ", max(attSENetCPU_valid))

xloc = np.linspace(1, len(resNet_train), len(resNet_train))

fig, ax = plt.subplots(1, 4, sharey=True, figsize=(16, 4))

ax[0].set_title("ResNet, pretrained")
ax[0].plot(xloc, resNet_pretrained_train, color=color_dict[1], label="CPU, train")
ax[0].plot(xloc, resNet_pretrained_valid, color=color_dict[1], linestyle="--", label="CPU, valid")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Accuracy", rotation=90)
ax[0].legend(loc=4)

ax[1].set_title("ResNet")
ax[1].plot(xloc, resNet_train, color=color_dict[0], label="GPU, train")
ax[1].plot(xloc, resNet_valid, color=color_dict[0], linestyle="--", label="GPU, valid")
ax[1].plot(xloc, resNetCPU_train, color=color_dict[1], label="CPU, train")
ax[1].plot(xloc, resNetCPU_valid, color=color_dict[1], linestyle="--", label="CPU, valid")
ax[1].set_xlabel("Epoch")
ax[1].legend(loc=4)

ax[2].set_title("Attention-ResNet")
ax[2].plot(xloc, attResNet_train, color=color_dict[2], label="GPU, train")
ax[2].plot(xloc, attResNet_valid, color=color_dict[2], linestyle="--", label="GPU, valid")
ax[2].plot(xloc, attResNetCPU_train, color=color_dict[3], label="CPU, train")
ax[2].plot(xloc, attResNetCPU_valid, color=color_dict[3], linestyle="--", label="CPU, valid")
ax[2].set_xlabel("Epoch")
ax[2].legend(loc=4)

ax[3].set_title("Attention-SENet")
ax[3].plot(xloc, attSENet_train, color=color_dict[4], label="GPU, train")
ax[3].plot(xloc, attSENet_valid, color=color_dict[4], linestyle="--", label="GPU, valid")
ax[3].plot(xloc, attSENetCPU_train, color=color_dict[5], label="CPU, train")
ax[3].plot(xloc, attSENetCPU_valid, color=color_dict[5], linestyle="--", label="CPU, valid")
ax[3].set_xlabel("Epoch")
ax[3].legend(loc=4)

#plt.show()
plt.savefig("train_curve.png", dpi=400)
'''

# Confusion matrix plot
pretrained_model = torch.load(os.path.join("checkpoint", "best_model_SEC_lrDA_LS.pt"), map_location=torch.device('cpu'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size = 224
data_dir = "../data/"
batch_size = 24
num_class = 20

pretrained_model.train(False)
total_correct = 0

train_loader, valid_loader = data.load_data(data_dir=data_dir, input_size=input_size, batch_size=batch_size)

confusion_mat = np.zeros(shape=(num_class, num_class))
for inputs, labels in valid_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = pretrained_model(inputs)
    _, predictions = torch.max(outputs, 1)
    total_correct += torch.sum(predictions == labels.data)
    predictions = predictions.detach().numpy()
    labels = labels.detach().numpy()
    for i in range(0, len(labels)):
        confusion_mat[labels[i]][predictions[i]] += 1
    epoch_acc = total_correct.double() / len(valid_loader.dataset)

print(epoch_acc.item())

for i in range(0, num_class):
    s = np.sum(confusion_mat[i])
    confusion_mat[i] = confusion_mat[i]/s

class_label = np.arange(num_class)

fig, ax = plt.subplots(1, figsize=(10, 10))
ax1 = ax.imshow(X=confusion_mat, cmap="coolwarm", vmax=1.0, vmin=0.0)
ax.set_xticks(np.arange(num_class))
ax.set_xticklabels(class_label)
ax.set_yticks(np.arange(num_class))
ax.set_yticklabels(class_label)

fig.colorbar(ax1, ax=ax)
#plt.show()
plt.savefig("AttSEResLRDA-LS_confMat.png", dpi=400)


