import os
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import torch
import data


DNN_model = torch.load(os.path.join("checkpoint", "best_model_SEC_lrDA.pt"), map_location=torch.device('cpu'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size = 224
data_dir = "../data/"
batch_size = 12

DNN_model.train(False)

train_loader, valid_loader = data.load_data(data_dir=data_dir, input_size=input_size, batch_size=batch_size)

train_data = []
train_label = []
for inputs, labels in train_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = DNN_model.features(inputs)
    outputs = np.squeeze(DNN_model.avgpool(outputs).detach().numpy(), axis=(2,3))
    labels = labels.detach().numpy()
    for i in range(0, len(labels)):
        train_data.append(outputs[i])
        train_label.append(labels[i])

train_data = np.array(train_data)
train_label = np.array(train_label)


SVM_param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
SVM_classifier = GridSearchCV(SVC(kernel="rbf", gamma="scale"), SVM_param_grid, cv=5)
SVM_classifier.fit(X=train_data, y=train_label)

'''
RF_param_grid = {"n_estimators": [25, 50, 100, 200, 400], "max_depth": [2, 4, None]}
RF_classifier = GridSearchCV(RandomForestClassifier(), RF_param_grid, cv=5)
RF_classifier.fit(X=train_data, y=train_label)
'''

train_correct = 0
for inputs, labels in train_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = DNN_model.features(inputs)
    outputs = np.squeeze(DNN_model.avgpool(outputs).detach().numpy(), axis=(2,3))
    labels = labels.detach().numpy()
    predict_outputs = SVM_classifier.predict(outputs)
    #predict_outputs = RF_classifier.predict(outputs)
    train_correct += np.sum(predict_outputs == labels)


train_acc = train_correct / len(train_loader.dataset)
print("Training Accuracy of ensemble DNN model: ", train_acc)

valid_correct = 0
for inputs, labels in valid_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = DNN_model.features(inputs)
    outputs = np.squeeze(DNN_model.avgpool(outputs).detach().numpy(), axis=(2,3))
    labels = labels.detach().numpy()
    predict_outputs = SVM_classifier.predict(outputs)
    #predict_outputs = RF_classifier.predict(outputs)
    valid_correct += np.sum(predict_outputs == labels)

valid_acc = valid_correct / len(valid_loader.dataset)
print("Validation Accuracy of ensemble DNN model: ", valid_acc)