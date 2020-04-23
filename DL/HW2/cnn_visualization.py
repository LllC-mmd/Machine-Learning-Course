"""
Created on Thu Oct 26 11:23:47 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import data
import torch
from torch.nn import ReLU
from misc_functions import (get_example_params,
                            convert_to_grayscale,
                            save_gradient_images,
                            get_positive_negative_saliency)
from vanilla_backprop import VanillaBackprop
from torchvision import transforms
from sklearn import manifold


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        # Register hook to the first layer
        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for pos, module in self.model.features._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, input_image, target_class):
        # Forward pass
        model_output = self.model(input_image.clone().detach().requires_grad_(True))
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        # gradients_as_arr = self.gradients.data.numpy()[0]
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr


if __name__ == '__main__':
    pretrained_model = torch.load(os.path.join("checkpoint", "best_model_SEC_lrDA.pt"), map_location=torch.device('cpu'))
    fig, ax = plt.subplots(1, figsize=(10, 10))

    # t-SNE
    '''
    input_size = 224
    batch_size = 12
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_dir = "../data/"
    train_loader, valid_loader = data.load_data(data_dir=data_dir, input_size=input_size, batch_size=batch_size)

    data_list = []
    label_list = []
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = pretrained_model.features(inputs)
        outputs = np.squeeze(pretrained_model.avgpool(outputs).detach().numpy(), axis=(2, 3))
        labels = labels.detach().numpy()
        for i in range(0, len(labels)):
            data_list.append(outputs[i])
            label_list.append(labels[i])

    data_list = np.array(data_list)
    label_list = np.array(label_list)
    tsne = manifold.TSNE(n_components=2, init='random', random_state=0, perplexity=10)
    Y = tsne.fit_transform(data_list)

    ax.scatter(Y[:, 0], Y[:, 1], c=label_list)
    # plt.show()
    plt.savefig("train_tsne.png", dpi=400)
    '''

    # Guided backprop
    (original_image, prep_img, target_class, file_name_to_export) = get_example_params('../data/valid/2/0003.jpg', 1)
    GBP = GuidedBackprop(pretrained_model)
    # Get gradients
    guided_grads = GBP.generate_gradients(prep_img, target_class)
    # Save colored gradients
    save_gradient_images(guided_grads, file_name_to_export + '_Guided_BP_color')
    # Convert to grayscale
    grayscale_guided_grads = convert_to_grayscale(guided_grads)
    # Save grayscale gradients
    save_gradient_images(grayscale_guided_grads, file_name_to_export + '_Guided_BP_gray')
    # Positive and negative saliency maps
    pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
    save_gradient_images(pos_sal, file_name_to_export + '_pos_sal')
    save_gradient_images(neg_sal, file_name_to_export + '_neg_sal')
    print('Guided backprop completed')

    # grad_times_image
    '''
    (original_image, prep_img, target_class, file_name_to_export) = get_example_params('../data/valid/1/0004.jpg', 1)
    # Vanilla backprop
    VBP = VanillaBackprop(pretrained_model)
    # Generate gradients
    vanilla_grads = VBP.generate_gradients(prep_img, target_class)

    grad_times_image = vanilla_grads[0] * prep_img.detach().numpy()[0]
    # Convert to grayscale
    grayscale_vanilla_grads = convert_to_grayscale(grad_times_image)
    # Save grayscale gradients
    save_gradient_images(grayscale_vanilla_grads,
                         file_name_to_export + '_Vanilla_grad_times_image_gray')
    print('Grad times image completed.')
    '''
