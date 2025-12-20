import torch
import torch.nn as nn
from torch.utils.data import Dataset
from datastuff import get_fix_dataloader
import networks
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score
from collections import OrderedDict
import torchvision


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_sizes=[]):
        super(MLP, self).__init__()
        
        # Define input and output layers
        self.input_size = input_size
        self.output_size = output_size
        
        # Create a list to hold the layers (including input and output layers)
        layers = []
        
        # Add input layer
        layers.append(nn.Linear(input_size, hidden_layer_sizes[0]) if hidden_layer_sizes else nn.Linear(input_size, output_size))
        layers.append(nn.ReLU())  # Activation function
        
        # Add hidden layers
        for i in range(len(hidden_layer_sizes) - 1):
            layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1]))
            layers.append(nn.ReLU())  # Activation function
        
        # Add output layer
        if hidden_layer_sizes:
            layers.append(nn.Linear(hidden_layer_sizes[-1], output_size))
        # Note: If there are no hidden layers, the input layer directly connects to the output layer
        
        # Combine all layers into a Sequential model
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # Forward pass through all the layers
        return self.layers(x)
    
    
class NumpyDataset(Dataset):
    def __init__(self, data, labels, discrete=False):
        self.data = torch.from_numpy(data).float()  # Convert NumPy array to torch.Tensor
        if discrete:
            self.labels = torch.from_numpy(labels).long()  # Convert labels to torch.Tensor
        else:
            self.labels = torch.from_numpy(labels).float()  # Convert labels to torch.Tensor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
def calc_acc(model, dataloader, device):
    # Model should be already set to evaluation mode
    with torch.no_grad():
        correct = 0
        total = 0
        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += Y.size(0)
            correct += (predicted == Y).sum().item()
    return correct/total, correct, total
        
    
def calc_acc_binary(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    total_positive = 0
    correct_positive = 0
    total_negative = 0
    correct_negative = 0
    total_samples = 0
    correct_samples = 0

    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the appropriate device (GPU/CPU)
            outputs = model(inputs)  # Forward pass
            
            # If the output dimension is 2, we use softmax to get probabilities, then argmax to get predictions
            predictions = torch.argmax(outputs, dim=1)

            # Update counts for positive class (label == 1)
            total_positive += (labels == 1).sum().item()
            correct_positive += ((predictions == 1) & (labels == 1)).sum().item()

            # Update counts for negative class (label == 0)
            total_negative += (labels == 0).sum().item()
            correct_negative += ((predictions == 0) & (labels == 0)).sum().item()

            # Update counts for total accuracy
            total_samples += labels.size(0)
            correct_samples += (predictions == labels).sum().item()

    positive_accuracy = correct_positive / total_positive if total_positive > 0 else 0
    negative_accuracy = correct_negative / total_negative if total_negative > 0 else 0
    total_accuracy = correct_samples / total_samples if total_samples > 0 else 0

    return positive_accuracy, negative_accuracy, total_accuracy


def calc_acc_mse(model, dataloader, device):
    # Model should be already set to evaluation mode
    with torch.no_grad():
        correct = 0
        total = 0
        error_distance = 0
        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += Y.size(0)
            correct += (predicted == Y).sum().item()
            error_distance += torch.abs(predicted-Y).sum()
    return correct/total, error_distance/total


def calc_split_acc_mse(model, dataloader, split_class_idx, device):
    # Set model to evaluation mode
    model.eval()

    # Initialize variables to track correct predictions for each split
    correct_first_split = 0
    correct_second_split = 0
    total_first_split = 0
    total_second_split = 0
    error_distance_first_split = 0
    error_distance_second_split = 0

    with torch.no_grad():  # Disable gradient computation for inference
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Get model predictions
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Get masks for first split and second split classes
            mask_first_split = labels < split_class_idx
            mask_second_split = labels >= split_class_idx

            # Calculate correct predictions for each split
            correct_first_split += torch.sum(preds[mask_first_split] == labels[mask_first_split]).item()
            correct_second_split += torch.sum(preds[mask_second_split] == labels[mask_second_split]).item()
            
            error_distance_first_split += torch.abs(preds[mask_first_split]-labels[mask_first_split]).sum().item()
            error_distance_second_split += torch.abs(preds[mask_second_split]-labels[mask_second_split]).sum().item()

            # Update totals
            total_first_split += torch.sum(mask_first_split).item()
            total_second_split += torch.sum(mask_second_split).item()

    # Calculate accuracies and mse
    accuracy_first_split = correct_first_split / total_first_split if total_first_split > 0 else 0
    accuracy_second_split = correct_second_split / total_second_split if total_second_split > 0 else 0
    err_dist_first_split = error_distance_first_split / total_first_split if total_first_split > 0 else 0
    err_dist_second_split = error_distance_second_split / total_second_split if total_second_split > 0 else 0
    
    return accuracy_first_split, accuracy_second_split, err_dist_first_split, err_dist_second_split 


def calc_split_mse(model, dataloader, split_value, device):
    # Set model to evaluation mode
    model.eval()

    # Initialize variables to track correct predictions for each split
    total_first_split = 0
    total_second_split = 0
    error_distance_first_split = 0
    error_distance_second_split = 0

    with torch.no_grad():  # Disable gradient computation for inference
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.reshape(-1)
            # Get model predictions
            preds = model(inputs).reshape(-1)

            # Get masks for first split and second split classes
            mask_first_split = labels < split_value
            mask_second_split = labels >= split_value

            # Calculate correct predictions for each split
            error_distance_first_split += torch.abs(preds[mask_first_split]-labels[mask_first_split]).sum().item()
            error_distance_second_split += torch.abs(preds[mask_second_split]-labels[mask_second_split]).sum().item()

            # Update totals
            total_first_split += torch.sum(mask_first_split).item()
            total_second_split += torch.sum(mask_second_split).item()

    # Calculate accuracies and mse
    err_dist_first_split = error_distance_first_split / total_first_split if total_first_split > 0 else 0
    err_dist_second_split = error_distance_second_split / total_second_split if total_second_split > 0 else 0
    
    return err_dist_first_split, err_dist_second_split 

def categorize_by_quantiles(data, k):
    # Calculate the quantiles
    quantiles = np.quantile(data, np.linspace(0, 1, k + 1)[1:-1])
    
    # Initialize an array to hold the category labels
    labels = np.zeros_like(data, dtype=int)
    
    # Assign categories based on quantiles
    for i, threshold in enumerate(quantiles):
        labels[data > threshold] = i + 1
    
    return labels



def categorize_data_with_fixed_median_threshold(data, k, fixed_median_threshold):
    # Step 1: Sort the data
    sorted_data = np.sort(data)
    
    # Step 2: Place the fixed median threshold
    mid_idx = k // 2
    
    # Step 3: Calculate lower and upper quantile-based thresholds
    lower_thresholds = np.quantile(sorted_data[sorted_data < fixed_median_threshold], 
                                   np.linspace(0, 1, mid_idx+1)[1:-1])
    upper_thresholds = np.quantile(sorted_data[sorted_data > fixed_median_threshold], 
                                   np.linspace(0, 1, k-mid_idx+1)[1:-1])
    
    # Step 4: Combine thresholds
    thresholds = np.concatenate((lower_thresholds, [fixed_median_threshold], upper_thresholds))
    
    # Step 5: Categorize the data
    categories = np.digitize(data, thresholds, right=True)

    return categories, thresholds

def has_trainable_parameters(model: torch.nn.Module) -> bool:
    # Iterate through all the parameters in the model
    for param in model.parameters():
        # Check if the parameter requires a gradient (is trainable)
        if param.requires_grad:
            return True
    return False


def freeze_all_parameters(model: torch.nn.Module):
    # Iterate through all the parameters in the model
    for param in model.parameters():
        # Disable gradient computation for each parameter
        param.requires_grad = False
        

def calculate_precision_recall(model, dataloader, device, threshold=-1, discrete=True):
    all_preds = []
    all_labels = []

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Apply softmax to get probabilities and take the argmax to get predictions
            if discrete:
                _, preds = torch.max(F.softmax(outputs, dim=1), 1)
            else:
                preds = outputs.reshape(-1)
                labels = labels.reshape(-1)        
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    # Concatenate the lists of predictions and labels into a single tensor
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Convert tensors to numpy arrays for use with sklearn
    all_preds = all_preds.numpy()
    all_labels = all_labels.numpy()
    
    # Convert into binary
    if threshold >= 0:
        pos_filter = all_preds >= threshold
        neg_filter = all_preds < threshold
        all_preds[pos_filter] = 1
        all_preds[neg_filter] = 0
        pos_filter = all_labels >= threshold
        neg_filter = all_labels < threshold
        all_labels[pos_filter] = 1
        all_labels[neg_filter] = 0        
        
    # Calculate precision and recall
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')

    return precision, recall, all_preds, all_labels


def get_model_results(model, dataloader, device):
    losses = []
    all_probs = []
    correct = 0
    all_labels = []
    with torch.no_grad():
        correct = 0
        for data, labels in dataloader:
            data = data.to(device)
            labels = labels.to(device)
            output = model(data)
            all_probs.append(F.softmax(output, dim=1))
            all_labels.append(labels)
            losses.append(F.cross_entropy(output, labels, reduction="none"))
            correct += (output.argmax(1).eq(labels).float()).sum().item()
    return torch.cat(all_probs).cpu().numpy(), torch.cat(losses).cpu().numpy(), torch.cat(all_labels).cpu().numpy(), correct/len(dataloader.dataset), correct

def get_model_predictions(model, dataloader, device):
    predictions = []
    confs = []
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            labels = labels.to(device)
            output = model(data)
            probs = F.softmax(output, dim=1)
            predictions.append(output.argmax(1))
            confs.append(torch.max(probs, dim=1)[0])
    return torch.cat(predictions).cpu().numpy().reshape(-1), torch.cat(confs).cpu().numpy().reshape(-1)

def get_features(args, fe, transform, domain, phase, device, batch_size=1024):
    all_features = []
    all_labels = []
    dataloader = get_fix_dataloader(args.txtdir, args.dataset, domain, phase, batch_size, img_tr=transform)

    with torch.no_grad():
        for data, labels in dataloader:
            features = fe.encode_image(data.to(device))
            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

def load_model(dataset, model_arch):
    if dataset.__class__.__name__ == "ImageNet":
        if model_arch == "resnet50":
            return torchvision.models.resnet50(pretrained=True)
        elif model_arch == "vitb16":
            return torchvision.models.vit_b_16(pretrained=True)
        elif model_arch == "densenet121":
            return torchvision.models.densenet121(pretrained=True)
        elif model_arch == "effnetb2":
            return torchvision.models.efficientnet_b2(pretrained=True)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
