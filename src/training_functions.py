"""
    training_functions.py
    di Mario Gabriele Carofano

    Questo modulo contiene le funzioni di supporto
    per l'addestramento e la valutazione dei modelli.

"""

#   ####################################################################    #
#   LIBRARIES

import torch
from collections import Counter

#   ####################################################################    #
#   FUNCTIONS

def train_epoch(model, loader, criterion, optimizer):

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        # inputs, labels = inputs.to(DEVICE), labels.to(DEVICE) # Remove comment if dataset does't fit into memory

        optimizer.zero_grad()
        outputs = model(inputs)
        # loss = criterion(torch.log(outputs + 1e-10), labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # scheduler.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100. * correct / total

    # end

def evaluate(model, loader, criterion):

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            # inputs, labels = inputs.to(DEVICE), labels.to(DEVICE) # Remove comment if dataset does't fit into memory
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100. * correct / total

    # end

def compute_class_weights(y_train : list, num_classes : int) -> torch.FloatTensor:

    class_counts = Counter(y_train)
    total_samples = sum(class_counts.values())
    weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 0)
        if count > 0:
            weights.append(total_samples / (num_classes * count))
        else:
            weights.append(1.0)

    return torch.FloatTensor(weights)

    # end