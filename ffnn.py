import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU() # The rectified linear unit; one valid choice of activation function
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)

        self.softmax = nn.LogSoftmax() # The softmax function that converts vectors into probability distributions; computes log probabilities for computational benefits
        self.loss = nn.NLLLoss() # The cross-entropy/negative log likelihood loss taught in class

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        # [to fill] obtain first hidden layer representation
        hidden_layer = self.W1(input_vector)
        hidden_layer = self.activation(hidden_layer) # Apply activation function
        
        # [to fill] obtain output layer representation
        output_layer = self.W2(hidden_layer)
        
        # [to fill] obtain probability dist.
        predicted_vector = self.softmax(output_layer)
        
        return predicted_vector


# Returns: 
# vocab = A set of strings corresponding to the vocabulary
def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab 


# Returns:
# vocab = A set of strings corresponding to the vocabulary including <UNK>
# word2index = A dictionary mapping word/token to its index (a number in 0, ..., V - 1)
# index2word = A dictionary inverting the mapping of word2index
def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index 
        index2word[index] = word 
    vocab.add(unk)
    return vocab, word2index, index2word 


# Returns:
# vectorized_data = A list of pairs (vector representation of input, y)
def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index)) 
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data



def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))

    return tra, val


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required = True, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", default = "to fill", help = "path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    # fix random seeds
    random.seed(42)
    torch.manual_seed(42)

    # load data
    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)
    
    # quick exploratory data analysis for report
    print("========== Exploratory data analysis ==========")
    print("Number of training examples: {}".format(len(train_data)))
    print("Number of validation examples: {}".format(len(valid_data)))
    print("Vocabulary size: {}".format(len(vocab)))
    print("1 star reviews in training data: {}".format(len([1 for _, y in train_data if y == 0])))
    print("2 star reviews in training data: {}".format(len([1 for _, y in train_data if y == 1])))
    print("3 star reviews in training data: {}".format(len([1 for _, y in train_data if y == 2])))
    print("4 star reviews in training data: {}".format(len([1 for _, y in train_data if y == 3])))
    print("5 star reviews in training data: {}".format(len([1 for _, y in train_data if y == 4])))
    print("1 star reviews in validation data: {}".format(len([1 for _, y in valid_data if y == 0])))
    print("2 star reviews in validation data: {}".format(len([1 for _, y in valid_data if y == 1])))
    print("3 star reviews in validation data: {}".format(len([1 for _, y in valid_data if y == 2])))
    print("4 star reviews in validation data: {}".format(len([1 for _, y in valid_data if y == 3])))
    print("5 star reviews in validation data: {}".format(len([1 for _, y in valid_data if y == 4])))
    print("Average training review length: {}".format(np.mean([len(review) for review, _ in train_data])))
    print("Average validation review length: {}".format(np.mean([len(review) for review, _ in valid_data])))

    # vectorize data
    print("========== Vectorizing data ==========")
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)
    

    model = FFNN(input_dim = len(vocab), h = args.hidden_dim)
    optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0.9)
    predicted_valid_labels = [] # save predicted labels for validation data
    true_valid_labels = [] # save true labels for validation data
    training_accuracies = [] # save training accuracies for each epoch
    validation_accuracies = [] # save validation accuracies for each epoch
    training_losses = [] # save training losses for each epoch
    validation_losses = [] # save validation losses for each epoch
    print("========== Training for {} epochs ==========".format(args.epochs))
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        loss = None
        epoch_loss = 0
        correct = 0
        total = 0
        start_time = time.time()
        print("Training started for epoch {}".format(epoch + 1))
        random.shuffle(train_data) # Good practice to shuffle order of training data
        minibatch_size = 16 
        N = len(train_data) 
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        print("Training completed for epoch {}".format(epoch + 1))
        training_accuracy = correct / total
        training_accuracies.append(training_accuracy)
        epoch_loss = epoch_loss / (N // minibatch_size)
        training_losses.append(epoch_loss)
        print("Training accuracy for epoch {}: {}".format(epoch + 1, training_accuracy))
        print("Training time for this epoch: {}".format(time.time() - start_time))


        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print("Validation started for epoch {}".format(epoch + 1))
        minibatch_size = 16 
        N = len(valid_data) 
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                # save final predicted and true labels for confusion matrix
                if epoch == args.epochs - 1:
                    true_valid_labels.append(gold_label) # save true label
                    predicted_valid_labels.append(predicted_label) # save prediction
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
        print("Validation completed for epoch {}".format(epoch + 1))
        validation_accuracy = correct / total
        validation_accuracies.append(validation_accuracy)
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, validation_accuracy))
        print("Validation time for this epoch: {}".format(time.time() - start_time))

        # write out training and validation restuls to results/test.out
        if not os.path.exists("results"):
            os.makedirs("results")
        if not os.path.exists("results/test.out"):
            f = open("results/test.out", "w")
        else:
            f = open("results/test.out", "a")
        f.write("Running with hidden dimension {} and {} epochs".format(args.hidden_dim, args.epochs))
        f.write("\n")
        f.write("Training accuracy for epoch {}: {}".format(epoch + 1, training_accuracy))
        f.write("\n")
        f.write("Validation accuracy for epoch {}: {}".format(epoch + 1, validation_accuracy))
        f.write("\n")
        f.write("\n")
        f.close()
        
    # confusion matrix for validation results
    print("========== Confusion matrix ==========")
    cm = confusion_matrix(true_valid_labels, predicted_valid_labels)
    
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig("results/confusion_matrix.png")
    plt.show()
    
    # plot learning curve for accuracy
    print("========== Accuracy learning curve ==========")
    plt.figure(figsize=(12, 6))
    plt.plot(training_accuracies, label='Training Accuracy')
    plt.plot(validation_accuracies, label='Validation Accuracy', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve (Training vs Validation)')
    plt.legend()
    plt.savefig("results/accuracy_curve.png")
    plt.show()
        
    
    