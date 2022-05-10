import math
from typing import Optional, Tuple

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision.models import feature_extraction, vision_transformer

from sklearn.model_selection import KFold

import time

import matplotlib.pyplot as plt

def hello_mtg_classification():
    print("Hello from mtg_classification.py!")

################################################################################

# train_model() is a function to train a model with the given loss criterion, 
# image_data, labels, and hyperparameters. 
#
# modified from train_captioner() in a5.helper
def train_model(
    model,
    output_size,
    num_categories,
    criterion,
    image_train,
    image_val,
    labels_train,
    labels_val,
    num_epochs,
    batch_size,
    learning_rate,
    lr_decay=1,
    device: torch.device = torch.device("cpu"),
    dtype = torch.float64,
    debug=False
):
  ''' 
  model: torchvision.models.model, our feature extractor
  criterion: torch.nn, some loss function, we use nn.CrossEntropyLoss()
  image_data: training set of tensors, shape (N, C, H, W) -- (280, 3, 224, 306)
  image_categories: training labels, shape (N,)

  Trains a torchvision model with a full fine-tune, adding a Linear layer for 
  classification on our MTG images dataset.

  '''
  if (type(model) == torchvision.models.squeezenet.SqueezeNet or
    type(model) == torchvision.models.convnext.ConvNeXt or
    type(model) == torchvision.models.vision_transformer.VisionTransformer):
    # add a linear layer with output size as the number of artists in our MTG dataset
    clf = nn.Sequential(
      model, 
      nn.Linear(output_size, num_categories)
    ).to(dtype=dtype, device=device)
  else:
    # Here, we modify the last fully-connected layer of the predefined model
    # to ouput class scores for the number of artists in our MTG dataset. 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_categories)
    clf = model.to(dtype=dtype, device=device)
  clf.train()

  labels_train = labels_train.type(torch.LongTensor)
  labels_train = labels_train.to(device)

  labels_val = labels_val.type(torch.LongTensor)
  labels_val = labels_val.to(device)

  optimizer = torch.optim.AdamW(
      filter(lambda p: p.requires_grad, clf.parameters()), learning_rate
  )
  lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
      optimizer, lambda epoch: lr_decay ** epoch
  )

  # sample minibatch data
  iter_per_epoch = math.ceil(image_train.shape[0] // batch_size)
  train_loss_history = []
  val_loss_history = []

  for i in range(num_epochs):
        start_t = time.time()
        clf.train()
        for j in range(iter_per_epoch):
          optimizer.zero_grad()
          clf.train()
          # training step
          images = image_train[j * batch_size : (j + 1) * batch_size].to(device)
          labels = labels_train[j * batch_size : (j + 1) * batch_size]
          with torch.set_grad_enabled(True):
            output = clf(images)
            loss = criterion(output, labels)
            loss.backward()
            train_loss_history.append(loss.item())
            optimizer.step()
        
          # validation step. do not update weights!
          images = image_val.to(device)
          labels = labels_val.to(device)
          with torch.set_grad_enabled(False):
            clf.eval()
            output = clf(images)
            val_loss = criterion(output, labels)
            val_loss_history.append(val_loss.item())

        end_t = time.time()

        print(
            "(Epoch {} / {}) training loss: {:.4f} val loss {:.4f} time per epoch: {:.1f}s".format(
                i, num_epochs, loss.item(), val_loss.item(), end_t - start_t
            )
        )
        if debug == True:
          print(output)
            
        lr_scheduler.step()

  # plot the training losses
  plt.plot(train_loss_history, label="train loss")
  plt.plot(val_loss_history, label="val loss")
  plt.legend()
  plt.xlabel("Iteration")
  plt.ylabel("Loss")
  plt.title("Loss history")
  plt.show()
  return clf, train_loss_history, val_loss_history


################################################################################


def evaluate_model(
    model,
    criterion,
    image_data,
    image_categories,
    device: torch.device = torch.device("cpu")
):
  with torch.no_grad():
    
    model = model.to(device)
    model.eval()

    image_data = image_data.to(device)

    image_categories = image_categories.type(torch.LongTensor)
    image_categories = image_categories.to(device)

    # no batches!
    # this works here because our val and test sets are so small.
    # in another project, this would be batched as well.
    output = model(image_data)
    loss = criterion(output, image_categories).item()
  
  return loss, output




################################################################################
