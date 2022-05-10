import torch
from torchvision import transforms
from PIL import Image
import os
import random

# file containing useful functions to load and transform jpeg images to pytorch tensors

def hello_image_processing():
  print("Hello from image_processing.py!")


def PIL_to_tensor_list(imgs_path, img_shape, img_per_artist, dtype, device):
  '''
  Transform images from jpeg to tensor. Should also work for other PIL
  formats, not just jpeg.

  img_path: location in Drive of the folder containing images. This folder should 
            contain num_categories folders, each containing the full set of images
            for a given category. 
  img_shape: a shape (H, W) to shrink all images to. 
  img_per_artist: number of images per category. Should be constant across categories. 
  dtype: constant dtype for data and models.
  device: Colab device in use.
  '''

  # torchvision conversion jpg to Tensor
  img2tensor = transforms.ToTensor()  

  # shrink all Tensors to the same size
  resize_tensor = transforms.Resize(img_shape)  

  Hamm_imgs = torch.zeros((img_per_artist, 3, img_shape[0], img_shape[1]), dtype=dtype, device=device)
  McKinnon_imgs = torch.zeros((img_per_artist, 3, img_shape[0], img_shape[1]), dtype=dtype, device=device)
  Nielsen_imgs = torch.zeros((img_per_artist, 3, img_shape[0], img_shape[1]), dtype=dtype, device=device)
  Voss_imgs = torch.zeros((img_per_artist, 3, img_shape[0], img_shape[1]), dtype=dtype, device=device)

  mtg_img_tensors = [Hamm_imgs, McKinnon_imgs, Nielsen_imgs, Voss_imgs]

  # i should go from 0 to 3, one for each artist folder
  for i, artist in enumerate(os.listdir(imgs_path)):
    artist_path = os.path.join(imgs_path, artist)
    # j should go from 0 to 99, one for each image in an artist folder
    for j, filename in enumerate(os.listdir(artist_path)):
      if filename.endswith(".jpg"):
        img_path = os.path.join(artist_path, filename)
        img = Image.open(img_path)
        # output of transforms.ToTensor() is a tensor with values on [0.0, 1.0]
        tnsr = resize_tensor(img2tensor(img))
        artist_tensor = mtg_img_tensors[i]
        artist_tensor[j] = tnsr

  return mtg_img_tensors


def shuffle_and_split(
    dataset,
    seed,
    img_shape,
    num_categories,
    num_train,
    num_val,
    num_test,
    img_per_artist,
    train_per_artist,
    val_per_artist,
    test_per_artist,
    dtype,
    device
  ):
  '''
  Splits the dataset into training, validation, and test sets and shuffles
  categories together while doing so. 

  dataset: list of num_categories tensors where each tensor is the full dataset
           of a single category/artist.
  '''
  random.seed(seed)
  # training set
  img_train = torch.zeros((num_train, 3, img_shape[0], img_shape[1]), dtype=dtype, device=device)
  labels_train = torch.zeros((num_train,), dtype=torch.long, device=device)

  # validation set
  img_val = torch.zeros((num_val, 3, img_shape[0], img_shape[1]), dtype=dtype, device=device)
  labels_val = torch.zeros((num_val,), dtype=torch.long, device=device)

  # tet set. do not touch.
  img_test = torch.zeros((num_test, 3, img_shape[0], img_shape[1]), dtype=dtype, device=device)
  labels_test = torch.zeros((num_test,), dtype=torch.long, device=device)

  # randomly select the test set
  test_idx = torch.tensor(random.sample(range(img_per_artist), test_per_artist))

  # construct the training and validation sets
  idx_list = []
  for i in range(img_per_artist):
    if i not in test_idx:
      idx_list.append(i)
  sample_idx = torch.tensor(idx_list)

  val_idx = torch.tensor(random.sample(list(sample_idx), val_per_artist))

  train_idx_list = []
  for i in range(img_per_artist):
    if i not in val_idx and i not in test_idx:
      train_idx_list.append(i)
  train_idx = torch.tensor(train_idx_list)


  for j in range(num_categories):
    imgs = dataset[j]
    img_test[j*test_per_artist:(j+1)*test_per_artist] = imgs[test_idx]
    labels_test[j*test_per_artist:(j+1)*test_per_artist] = j
    img_train[j*train_per_artist:(j+1)*train_per_artist] = imgs[train_idx]
    labels_train[j*train_per_artist:(j+1)*train_per_artist] = j
    img_val[j*val_per_artist:(j+1)*val_per_artist] = imgs[val_idx]
    labels_val[j*val_per_artist:(j+1)*val_per_artist] = j

  # shuffle the training set so the batches to the model are not all from one artist
  # and shuffle the test set.
  shuffle_train = torch.randperm(num_train, device=device)
  shuffle_val = torch.randperm(num_val, device=device)
  shuffle_test = torch.randperm(num_test, device=device)

  img_train = img_train[shuffle_train]
  img_test = img_test[shuffle_test]
  img_val = img_val[shuffle_val]
  labels_train = labels_train[shuffle_train]
  labels_test = labels_test[shuffle_test]
  labels_val = labels_val[shuffle_val]

  return img_train, img_val, img_test, labels_train, labels_val, labels_test
