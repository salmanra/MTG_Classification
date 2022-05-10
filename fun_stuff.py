import os
import torch
from torchvision import transforms
from PIL import Image
from PIL import Image
import matplotlib.pyplot as plt

def view_sample(imgs_path, img_shape):

  '''
  use view_tensor_sample instead
  '''
  img2tensor = transforms.ToTensor()  
  resize_tensor = transforms.Resize(img_shape)  
	# randomly view one image from each artist
  for i, artist in enumerate(os.listdir(imgs_path)):
    artist_path = os.path.join(imgs_path, artist)
    file_idx = torch.randint(100, (1,)).item()
    artist_path = os.path.join(imgs_path, artist)
    filename =  os.listdir(artist_path)[file_idx]
    img_path = os.path.join(artist_path, filename)
    img = Image.open(img_path)
    tnsr = resize_tensor(img2tensor(img))
    plt.imshow(tnsr.permute(1, 2, 0))
    plt.axis("off")
    plt.title(filename)
    plt.show()

def view_tensor_sample(dataset, labels, num_categories):
  '''
  Randomly view one image from each artist.

  dataset: one tensor. one of img_train, img_val, or img_test. 
  '''
  artists = []
  for i in range(num_categories):
    artists.append(dataset[labels == i])

  
  for i, artist in enumerate(artists):
    num_imgs = artist.shape[0]
    art_idx = torch.randint(num_imgs, (1,)).item()
    img = artist[art_idx].permute(1, 2, 0)
    plt.imshow(img)
    plt.axis("off")
    title = 'Category: ' + str(i) + '\nIndex ' + str(art_idx)
    plt.title(title)
    plt.show()