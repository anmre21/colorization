from colorizers import *
import time
import glob as glob
from PIL import Image
from skimage import color
import numpy as np
import pickle
import itertools
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary


"""## Step 1: Build dataloaders for train and test

We will first build dataloaders with PyTorch built-in classes. 
"""

class Edges2Image(Dataset):
  def __init__(self, root_dir, split='train'):
    """
    Args:
        root_dir: the directory of the dataset
        split: "train" or "val"
        transform: pytorch transformations.
    """

    self.files = glob.glob(root_dir + '/*.jpg')

  def __len__(self):
    return len(self.files)

  def __getitem__(self, idx):
    img = load_img(self.files[idx])
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
    tens_l_rs = tens_l_rs.cuda()
    return tens_l_rs

# transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
# ])

# tr_dt should be an Edges2Image class containing the training images
# te_dt should be an Edges2Image class containing the validation images
# For the train_loader, please use a batch size of 4 and set shuffle to True
# For the test_loader, please use a batch size of 5 and set shuffle to False
tr_dt = Edges2Image("mini-edges2shoes/train", split = "train")
te_dt = Edges2Image("mini-edges2shoes/val", split = "val")

train_loader = DataLoader(tr_dt, batch_size=4, shuffle = True)
test_loader = DataLoader(te_dt, batch_size=4, shuffle = True)

# You should have 1,000 training and 100 testing images for mini-edges2shoes dataset
# Or 720 training and 81 testing images for Pokemon images
print('Number of training images {}, number of testing images {}'.format(len(tr_dt), len(te_dt)))

# Sample Output used for visualization
test = test_loader.__iter__().__next__()
img_size = 256
fixed_y_ = test[:, :, :, img_size:].cuda()
fixed_x_ = test[:, :, :, 0:img_size].cuda()
print(len(train_loader))
print(len(test_loader))
print(fixed_y_.shape)

# plot sample image
fig, axes = plt.subplots(2, 2)
axes = np.reshape(axes, (4, ))
for i in range(4):
  example = train_loader.__iter__().__next__()[i].cpu().numpy()
  axes[i].imshow(example)
  axes[i].axis('off')

plt.savefig(f'./train_pics/example.png')


def train(model, num_epochs = 20):
  hist_losses = []
  optimizer = optim.Adam(model.parameters(), lr = 0.0002, betas = (0.5, 0.999))
  CE_loss = nn.CrossEntropyLoss().cuda()

  print('training start!')
  start_time = time.time()
  for epoch in range(num_epochs):
    print('Start training epoch %d' % (epoch + 1))
    losses = []
    epoch_start_time = time.time()
    num_iter = 0
    for x_ in train_loader:

      y_ = x_[:, :, :, img_size:]
      x_ = x_[:, :, :, 0:img_size]
      
      x_, y_ = x_.cuda(), y_.cuda()

      #Train the discriminator
      model.zero_grad()

      result = model(x_)
      

      # Hint: you could use following loss to complete following function
      #BCE_loss = nn.BCELoss().cuda()
      #L1_loss = nn.L1Loss().cuda()
      CE_loss = nn.CrossEntropyLoss().cuda()

        
      loss = CE_loss(result, torch.zeros(result.size()).cuda())

      loss.backward()

      optimizer.step()

      loss_val = loss.detach().item()
      losses.append(loss_val)
      hist_losses.append(loss_val)
      
      num_iter += 1

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time

    print('[%d/%d] - using time: %.2f seconds' % ((epoch + 1), num_epochs, per_epoch_ptime))
    print('loss: %.3f' % (torch.mean(torch.FloatTensor(losses))))
    if epoch == 0 or (epoch + 1) % 5 == 0:
      with torch.no_grad():
        show_result(model, fixed_x_, fixed_y_, (epoch+1))

  end_time = time.time()
  total_ptime = end_time - start_time
  print("Total time taken to train:", total_ptime)

  return hist_losses


def show_result(G, x_, y_, num_epoch):
  predict_images = G(x_)

  fig, ax = plt.subplots(x_.size()[0], 3, figsize=(6,10))

  for i in range(x_.size()[0]):
    ax[i, 0].get_xaxis().set_visible(False)
    ax[i, 0].get_yaxis().set_visible(False)
    ax[i, 1].get_xaxis().set_visible(False)
    ax[i, 1].get_yaxis().set_visible(False)
    ax[i, 2].get_xaxis().set_visible(False)
    ax[i, 2].get_yaxis().set_visible(False)
    ax[i, 0].cla()
    ax[i, 0].imshow(process_image(x_[i]))
    ax[i, 1].cla()
    ax[i, 1].imshow(process_image(postprocess_tens(y_[i], predict_images[i])))
    ax[i, 2].cla()
    ax[i, 2].imshow(process_image(y_[i]))
  
  plt.tight_layout()
  label_epoch = 'Epoch {0}'.format(num_epoch)
  fig.text(0.5, 0, label_epoch, ha='center')
  label_input = 'Input'
  fig.text(0.18, 1, label_input, ha='center')
  label_output = 'Output'
  fig.text(0.5, 1, label_output, ha='center')
  label_truth = 'Ground truth'
  fig.text(0.81, 1, label_truth, ha='center')

  print(f"saving figure for epoch {num_epoch}")
  plt.savefig(f'./train_pics/epoch_{num_epoch}.png')

# Helper function for showing result.
def process_image(img):
  return (img.cpu().data.numpy().transpose(1, 2, 0) + 1) / 2

if __name__ == "__main__":
  model = eccv16(pretrained=False)
  train(model)
  torch.save(model.state_dict(), './models/state.pth')