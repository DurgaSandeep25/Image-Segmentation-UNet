!pip install segmentation-models-pytorch
!pip install -U git+https://github.com/albumentations-team/albumentations
!pip install --upgrade opencv-contrib-python

"""# Download Dataset

original author of the dataset :
https://github.com/VikramShenoy97/Human-Segmentation-Dataset

"""

!git clone https://github.com/parth1620/Human-Segmentation-Dataset-master.git

# Import Packages
import sys
sys.path.append('/content/Human-Segmentation-Dataset-master')

import torch
import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tqdm import tqdm

import helper

"""# Setup Configurations"""

# Dataset Path
CSV_FILE = '/content/Human-Segmentation-Dataset-master/train.csv'
DATA_DIR = '/content/'

# GPU
DEVICE='cuda'

EPOCHS=25
BATCH_SIZE=16
LR=0.003
IMAGE_SIZE=320 # data has different image sizes

# U-Net Related
ENCODER = 'timm-efficientnet-b0'
WEIGHTS = 'imagenet'

# Load the Data
df = pd.read_csv(CSV_FILE)

row = df.iloc[123]
image_path = row.images
mask_path = row.masks

image = cv2.imread(DATA_DIR + image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask = cv2.imread(DATA_DIR + mask_path, cv2.IMREAD_GRAYSCALE)/255.0

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

ax1.set_title('INPUT')
ax1.imshow(image)

ax2.set_title('GROUND TRUTH')
ax2.imshow(mask,cmap = 'gray')

train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

"""# Augmentation Functions

albumentation documentation : https://albumentations.ai/docs/
"""

# In Segmentation dataset, label i.e., ground truth also wil be changed
import albumentations as A
def get_train_augs():
  return A.Compose([
      A.Resize(IMAGE_SIZE, IMAGE_SIZE),
      A.HorizontalFlip(p = 0.5),
      A.VerticalFlip(p=0.5)
  ], is_check_shapes=False)

def get_valid_augs():
  return A.Compose([
      A.Resize(IMAGE_SIZE, IMAGE_SIZE)
  ], is_check_shapes=False)

"""# Custom Dataset"""

from torch.utils.data import Dataset
class SegmentationDataset(Dataset):
  def __init__(self, df, augmentations=None):
    self.df = df
    self.augmentations=augmentations

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    img_path = row.images
    mask_path = row.masks

    image = cv2.imread(DATA_DIR + img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask= cv2.imread(DATA_DIR + mask_path, cv2.IMREAD_GRAYSCALE) # height and width
    mask = np.expand_dims(mask, axis=-1)

    if self.augmentations:
      data = self.augmentations(image=image, mask=mask)
      image = data['image']
      mask = data['mask']

    # height x width x channel -> Channel x height x width
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)

    image = torch.Tensor(image)/255.0
    mask = torch.round(torch.Tensor(mask)/255.0)

    return image, mask

trainset = SegmentationDataset(train_df, get_train_augs())
validset = SegmentationDataset(valid_df, get_valid_augs())

print(f"Size of Trainset : {len(trainset)}")
print(f"Size of Validset : {len(validset)}")

image, mask = trainset[1]
helper.show_image(image, mask)

"""# Dataset -> DataLoader (Mini-Batches)"""

from torch.utils.data import DataLoader

trainloader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True)
validloader = DataLoader(dataset=validset, batch_size=BATCH_SIZE, shuffle=True)

print(len(trainloader), len(validloader))

"""# Segmentation Model

segmentation_models_pytorch documentation : https://smp.readthedocs.io/en/latest/
"""

from torch import nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss

class SegmentationModel(nn.Module):

  def __init__(self):
    super().__init__()
    self.arc = smp.Unet(encoder_name=ENCODER, encoder_weights=WEIGHTS,
                        in_channels=3, classes = 1, activation=None)

  def forward(self, image, masks=None):
    logits = self.arc(image)

    if masks!=None:
      loss1 = DiceLoss(mode='binary')(logits, masks)
      loss2 = nn.BCEWithLogitsLoss()(logits, masks)
      return logits, loss1 + loss2

    return logits

model = SegmentationModel()
model = model.to(DEVICE)

"""# Train and Validation Function"""

def train_fn(dataloader, model, optimizer):

  model.train()
  total_loss = 0.0

  for image, mask in dataloader:
    image = image.to(DEVICE)
    mask = mask.to(DEVICE)

    optimizer.zero_grad()
    logits,loss = model(image, mask)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
  return total_loss/len(dataloader)


def valid_fn(dataloader, model):

  model.eval()
  total_loss = 0.0
  with torch.no_grad():
    for image, mask in dataloader:
      image = image.to(DEVICE)
      mask = mask.to(DEVICE)
      logits, loss = model(image, mask)
      total_loss += loss.item()
  return total_loss/len(dataloader)

"""# Training Loop"""

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_valid_loss = float('inf')
train_loss_lis = []
valid_loss_lis = []


for i in range(EPOCHS):
  train_loss =  train_fn(trainloader, model, optimizer)
  valid_loss = valid_fn(validloader, model)

  train_loss_lis.append(train_loss)
  valid_loss_lis.append(valid_loss)

  if valid_loss < best_valid_loss:
    best_valid_loss = valid_loss
    torch.save(model.state_dict(), 'best_model_image_segmentation.pt')
    print("Model Saved")

plt.plot(np.arange(len(train_loss_lis)), train_loss_lis, label = 'Training Loss')
plt.plot(np.arange(len(valid_loss_lis)), valid_loss_lis, label = 'Validation Loss')
plt.legend()
plt.title("Training and Evaluation")
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy Loss')
plt.show()

"""# Inference"""

idx = 20
model.load_state_dict(torch.load('/content/best_model_image_segmentation.pt'))

image, label = validset[idx]

logits_mask = model(image.to(DEVICE).unsqueeze(0)) # C H W -> 1 C H W
pred_mask = torch.sigmoid(logits_mask)
pred_mask = (pred_mask > 0.5)*1.0

helper.show_image(image, label, pred_mask.detach().cpu().squeeze(0))

idx = 13

image, label = validset[idx]

logits_mask = model(image.to(DEVICE).unsqueeze(0)) # C H W -> 1 C H W
pred_mask = torch.sigmoid(logits_mask)
pred_mask = (pred_mask > 0.5)*1.0

helper.show_image(image, label, pred_mask.detach().cpu().squeeze(0))



















