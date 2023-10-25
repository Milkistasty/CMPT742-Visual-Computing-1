from model import UNet
from dataloader import Cell_data

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torch.optim as optim
import matplotlib.pyplot as plt

import os

#import any other libraries you need below this line
import wandb  # import weights & biases
os.environ["WANDB_DISABLE_SYMLINKS"] = "true"

# Paramteres

# learning rate
lr = 1e-2
# number of training epochs
epoch_n = 5
# input image-mask size
image_size = 572
# root directory of project
root_dir = os.getcwd()
# training batch size
batch_size = 4
# use checkpoint model for training
load = False
# use GPU for training
gpu = True
# hyperparam for SGD
# prevent the optimizer from getting stuck in local minima and to speed up convergence
momentum = 0.99
# hyperparam for SGD
# L2 regularization. It helps prevent overfitting by adding a penalty to the magnitude of the weights
weight_decay = 0.0005

# Initialize W&B for logging
wandb.init(
    # set the wandb project where this run will be logged
    project = "unet",
    
    # track hyperparameters and run metadata
    config = {
    "learning_rate": lr,
    "epochs": epoch_n,
    "image_size": image_size,
    "batch_size": batch_size,
    "gpu": gpu,
    "momentum": momentum,
    "weight_decay": weight_decay
    }
)

# 1. Create dataset
data_dir = os.path.join(root_dir, 'data/cells')

# 2. Split into train / validation partitions and Create data loaders
trainset = Cell_data(data_dir=data_dir, size=image_size)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = Cell_data(data_dir=data_dir, size=image_size, train=False)
testloader = DataLoader(testset, batch_size=batch_size)

# set up devices for training
device = torch.device('cuda:0' if gpu else 'cpu')
wandb.log({"device": str(device)})

model = UNet().to(device)
# (Initialize logging)
wandb.watch(model, log="all")  # Watch model for logging all gradients and parameters

if load:
    print('loading model')
    model.load_state_dict(torch.load('checkpoint.pt'))

# 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
criterion = nn.CrossEntropyLoss()

# Change the optimizer to SGD
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

# 5. Begin training
for e in range(epoch_n):
    # init the loss to 0 for each epoch
    epoch_loss = 0
    model.train()
    # print("now it is train round")

    # training round
    for i, data in enumerate(trainloader):
        image, label = data

        image = image.to(device)
        label = label.squeeze(1).long().to(device)

        # print(f"Image tensor shape before model call: {image.shape}")

        pred = model(image)

        # print(f"Predicted output shape: {pred.shape}")
        crop_x = (label.shape[1] - pred.shape[2]) // 2
        crop_y = (label.shape[2] - pred.shape[3]) // 2

        label = label[:, crop_x: label.shape[1] - crop_x, crop_y: label.shape[2] - crop_y]

        loss = criterion(pred, label)
        
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()

        print('batch %d --- Loss: %.4f' % (i, loss.item() / batch_size))
    print('Epoch %d / %d --- Loss: %.4f' % (e + 1, epoch_n, epoch_loss / trainset.__len__()))

    torch.save(model.state_dict(), 'checkpoint.pt')

    # Log training loss to W&B
    wandb.log({"Training Loss": epoch_loss / trainset.__len__()})

    # set the model in eval mode
    model.eval()

    total = 0
    correct = 0
    total_loss = 0

    # print("now it is eval round")

    # Evaluation round
    with torch.no_grad():
        for i, data in enumerate(testloader):
            image, label = data

            image = image.to(device)
            label = label.squeeze(1).long().to(device)

            pred = model(image)
            crop_x = (label.shape[1] - pred.shape[2]) // 2
            crop_y = (label.shape[2] - pred.shape[3]) // 2

            label = label[:, crop_x: label.shape[1] - crop_x, crop_y: label.shape[2] - crop_y]

            loss = criterion(pred, label)
            total_loss += loss.item()

            _, pred_labels = torch.max(pred, dim=1)

            total += label.shape[0] * label.shape[1] * label.shape[2]
            correct += (pred_labels == label).sum().item()

        print('Accuracy: %.4f ---- Loss: %.4f' % (correct / total, total_loss / testset.__len__()))
        # Log validation loss and accuracy to W&B
        wandb.log({"Validation Accuracy": correct / total, "Validation Loss": total_loss / testset.__len__()})

torch.save(model.state_dict(), 'checkpoint.pt')
wandb.save('checkpoint.pt')

#testing and visualization

model.eval()

output_masks = []
output_labels = []

with torch.no_grad():
    for i in range(testset.__len__()):
        image, labels = testset.__getitem__(i)

        input_image = image.unsqueeze(0).to(device)
        pred = model(input_image)

        output_mask = torch.max(pred, dim=1)[1].cpu().squeeze(0).numpy()

        crop_x = (labels.shape[0] - output_mask.shape[0]) // 2
        crop_y = (labels.shape[1] - output_mask.shape[1]) // 2
        labels = labels[crop_x: labels.shape[0] - crop_x, crop_y: labels.shape[1] - crop_y].numpy()

        output_masks.append(output_mask)
        output_labels.append(labels)

fig, axes = plt.subplots(testset.__len__(), 2, figsize = (20, 20))

for i in range(testset.__len__()):
  axes[i, 0].imshow(output_labels[i].squeeze())
  axes[i, 0].axis('off')
  axes[i, 1].imshow(output_masks[i].squeeze())
  axes[i, 1].axis('off')

plt.show()

# Finish the W&B run
wandb.finish()
