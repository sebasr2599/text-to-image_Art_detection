import time
import copy
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder

PATH = './data'
BATCH_SIZE=512
EPOCHS=1
LR=0.001

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Current device: {device}')

# Getting dataset, transforming to tensor and spliting
transform_dataset = T.Compose([
        T.Resize((250,250)),
        T.ToTensor()
    ])




my_dataset = ImageFolder(PATH,transform=transform_dataset)

 # Dataset split
train_set_size = int(len(my_dataset) * 0.8)
test_set_size = len(my_dataset) - train_set_size
train_set, val_set, test_set = random_split(my_dataset, [train_set_size, test_set_size//2,test_set_size//2])

train_loader = DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)
val_loader = DataLoader(val_set,batch_size=BATCH_SIZE,shuffle=True)
test_loader = DataLoader(test_set,batch_size=BATCH_SIZE,shuffle=True)

dataloaders = {
        'train':DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True),
        'val':DataLoader(val_set,batch_size=BATCH_SIZE,shuffle=True)
        }

dataset_sizes = {
        'train':len(train_set),
        'val':len(val_set)
        } 

print('Train data set:', len(train_set))
print(f'Validation set: {len(val_set)}')
print('Test data set:', len(test_set))

# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.axis('off')
#     plt.show()


# # get some random training images
# dataiter = iter(data_loader)
# images, labels = next(dataiter)

# # show images
# imshow(torchvision.utils.make_grid(images))

# for batch_number, (images, labels) in enumerate(data_loader):
#     print(f'batch number {batch_number}, label: {labels}')
#     print(batch_number, labels)
#     break

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# importing resnet for TL
model = models.resnet18()
#freezing parameters
for param in model.parameters():
    param.requires_grad = False


num_features = model.fc.in_features
model.fc = nn.Linear(num_features,2)
model = model.to(device)

optimizer = torch.optim.SGD(model.fc.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# training model
train_model(model, criterion, optimizer, exp_lr_scheduler, EPOCHS)
torch.save(model.state_dict(), 'model_weights.pth')





