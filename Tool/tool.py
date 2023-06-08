import torch
from torchvision import models
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Current device: {device}')

# image transform function
transform_func = T.Compose([
    T.Resize((250, 250)),
    T.ToTensor()
])

# Model Config
model = models.resnet18()
# freezing parameters
for param in model.parameters():
    param.requires_grad = False


num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model = model.to(device)

# This line uses .load() to read a .pth file and load the network weights on to the architecture.
model.load_state_dict(torch.load('./model_weights.pth',
                      map_location=torch.device('cpu')))
model.eval()

# Test Images
img = Image.open(input("Enter the name of the file\n"))
img.show()
inputImg = transform_func(img)

inputImgUn = torch.unsqueeze(inputImg, 0)

# Predict
with torch.no_grad():
    pred = model(inputImgUn)

pred = pred.detach().cpu().numpy()[0]

print("The image was AI generated") if pred[0] > pred[1] else print(
    "The image is real")
