import torch
import torch.nn as torchnn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Re-define your exact ResNet9 Architecture
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [torchnn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              torchnn.BatchNorm2d(out_channels),
              torchnn.ReLU(inplace=True)]
    if pool:
        layers.append(torchnn.MaxPool2d(4))
    return torchnn.Sequential(*layers)

class ResNet9(torchnn.Module):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True) 
        self.res1 = torchnn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True) 
        self.conv4 = ConvBlock(256, 512, pool=True) 
        self.res2 = torchnn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = torchnn.Sequential(torchnn.MaxPool2d(4),
                                       torchnn.Flatten(),
                                       torchnn.Linear(512, num_diseases))
        
    def forward(self, xb): 
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

# 2. Setup Data Loading
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Point this to your downloaded PlantVillage dataset directory
dataset = datasets.ImageFolder(root='models/PlantVillage/color', transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 3. Initialize Model, Loss, and Optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet9(3, 20).to(device)
criterion = torchnn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. Minimal Training Loop 
epochs = 5
print(f"Training on {device}...")
for epoch in range(epochs):
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
    print(f"Epoch [{epoch+1}/{epochs}] completed.")

torch.save(model.state_dict(), 'plant_disease_model.pth')
print("Success! plant_disease_model.pth has been generated.")