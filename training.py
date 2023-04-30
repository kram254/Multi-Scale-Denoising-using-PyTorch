import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# Define the network architecture
class Encoder1(nn.Module):
    def __init__(self):
        super(Encoder1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        return x

class Encoder2(nn.Module):
    def __init__(self):
        super(Encoder2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(192, 64, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = nn.functional.relu(self.deconv1(x))
        x = nn.functional.relu(self.deconv2(x))
        x = self.deconv3(x)
        return x

class MultiScaleDenoising(nn.Module):
    def __init__(self):
        super(MultiScaleDenoising, self).__init__()
        self.encoder1 = Encoder1()
        self.encoder2 = Encoder2()
        self.decoder = Decoder()
        
    def forward(self, x1, x2):
        x1 = self.encoder1(x1)
        x2 = self.encoder2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.decoder(x)
        return x

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        
        for i in range(1, 11):
            img_name = f"image{i}.jpg"
            img_path = os.path.join(self.root_dir, img_name)
            img = Image.open(img_path).convert("RGB")
            self.data.append(img)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img

def main():
    try:
        # Define the data loaders
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        train_dataset = ImageDataset("training_data/noisy_images", transform=train_transform) 
        train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

        # Define the loss function
        criterion = nn.MSELoss()

        # Define the optimizer
        lr = 0.001
        model = MultiScaleDenoising()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Define the training loop
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        num_epochs = 10
        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch_idx, data in enumerate(train_loader):
                data1, data2 = data[:, :, :32, :32], data
                data1 = data1.to(device)
                data2 = data2.to(device)
                optimizer.zero_grad()
                outputs = model(data1, data2)
                loss = criterion(outputs, data2)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
                if (batch_idx+1) % 6 == 0:
                    print('[Epoch %d, Batch %d/%d] Loss: %.3f' % (epoch+1, batch_idx+1, len(train_loader), running_loss/6))
                    running_loss = 0.0

        # Save the trained model
        torch.save(model.state_dict(), './multiscale_denoising.pth')
        print("Model saved!")
    
    except Exception as e:
        print(f"An error occurred during training: {e}")

if __name__ == '__main__':
    main()
