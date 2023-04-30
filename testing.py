import os
import torch
import torchvision.transforms as transforms
from PIL import Image

# Define the network architecture
class Encoder1(torch.nn.Module):
    def __init__(self):
        super(Encoder1, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        return x

class Encoder2(torch.nn.Module):
    def __init__(self):
        super(Encoder2, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        return x

class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv1 = torch.nn.ConvTranspose2d(192, 64, kernel_size=3, padding=1)
        self.deconv2 = torch.nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.deconv3 = torch.nn.ConvTranspose2d(32, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.deconv1(x))
        x = torch.nn.functional.relu(self.deconv2(x))
        x = self.deconv3(x)
        return x

class MultiScaleDenoising(torch.nn.Module):
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

def test(model_path, test_dir):
    try:
        # Load the saved model
        model = MultiScaleDenoising()
        model.load_state_dict(torch.load(model_path))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Define the transform for the test images
        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Test the model on the images in the test directory
        for file_name in os.listdir(test_dir):
            if file_name.endswith(".jpg"):
                img_path = os.path.join(test_dir, file_name)
                img = Image.open(img_path).convert("RGB")
                img = test_transform(img).unsqueeze(0).to(device)
                output = model(img[:, :, :, :32], img)
                output = output.clamp(0, 1).squeeze().permute(1, 2, 0).cpu().detach().numpy()

                # Save the denoised image
                output = (output * 255).astype('uint8')
                denoised_img = Image.fromarray(output)
                denoised_img.save(f"denoised_{file_name}")
                
        print("Testing complete!")
    
    except Exception as e:
        print(f"An error occurred during testing: {e}")

if __name__ == '__main__':
    model_path = './multiscale_denoising.pth'
    test_dir = './testing_data'
    test(model_path, test_dir)
