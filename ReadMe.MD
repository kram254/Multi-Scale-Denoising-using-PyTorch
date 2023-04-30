# Multi-Scale Denoising using PyTorch

This is a Python project that demonstrates how to use PyTorch to build a deep neural network for multi-scale image denoising. The project uses a custom implementation of a multi-scale denoising algorithm that combines information from two different scales to produce a clean image.



https://user-images.githubusercontent.com/33391934/235365239-b72b7ad4-bf6e-46bf-9bcf-ceac09ae788f.mp4



## Requirements
Python 3.6 or higher
PyTorch 1.8.1 or higher
Pillow 8.2.0 or higher
NumPy 1.20.3 or higher

## Installation
Clone the repository:
```
https://github.com/kram254/Multi-Scale-Denoising-using-PyTorch.git
```
Install the required packages:

```
pip install -r requirements.txt
```

## Usage
### Training
To train the model, run the following command:
```
python training.py
```
This will train the model on the images in the training_data/noisy_images directory and save the trained model to the multiscale_denoising.pth file.

### Testing
To test the trained model on new images, run the following command:
```
python testing.py <model_path> <test_dir>
```

`model_path` is the path to the saved model file (e.g., multiscale_denoising.pth).
`test_dir` is the path to the directory containing the test images.

This will test the model on the images in the test_dir directory and display the results.

## Code Explanation
##### Encoder1, Encoder2, and Decoder Classes
These classes define the encoder and decoder components of the multi-scale denoising network. The `Encoder1` and `Encoder2` classes implement the two encoders, while the `Decoder` class implements the decoder.

```
class Encoder1(nn.Module):
    def __init__(self):
        super(Encoder1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        return x
```

#### MultiScaleDenoising Class
This class defines the multi-scale denoising network by combining the encoders and decoder into a single network.

```
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
```

#### ImageDataset Class
This class defines the dataset used for training the model. It loads the noisy images from the training_data/noisy_images directory and applies the specified transformations to them.
```
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        
        for i in range(1, 11):
            img_name = f"image{i}.jpg"
            img_path = os.path.join(self.root_dir, img_name)
            
```

# Testing the Model
To test the model, we have provided a `testing.py` script. This script loads the trained model and applies it to a set of test images. The output of the script is a set of denoised images.

To run the script, simply execute the following command:
```
python testing.py --model_path <path_to_trained_model> --test_dir <path_to_test_directory>
```
where `<path_to_trained_model>` is the file path to the saved model and `<path_to_test_directory>` is the file path to the directory containing the test images.

### Libraries Used
This project uses the following Python libraries:

`torch` : an open source machine learning framework that provides a wide range of tools and libraries for building and training deep learning models.
`torchvision` : a package that provides access to popular datasets, model architectures, and image transformation functions for deep learning in PyTorch.
`Pillow` : the Python Imaging Library that adds support for opening, manipulating, and saving many different image file formats.
`os` : a Python library for interacting with the operating system.
`argparse` : a Python library for parsing command-line arguments.
`numpy` : a Python library for working with arrays and numerical operations.

## Conclusion
This project demonstrates how to build a multi-scale denoising model using PyTorch. With the provided training and testing scripts, it is easy to train and test the model on a set of noisy images. This project can be extended to other image denoising tasks by modifying the architecture and training data.
