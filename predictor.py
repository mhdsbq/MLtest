import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def image_loader(filepath, transform):
    """load image, returns image tensor"""
    input_shape = (32, 32)
    image = Image.open(filepath)
    image = image.resize(input_shape)
    image = transform(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = torch.unsqueeze(image, 0)
    return image  

def model_weights_loader(model, filepath):
    model.load_state_dict(torch.load(filepath))
    return model
    

def predictor(file_path):
    net = Net()
    PATH = './model/cifar_net.pth'
    CLASSES = ('ship', 'truck')

    model = model_weights_loader(net, PATH)

    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    image = image_loader(file_path, transform)
    model.eval()
    result = model(image)

    result = torch.nn.functional.softmax(result[0]).tolist()
    result[0], result[1] = round(result[0], 2), round(result[1], 2)
    prediction, probablity = (CLASSES[0], result[0]) if result[0] > result[1] else (CLASSES[1], result[1])


    return str(f"{{\"prediction\": \"{prediction}\", \"probablity\": \"{probablity}\"}}")


