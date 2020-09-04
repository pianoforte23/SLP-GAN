import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from advGAN import AdvGAN_Attack
import os
from torchvision import models
import torch.nn as nn

# os.environ['CUDA_VISIBLE_DEVICES'] = "1"

use_cuda=True
image_nc=3
epochs = 60
batch_size = 256
BOX_MIN = 0
BOX_MAX = 1

print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# target_model_path = './GTSRB_VGG16.pth'
target_model_path = './GTSRB_SqueezeNetV1_128.pth'

# target_model = models.vgg16(pretrained=True)
target_model = models.squeezenet1_1(pretrained=True)

# n_inputs = target_model.classifier[6].in_features
n_classes = 43
# target_model.classifier[6] = nn.Linear(n_inputs, n_classes)
target_model.classifier[1] = nn.Conv2d(512, n_classes, kernel_size=1)
target_model.load_state_dict(torch.load(target_model_path))
target_model.eval()
target_model = target_model.to(device)

transform = transforms.Compose(
                [
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]
            )
dataset = datasets.ImageFolder(root="dataset/GTSRB/Train/",
                               transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
advGAN = AdvGAN_Attack(device,
                          target_model,
                          n_classes,
                          image_nc,
                          BOX_MIN,
                          BOX_MAX)

advGAN.train(dataloader, epochs)