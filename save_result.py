import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import models
import models as model
import numpy as np
import os
from advGAN import AdvGAN_Attack
import extract_gradcam as gradcam
import torch.nn as nn

transform = transforms.Compose(
                [
                    transforms.Resize((128, 128)),
                    transforms.ToTensor()
                ]
            )

use_cuda=True
image_nc=3
batch_size = 64

gen_input_nc = image_nc

print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# load the pretrained target model
target_model_path = './GTSRB_SqueezeNetV1_128.pth'
target_model = models.squeezenet1_1(pretrained=True)

n_classes = 43

target_model.classifier[1] = nn.Conv2d(512, n_classes, kernel_size=1)
target_model.eval()
target_model = target_model.to(device)
target_layer = target_model.classifier[1]

advGAN = AdvGAN_Attack(device,
                       target_model,
                       n_classes,
                       image_nc, 0, 1)

# load the generator of adversarial examples
pretrained_generator_path = './models/netG_epoch_60.pth'
pretrained_G = model.Generator(gen_input_nc, image_nc).to(device)
pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
pretrained_G.eval()

GTSRB_data = datasets.ImageFolder(root="dataset/GTSRB/Train/",
                                  transform=transform)

sample_index = np.random.randint(len(GTSRB_data))
sample_data, sample_label = GTSRB_data.__getitem__(sample_index)
sample_data.unsqueeze_(0)

perturbation = pretrained_G(sample_data.cuda())
props = gradcam.gradcam_area(target_model, target_layer, sample_data[0])
partial_perturbation = advGAN.make_partial_perturbation(perturbation[0], props)
partial_perturbation.unsqueeze_(0)

perturbation = torch.clamp(partial_perturbation, -0.3, 0.3)
adv_sample = perturbation + sample_data.cuda()
adv_sample = torch.clamp(adv_sample, 0, 1)

adv_sample_path = 'result/adv_sample.png'
real_sample_path = 'result/real_sample.png'

adv_sample_copy = adv_sample.detach().cpu().clone().numpy()
real_sample_copy = sample_data.clone().numpy()
plt.imshow(np.transpose(adv_sample_copy[0], (1, 2, 0)))
plt.savefig(adv_sample_path)
plt.imshow(np.transpose(real_sample_copy[0], (1, 2, 0)))
plt.savefig(real_sample_path)

print("class of real sample : ", sample_label)
print("predicted class of adv sample : ", torch.argmax(target_model(adv_sample), 1))