import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import numpy as np
from torchvision import models
import torch.nn as nn
from torchsummary import summary

if __name__ == "__main__":
    use_cuda = True
    image_nc = 3
    batch_size = 64
    n_classes = 43

    # Define what device we are using
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    transform = transforms.Compose(
                    [
                        transforms.Resize((128, 128)),                       
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]
                )   
    dataset = datasets.ImageFolder(root="dataset/GTSRB/Train/",
                                   transform=transform)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # training the target model
    target_model = models.squeezenet1_1(pretrained=True)
    target_model_path = './GTSRB_SqueezeNetV1_128.pth'
    
    for param in target_model.parameters():
        param.requires_grad = False
    n_inputs = target_model.fc.in_features
        
    target_model.classifier[1] = nn.Conv2d(512, n_classes, kernel_size=1)
    
    target_model.fc = nn.Linear(n_inputs, n_classes, bias=True)
    target_model = target_model.to(device)
    summary(
        target_model,
        input_size=(3, 128, 128),
        batch_size=batch_size,
        device='cuda'
    )
    opt_model = torch.optim.Adam(target_model.parameters(), lr=0.001)
    epochs = 100
    val_loss_min = np.Inf
    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        train_num = 0
        target_model.train()
        
        if epoch == 60:
            opt_model = torch.optim.Adam(target_model.parameters(), lr=0.0001)
        for i, data in enumerate(dataloader, 0):
            train_imgs, train_labels = data
            train_imgs, train_labels = train_imgs.to(device), train_labels.to(device)
            
            logits_model = target_model(train_imgs)
            loss_model = F.cross_entropy(logits_model, train_labels)
            opt_model.zero_grad()
            loss_model.backward()
            opt_model.step()

            train_loss += loss_model.item() * train_imgs.size(0)
            _, pred = torch.max(logits_model, dim=1)
            train_num += train_labels.size(0)
            train_acc += (pred == train_labels).sum().item()
            
        train_acc = 100 * train_acc / train_num      
        train_loss /= len(dataloader.dataset)

        print('train_loss in epoch %d: %f' % (epoch, train_loss))
        print('train_acc in epoch %d: %.2f %%' % (epoch, train_acc))
        
        torch.save(target_model.state_dict(), target_model_path)