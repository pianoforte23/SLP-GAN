import torch.nn as nn
import torch
import numpy as np
import models
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
import extract_gradcam as gradcam

models_path = './models/'


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class AdvGAN_Attack:
    def __init__(self,
                 device,
                 model,
                 model_num_labels,
                 image_nc,
                 box_min,
                 box_max):
        output_nc = image_nc
        self.device = device
        self.model_num_labels = model_num_labels
        self.model = model
        self.input_nc = image_nc
        self.output_nc = output_nc
        self.box_min = box_min
        self.box_max = box_max
        self.transform_vgg = transforms.Compose(
                                 [
                                     transforms.ToPILImage(),
                                     transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                 ]
                             )

        self.gen_input_nc = image_nc
        self.netG = models.Generator(self.gen_input_nc, image_nc).to(device)
        self.netDisc = models.Discriminator(image_nc).to(device)

        # initialize all weights
        self.netG.apply(weights_init)
        self.netDisc.apply(weights_init)
        
        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=0.001)
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                            lr=0.001)

        if not os.path.exists(models_path):
            os.makedirs(models_path)
            
    def transform_img_batch(self, img_batch):
        img = img_batch[0]
        new_img_batch = self.transform_vgg(img.cpu())
        new_img_batch.unsqueeze_(0)
        for i in range(img_batch.shape[0] - 1):
            img = img_batch[i + 1]
            img = self.transform_vgg(img.cpu())
            img.unsqueeze_(0)
            new_img_batch = torch.cat((new_img_batch, img), 0)
            
        return new_img_batch
    
    def make_partial_perturbation(self, perturbation, props):
        partial_perturbation = torch.zeros_like(perturbation, device=self.device)
        for box in props:
            bbox = box.bbox
            bbox_copy = list(bbox)
            '''
            for i in range(len(bbox_copy)):
                bbox_copy[i] = (bbox_copy[i] * 64) // 224
            '''
            for i in range(bbox_copy[0], bbox_copy[2]):
                for j in range(bbox_copy[1], bbox_copy[3]):
                    partial_perturbation[:, i, j] = perturbation[:, i, j]
                    
        return partial_perturbation

    def train_batch(self, x, labels):
        x_cuda = x.to(self.device)
        # optimize D
        for i in range(1):
            perturbation = self.netG(x_cuda)
            # transform_img = self.transform_vgg(x[0].cpu())
            # props = gradcam.gradcam_area(self.model, transform_img)
            props = gradcam.gradcam_area(self.model, self.model.classifier[1], x[0])
            partial_perturbation = self.make_partial_perturbation(perturbation[0], props)
            partial_perturbation.unsqueeze_(0)
            for j in range(len(x) - 1):
                # transform_img = self.transform_vgg(x[j + 1].cpu())
                # props = gradcam.gradcam_area(self.model, transform_img)
                props = gradcam.gradcam_area(self.model, self.model.classifier[1], x[j + 1])
                sub_partial_perturbation = self.make_partial_perturbation(perturbation[j + 1], props)
                sub_partial_perturbation.unsqueeze_(0)
                partial_perturbation = torch.cat((partial_perturbation, sub_partial_perturbation), 0)

            # add a clipping trick
            adv_images = torch.clamp(partial_perturbation, -0.3, 0.3) + x_cuda
            adv_images = torch.clamp(adv_images, self.box_min, self.box_max)

            self.optimizer_D.zero_grad()
            pred_real = self.netDisc(x_cuda)
            loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real, device=self.device))
            loss_D_real.backward()

            # pred_fake = self.netDisc(adv_images.detach())
            pred_fake = self.netDisc(adv_images)
            loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake, device=self.device))
            loss_D_fake.backward(retain_graph=True)
            loss_D_GAN = loss_D_fake + loss_D_real
            self.optimizer_D.step()

        # optimize G
        for i in range(1):
            self.optimizer_G.zero_grad()

            # cal G's loss in GAN
            pred_fake = self.netDisc(adv_images)
            loss_G_fake = F.mse_loss(pred_fake, torch.ones_like(pred_fake, device=self.device))
            loss_G_fake.backward(retain_graph=True)

            # calculate perturbation norm
            C = 0.1
            loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))
            # loss_perturb = torch.max(loss_perturb - C, torch.zeros(1, device=self.device))

            # cal adv loss
            # transform_adv_images = self.transform_img_batch(adv_images)
            # logits_model = self.model(transform_adv_images.cuda())
            logits_model = self.model(adv_images)
            probs_model = F.softmax(logits_model, dim=1)
            onehot_labels = torch.eye(self.model_num_labels, device=self.device)[labels.to(self.device)]

            # C&W loss function
            real = torch.sum(onehot_labels * probs_model, dim=1)
            other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1)
            zeros = torch.zeros_like(other)
            loss_adv = torch.max(real - other, zeros)
            loss_adv = torch.sum(loss_adv)

            # maximize cross_entropy loss
            # loss_adv = -F.mse_loss(logits_model, onehot_labels)
            # loss_adv = - F.cross_entropy(logits_model, labels)

            adv_lambda = 5
            pert_lambda = 1
            loss_G = adv_lambda * loss_adv + pert_lambda * loss_perturb
            loss_G.backward()
            self.optimizer_G.step()

        return loss_D_GAN.item(), loss_G_fake.item(), loss_perturb.item(), loss_adv.item()

    def train(self, train_dataloader, epochs):
        for epoch in range(1, epochs+1):

            if epoch == 50:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.0001)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=0.0001)
            if epoch == 80:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.00001)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=0.00001)
            loss_D_sum = 0
            loss_G_fake_sum = 0
            loss_perturb_sum = 0
            loss_adv_sum = 0
            print(len(train_dataloader))
            for i, data in enumerate(train_dataloader, start=0):
                images, labels = data
                # images, labels = images.to(self.device), labels.to(self.device)

                loss_D_batch, loss_G_fake_batch, loss_perturb_batch, loss_adv_batch = \
                    self.train_batch(images, labels)
                
                print("loss_D_batch: %.3f, loss_G_fake: %.3f, loss_perturb: %.3f, loss_adv: %.3f" % (loss_D_batch, loss_G_fake_batch, loss_perturb_batch, loss_adv_batch))
                loss_D_sum += loss_D_batch
                loss_G_fake_sum += loss_G_fake_batch
                loss_perturb_sum += loss_perturb_batch
                loss_adv_sum += loss_adv_batch
                print(i, "step...")

            # print statistics
            num_batch = len(train_dataloader)
            print("epoch %d:\nloss_D: %.3f, loss_G_fake: %.3f,\
             \nloss_perturb: %.3f, loss_adv: %.3f, \n" %
                  (epoch, loss_D_sum/num_batch, loss_G_fake_sum/num_batch,
                   loss_perturb_sum/num_batch, loss_adv_sum/num_batch))

            # save generator
            if epoch % 20 == 0:
                netG_file_name = models_path + 'netG_epoch_' + str(epoch) + '.pth'
                torch.save(self.netG.state_dict(), netG_file_name)

