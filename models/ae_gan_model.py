from collections import OrderedDict
import torch
from torch import nn
from .base_model import BaseModel
from . import networks

import numpy as np
import pandas as pd
import torch.nn.functional as F
import math

from .evidence_loss import get_loss
class AEGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(norm='batch', netG='mh_resnet_6blocks', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):

        BaseModel.__init__(self, opt)

        self.epoch = 1

        self.n_input_modal = opt.n_input_modal # 3
        self.loss_names = ['G_GAN', 'G_L1', 'gram_contrastive', 'D_real', 'D_fake', 'evidence', 'SR_L1', 'G_SR']
        if self.isTrain:
            self.model_names = ['G']
        else:
            self.model_names = ['G']
        self.netG = networks.define_MHG(opt.n_input_modal, opt.input_nc+opt.n_input_modal+1, opt.output_nc, opt.ngf, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # opt.n_input_modal:3, opt.input_nc+opt.n_input_modal+1=1+3+1=5, opt.output_nc:1, opt.ngf:64, opt.norm:batch, opt.no_dropout:False, opt.init_type:normal, opt.init_gain:0.02
        if self.isTrain:
            self.criterionCls = nn.CrossEntropyLoss()
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device) # opt.gan_mode: vanilla
            self.criterionL1 = nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)) # opt.lr: 0.0001, opt.beta1: 0.5
            self.optimizers.append(self.optimizer_G)

            self.criterionL2 = torch.nn.MSELoss()
            self.criterionKL = torch.nn.KLDivLoss()

        self.all_modal_names = opt.modal_names
        self.sr_weight = opt.sr_weight
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def set_input(self, input):

        self.real_A = input['A'].to(self.device)    # Input image real_A [1, 15, 256, 256]
        self.real_B = input['B'].to(self.device)    # Target image real_B [1, 1, 256, 256]
        self.real_B_no_mask = input['B'][:, :self.opt.input_nc].to(self.device) # Remove the mask of the target modality and retain only the target modality image [1, 1, 256, 256]
        # print('===', input['modal_names']) # eg: [('t1ce',), ('t2',), ('flair',), ('t1',)]
        self.modal_names = [i[0] for i in input['modal_names']]
        target_modal_names = input['modal_names'][-1]
        self.real_B_Cls = torch.tensor([self.all_modal_names.index(i) for i in target_modal_names]).to(self.device)

        if hasattr(self, 'sr'):
            self.sr.real_A = self.real_B_no_mask
            self.sr.real_B = self.real_B_no_mask

        self.image_path = input['B_path']
        self.Slice_Index = input['Slice_Index']  # Slice index

    def forward(self, train=False):

        slice_idx_list = self.Slice_Index
        sample_idx_list = []
        image_path = self.image_path
        for file in image_path:
            file_name = file.split('/')[-1]  # BraTS20_Training_282_flair.nii
            sample_idx = file_name.split('_')[-2]  # 282
            sample_idx_list.append(sample_idx)

        if train:
            self.fake_B, self.decoder_features, self.modality_text, self.modality_img1, self.modality_img2, self.modality_img3 = self.netG(self.real_A, sample_idx_list, slice_idx_list, True)
        else:
            self.fake_B = self.netG(self.real_A, sample_idx_list, slice_idx_list, train)  # G(A)


    def backward_D(self):

        # fake
        fake_B = self.fake_B.detach() # fake_B from generator
        _, g_pred_fake, _ = self.sr.netG(fake_B.detach()) # self.sr = sr_model
        self.loss_D_fake = self.criterionGAN(g_pred_fake, False)

        fake_rec_B = self.sr.fake_B.detach()  # fake_B from autoencoder
        _, e_pred_fake, _ = self.sr.netG(fake_rec_B)
        self.loss_D_fake += self.criterionGAN(e_pred_fake, False)
        # Real
        _, pred_real, _ = self.sr.netG(self.real_B_no_mask)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        self.loss_SR_L1 = self.sr.compute_loss()
        # combine loss and calculate gradients

        ############################---Evidence loss---############################
        # 1. Flatten the PatchGAN outputs while retaining a single channel, [N,1], N=1922
        evi_g_pred_fake = g_pred_fake.view(-1).unsqueeze(1)  # [1922, 1]
        evi_e_pred_fake = e_pred_fake.view(-1).unsqueeze(1)  # [1922, 1]
        evi_pred_real = pred_real.view(-1).unsqueeze(1)  # [1922, 1]

        # 2. Derive binary classification evidence from the PatchGAN "realness probability"
        # Logic: realness probability p → real-class evidence=Softplus(p), fake-class evidence=Softplus(1-p)
        # Ensure that the evidence for both classes is non-negative and semantically consistent
        # (the larger p is, the greater the real-class evidence and the smaller the fake-class evidence)
        softplus = nn.Softplus()
        # Fake images (g_pred_fake/e_pred_fake): low realness p, high fake-class evidence, low real-class evidence
        evi_g_pred_fake = torch.cat([softplus(-evi_g_pred_fake), softplus(evi_g_pred_fake)],
                                    dim=1)  # [1922, 2] (column 0 is the fake class, column 1 is the real class)
        evi_e_pred_fake = torch.cat([softplus(-evi_e_pred_fake), softplus(evi_e_pred_fake)], dim=1)  # [1922, 2]
        # Real images (pred_real): high realness p, high real-class evidence, low fake-class evidence
        # (the logic is consistent, so no reversal is required)
        evi_pred_real = torch.cat([softplus(-evi_pred_real), softplus(evi_pred_real)], dim=1)  # [1922, 2]

        evi_label_fake = torch.zeros_like(g_pred_fake.view(-1)).to(torch.int64).to(torch.device('cuda:0'))
        evi_label_fake = F.one_hot(evi_label_fake, 2)

        evi_label_real = torch.ones_like(g_pred_fake.view(-1)).to(torch.int64).to(torch.device('cuda:0'))
        evi_label_real = F.one_hot(evi_label_real, 2)

        evidence_fake = dict()
        evidence_fake[0] = evi_g_pred_fake
        evidence_fake[1] = evi_e_pred_fake
        evidence_real = dict()
        evidence_real[0] = evi_pred_real

        evidence_fake_loss = get_loss(evidence_fake, evi_label_fake, epoch_num=self.epoch, num_classes=2, annealing_step=50, device=torch.device('cuda:0'))
        evidence_real_loss = get_loss(evidence_real, evi_label_real, epoch_num=self.epoch, num_classes=2, annealing_step=50, device=torch.device('cuda:0'))

        self.loss_evidence = evidence_fake_loss + evidence_real_loss
        ############################---Evidence loss---############################

        self.loss_D = (self.loss_D_fake * 0.5 + self.loss_D_real) * 0.5 + self.loss_SR_L1 + 0.1 * self.loss_evidence
        self.loss_D.backward()
        # 1. self.loss_D_fake: fake image loss
        # (__1__ discriminator evaluates the fake image fake_B generated by the generator,
        # __2__ discriminator evaluates the reconstructed image fake_rec_B generated by the autoencoder)
        # 2. self.loss_D_real: real image loss
        # 3. self.loss_SR_L1: reconstruction loss

    def backward_G(self):
        _, g_pred_fake, _ = self.sr.netG(self.fake_B) 
        self.loss_G_GAN = self.criterionGAN(g_pred_fake, True)
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B_no_mask) * self.opt.lambda_L1 # self.opt.lambda_L1 = 100
        self.loss_G_SR = 0
        sr_decoder_features = self.sr.get_features()
        for i in range(len(sr_decoder_features)):
            self.loss_G_SR += self.criterionKL(torch.nn.functional.log_softmax(self.decoder_features[i], dim=1), torch.nn.functional.softmax(sr_decoder_features[i], dim=1)) * self.sr_weight # self.sr_weight = 0.1

        modality_text = F.normalize(self.modality_text, dim=-1)
        modality_img1 = F.normalize(self.modality_img1, dim=-1)
        modality_img2 = F.normalize(self.modality_img2, dim=-1)
        modality_img3 = F.normalize(self.modality_img3, dim=-1)

        self.loss_gram_contrastive = self.gram_contrastive_loss(modality_text, modality_img1, modality_img2, modality_img3)

        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_SR + 0.001 * self.loss_gram_contrastive



        self.loss_G.backward()
        # 1. self.loss_G_GAN: adversarial loss
        # 2. self.loss_G_L1: reconstruction loss
        # 3. self.loss_G_SR: KL divergence loss (self-representation loss), which forces the generator's feature
        # distribution to be consistent with the feature distribution of the super-resolution model and improves
        # the semantic consistency and feature plausibility of the generated images
        # Obtain the generator decoder features decoder_features and the super-resolution model decoder features
        # sr_decoder_features, and compute the KL divergence (criterionKL) for each feature layer to measure the
        # distribution difference between them

    def optimize_parameters(self):
        self.set_requires_grad(self.sr.netG, True)  # enable backprop for D
        self.sr.forward()
        self.forward(True)                   # compute fake images: G(A)

        self.optimizer_SR.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_SR.step()          # update D's weights
        self.set_requires_grad(self.sr.netG, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def compute_visuals(self):
        """Calculate additional output images for tensorboard visualization"""
        pass

    def get_current_visuals(self):
        modal_imgs = []  # Used to store input modality images
        for i in range(self.n_input_modal):  # n_input_modal = 3, the number of input modalities is 3
            modal_imgs.append(self.real_A[:, i * (self.n_input_modal + 1 + self.opt.input_nc):i * (self.n_input_modal + 1 + self.opt.input_nc) + self.opt.input_nc, :, :]) # self.n_input_modal + 1 + self.opt.input_nc = 5
        # First modality: [:, 0:1, :, :], input modality 1 image without a mask, [1, 1, 256, 256]
        # Second modality: [:, 5:6, :, :], input modality 2 image without a mask, [1, 1, 256, 256]
        # Third modality: [:, 10:11, :, :], input modality 3 image without a mask, [1, 1, 256, 256]
        modal_imgs.append(self.real_B_no_mask)  # Ground-truth target modality image without a mask, [1, 1, 256, 256]
        visual_ret = OrderedDict()  # Initialize an ordered dictionary named visual_ret to store each modality name and its corresponding image data, ensuring that the results are stored in a fixed order
        for name, img in zip(self.modal_names, modal_imgs):
            visual_ret[name] = img

        visual_ret['fake_' + self.modal_names[-1]] = self.fake_B # Generated image produced by the GAN network

        if hasattr(self, 'sr'):
            visual_ret['reconstruct'] = self.sr.fake_B # Reconstructed image produced by the autoencoder

        return visual_ret  # visual_ret contains the input modality images, target modality image, and generated image, returned as an ordered dictionary

    def add_srmodel(self, sr_model):
        self.sr = sr_model
        self.optimizer_SR = torch.optim.Adam(self.sr.netG.parameters(), lr=self.opt.lr,
                                            betas=(self.opt.beta1, 0.999))# losses.py

    def volume_computation(self, anchor, *inputs):
        """
        General function to compute volume for contrastive learning loss functions.
        Compute the volume metric for each vector in anchor batch and all the other modalities listed in *inputs.

        Args:
        - anchor (torch.Tensor): Tensor of shape (batch_size1, dim)
        - *inputs (torch.Tensor): Variable number of tensors of shape (batch_size2, dim)

        Returns:
        - torch.Tensor: Tensor of shape (batch_size1, batch_size2) representing the volume for each pair.
        """
        batch_size1 = anchor.shape[0]
        batch_size2 = inputs[0].shape[0]

        # Compute pairwise dot products for language with itself
        aa = torch.einsum('bi,bi->b', anchor, anchor).unsqueeze(1).expand(-1, batch_size2)

        # Compute pairwise dot products for language with each input
        l_inputs = [anchor @ input.T for input in inputs]

        # Compute pairwise dot products for each input with themselves and with each other
        input_dot_products = []
        for i, input1 in enumerate(inputs):
            row = []
            for j, input2 in enumerate(inputs):
                dot_product = torch.einsum('bi,bi->b', input1, input2).unsqueeze(0).expand(batch_size1, -1)
                row.append(dot_product)
            input_dot_products.append(row)

        # Stack the results to form the Gram matrix for each pair
        G = torch.stack([
            torch.stack([aa] + l_inputs, dim=-1),
            *[torch.stack([l_inputs[i]] + input_dot_products[i], dim=-1) for i in range(len(inputs))]
        ], dim=-2)

        # Compute the determinant for each Gram matrix
        gram_det = torch.det(G.float())

        # Compute the square root of the absolute value of the determinants
        res = torch.sqrt(torch.abs(gram_det))

        return res


    def gram_contrastive_loss(self, text_feat, *image_feats, temperature=0.07, label_smoothing=0.1):
        """
        GRAM-style contrastive loss for multimodal alignment.

        Args:
            text_feat (Tensor): [B, D] — text modality as anchor
            *image_feats (Tensor): each [B, D] — e.g., t1, t2, flair
            temperature (float): softmax temperature
            label_smoothing (float): label smoothing factor

        Returns:
            loss (Tensor): scalar
        """
        B = text_feat.shape[0]
        device = text_feat.device

        # Compute volume matrix: text[i] vs (image_feats[j])
        volume = self.volume_computation(text_feat, *image_feats)  # [B, B]
        # print('--------')
        # print(text_feat)

        volume = volume / temperature

        # Transposed version: image set[j] vs text[i]
        volumeT = self.volume_computation(text_feat, *image_feats).T
        volumeT = volumeT / temperature

        targets = torch.linspace(0, B - 1, B, dtype=int, device=device)
        # targets = torch.arange(B, device=device).long()

        loss_i2t = F.cross_entropy(-volume, targets, label_smoothing=label_smoothing)   # text → images
        loss_t2i = F.cross_entropy(-volumeT, targets, label_smoothing=label_smoothing)  # images → text

        return (loss_i2t + loss_t2i) / 2