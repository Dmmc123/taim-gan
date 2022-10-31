"""Module to train the GAN model"""

import torch
from torch import nn
from typing import Any
from src.models.modules.text_encoder import TextEncoder
from src.models.modules.image_encoder import InceptionEncoder
from src.models.modules.generator import Generator
from src.models.modules.discriminator import Discriminator
from src.models.modules.image_encoder import VGGEncoder
from src.models.utils import copy_gen_params
from src.models.utils import define_optimizers
from src.models.utils import prepare_labels
# from losses import discriminator_loss, generator_loss


def main():
    def num_params(net):
        return sum(p.numel() for p in net if p.requires_grad)

    Ng = 32
    D = 256
    condition_dim = 100
    noise_dim = 100
    vocab_len = 10000

    generator = Generator(Ng, D, condition_dim, noise_dim)
    discriminator = Discriminator(D)
    text_encoder = TextEncoder(vocab_len, D, D // 2)
    image_encoder = InceptionEncoder(D)
    vgg_encoder = VGGEncoder()

    networks = {}
    networks['generator'] = generator
    networks['discriminator'] = discriminator
    networks['text_encoder'] = text_encoder
    networks['image_encoder'] = image_encoder
    networks['vgg_encoder'] = vgg_encoder


    for name, net in networks.items():
        print(f'{name} has {num_params(net.parameters())} parameters')


if __name__ == '__main__':
    main()

def train(data_loader: Any, vocab_len: int, config_dict: dict):
    """
    Function to train the GAN model
    :param data_loader: Data loader for the dataset
    :param vocab_len: Length of the vocabulary
    :param config_dict: Dictionary containing the configuration parameters
    """
    Ng, D, condition_dim, noise_dim, disc_lr, gen_lr, batch_size, device, epochs, const_dict = config_dict['Ng'], config_dict['D'],\
        config_dict['condition_dim'], config_dict['noise_dim'], config_dict['disc_lr'],\
        config_dict['gen_lr'], config_dict['batch_size'], config_dict['device'], config_dict['epochs'], config_dict['const_dict']
    
    smooth_val_gen = const_dict['smooth_val_gen']
    generator = Generator(Ng, D, condition_dim, noise_dim)
    discriminator = Discriminator(D)
    text_encoder = TextEncoder(vocab_len, D, D // 2)
    image_encoder = InceptionEncoder(D)
    vgg_encoder = VGGEncoder()

    g_param_avg = copy_gen_params(generator)

    optimizer_G, optimizer_D = define_optimizers(generator, discriminator, disc_lr, gen_lr)

    labels_real, labels_fake, labels_match = prepare_labels(batch_size, device)

    for epoch in range(1, epochs + 1):
        for batch_idx, (images, correct_capt, correct_capt_len, curr_class,\
            wrong_capt, wrong_capt_len, wrong_class, word_labels) in enumerate(data_loader):

            noise = torch.randn(batch_size, noise_dim).to(device)

            word_emb, sent_emb = text_encoder(correct_capt)
            word_emb, sent_emb = word_emb.detach(), sent_emb.detach()

            wrong_word_emb, wrong_sent_emb = text_encoder(wrong_capt)
            wrong_word_emb, wrong_sent_emb = wrong_word_emb.detach(), wrong_sent_emb.detach()

            local_incept_feat, global_incept_feat = image_encoder(images)

            vgg_feat = vgg_encoder(images)

            # Generate Fake Images
            fake_imgs, mu_tensor, logvar = generator(noise, sent_emb, word_emb, global_incept_feat, local_incept_feat, vgg_feat)

            # Update Discriminator
            optimizer_D.zero_grad()
            loss_discri = discriminator_loss(discriminator, images, fake_imgs, sent_emb, word_emb,\
                labels_real, labels_fake, correct_capt_len, image_encoder, curr_class, word_labels)

            loss_discri.backward()
            optimizer_D.step()

            # Update Generator
            optimizer_G.zero_grad()
            loss_gen = generator_loss(discriminator, image_encoder, fake_imgs, labels_real, word_emb, sent_emb,\
                labels_match, correct_capt_len, curr_class, vgg_encoder, images, device, const_dict)
            
            kl_loss = torch.mean( -0.5 * (1 + 0.5 * logvar - mu_tensor.pow(2) - torch.exp(logvar)) ) # -0.5 * (1 + log(sigma) - mu^2 - sigma^2)

            loss_gen += kl_loss
            loss_gen.backward()
            optimizer_G.step()

            # Update the moving average of the generator parameters
            for p, avg_p in zip(generator.parameters(), g_param_avg):
                avg_p = smooth_val_gen * avg_p + (1 - smooth_val_gen) * p.data

            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch [{epoch}/{epochs}], Batch [{batch_idx + 1}/{len(data_loader)}],\
                    Loss D: {loss_discri.item():.4f}, Loss G: {loss_gen.item():.4f}')

            if (batch_idx + 1) % 1000 == 0:
                g_backup_params = copy_gen_params(generator)




            #check whether you need to use mask = (correct_capt == 0) or not.


            # images = images.to(device)
            # captions = captions.to(device)
            # lengths = lengths.to(device)

            # # Train the discriminator
            # optimizer_D.zero_grad()
            # real_features = image_encoder(images)
            # fake_features = generator(captions, lengths, g_param_avg)
            # real_logits = discriminator(real_features)
            # fake_logits = discriminator(fake_features)
            # loss_D_real = nn.BCEWithLogitsLoss()(real_logits, labels_real)
            # loss_D_fake = nn.BCEWithLogitsLoss()(fake_logits, labels_fake)
            # loss_D = loss_D_real + loss_D_fake
            # loss_D.backward()
            # optimizer_D.step()

            # # Train the generator
            # optimizer_G.zero_grad()
            # fake_features = generator(captions, lengths, g_param_avg)
            # fake_logits = discriminator(fake_features)
            # loss_G = nn.BCEWithLogitsLoss()(fake_logits, labels_real)
            # loss_G.backward()
            # optimizer_G.step()

            # Update the moving average of the generator parameters
            for i, p in enumerate(generator.parameters()):
                g_param_avg[i] = 0.999 * g_param_avg[i] + 0.001 * p.data

            # Print the losses
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch} | Batch: {batch_idx} | Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f}')

