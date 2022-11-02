"""Module to train the GAN model"""

import torch
from typing import Any
from src.models.modules.text_encoder import TextEncoder
from src.models.modules.image_encoder import InceptionEncoder
from src.models.modules.generator import Generator
from src.models.modules.discriminator import Discriminator
from src.models.modules.image_encoder import VGGEncoder
from src.models.utils import copy_gen_params, define_optimizers, prepare_labels, load_params, save_image_and_caption, save_model
from losses import discriminator_loss, generator_loss, KL_loss

def train(data_loader: Any, config_dict: dict):
    #check whether you need to use mask = (correct_capt == 0) or not.
    """
    Function to train the GAN model
    :param data_loader: Data loader for the dataset
    :param vocab_len: Length of the vocabulary
    :param config_dict: Dictionary containing the configuration parameters
    """
    Ng, D, condition_dim, noise_dim, disc_lr, gen_lr, batch_size,\
        device, epochs, vocab_len, ix2word, output_dir, snapshot, const_dict\
        = config_dict['Ng'], config_dict['D'],config_dict['condition_dim'],\
        config_dict['noise_dim'], config_dict['disc_lr'],config_dict['gen_lr'],\
        config_dict['batch_size'], config_dict['device'],config_dict['epochs'],\
        config_dict['vocab_len'], config_dict['ix2word'],\
        config_dict['output_dir'], config_dict['snapshot'],\
        config_dict['const_dict']
    
    smooth_val_gen = const_dict['smooth_val_gen']
    lambda4 = const_dict['lambda4']
    generator = Generator(Ng, D, condition_dim, noise_dim)
    discriminator = Discriminator(D)
    text_encoder = TextEncoder(vocab_len, D, D // 2)
    image_encoder = InceptionEncoder(D)
    vgg_encoder = VGGEncoder()

    g_param_avg = copy_gen_params(generator)

    optimizer_G, optimizer_D = define_optimizers(generator, discriminator, disc_lr, gen_lr)

    labels_real, labels_fake, labels_match = prepare_labels(batch_size, device)

    for epoch in range(1, epochs + 1):
        for batch_idx, (images, correct_capt, correct_capt_len, curr_class, word_labels) in enumerate(data_loader):

            noise = torch.randn(batch_size, noise_dim).to(device)

            word_emb, sent_emb = text_encoder(correct_capt)
            word_emb, sent_emb = word_emb.detach(), sent_emb.detach()

            local_incept_feat, global_incept_feat = image_encoder(images)

            vgg_feat = vgg_encoder(images)

            # Generate Fake Images
            fake_imgs, mu_tensor, logvar = generator(noise, sent_emb, word_emb, global_incept_feat, local_incept_feat, vgg_feat)

            local_fake_incept_feat, global_fake_incept_feat = image_encoder(fake_imgs)
            vgg_feat_fake = vgg_encoder(fake_imgs)

            # Update Discriminator
            optimizer_D.zero_grad()
            loss_discri = discriminator_loss(logits, labels_real, labels_fake, lambda4)

            loss_discri.backward()
            optimizer_D.step()

            # Update Generator
            optimizer_G.zero_grad()
            loss_gen = generator_loss(logits, local_fake_incept_feat, global_fake_incept_feat, labels_real, word_emb, sent_emb,\
                labels_match, correct_capt_len, curr_class, vgg_feat, vgg_feat_fake, const_dict)
            
            kl_loss = KL_loss(mu_tensor, logvar)

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
                with torch.no_grad():
                    g_backup_params = copy_gen_params(generator)
                    load_params(generator, g_param_avg)
                    fake_imgs, _, _ = generator(noise, sent_emb, word_emb, global_incept_feat, local_incept_feat, vgg_feat)
                    save_image_and_caption(fake_imgs, images, correct_capt, ix2word, batch_idx, epoch, output_dir)
                    load_params(generator, g_backup_params)

            if epoch % snapshot == 0 and epoch != 0:
                save_model(generator, discriminator, g_param_avg, epoch, output_dir)

    save_model(generator, discriminator, g_param_avg, epochs, output_dir)
