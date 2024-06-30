import argparse

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import datasets, transforms
from torch.autograd import Variable

from DiffuAug.srcs.generation.sagan.models import (model_duke, model_mnist)
from DiffuAug.srcs import utility

from tqdm import tqdm


def process_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr_gen', type=float, default=1e-4)
    parser.add_argument('--lr_disc', type=float, default=2e-4)
    parser.add_argument('--loss', type=str, default='hinge')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--load', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--sampling_dir', type=str, default='./out')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--channels', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=28)
    
    args = parser.parse_args()
    return args


def train(args, epochs):
    batch_size_mult = 10
    disc_iters = 1
    Z_dim = 128
    channels = args.channels
    img_size = args.img_size
    
    # if args.model == "fashion":
    #     channels = 1
    #     img_size = 28
    # else:
    #     channels = 3
    #     img_size = 32
    
    # 모델 선언
    if args.model == "fashion":
        discriminator = model_mnist.Discriminator().cuda()
        generator = model_mnist.Generator(Z_dim).cuda()
        batch_size_mult = 10
        Z_dim = 128
    elif args.model == "duke":
        # Z_dim = 512
        discriminator = model_duke.Discriminator().cuda()
        generator = model_duke.Generator(Z_dim).cuda()
    else:
        discriminator = model.Discriminator().cuda()
        generator = model.Generator(Z_dim).cuda()
    
    # 모델 load옵션이 있다면 checkpoint 불러오기
    if args.load is not None:
        cp_disc = torch.load(os.path.join(args.checkpoint_dir, 'disc_{}'.format(args.load)))
        cp_gen = torch.load(os.path.join(args.checkpoint_dir, 'gen_{}'.format(args.load)))
        discriminator.load_state_dict(cp_disc)
        generator.load_state_dict(cp_gen)
        print('Loaded checkpoint (epoch {})'.format(args.load))

    # data loader 선언
    if args.model == "fashion":
        data_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(args.data_dir, train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5), (0.5))])),
                batch_size=args.batch_size*batch_size_mult, shuffle=True, num_workers=1, pin_memory=True)
    elif args.model == "duke":
        data_loader = utility.get_duke_dataloader(
            png_dir=args.data_dir,
            train_batchsize=args.batch_size * batch_size_mult,
            img_size=args.img_size,
            num_workers=4
        )

    # Gradient가 필요한 parameter만 optimizer에 넣어주기 위해 filter를 사용함. 
    # spectral noram은 gradient가 필요없는 parameter를 만들어내기 때문에 이를 제외해줌
    optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr_disc, betas=(0.0,0.9))
    optim_gen  = optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr=args.lr_gen, betas=(0.0,0.9))

    scheduler_d = optim.lr_scheduler.ExponentialLR(optim_disc, gamma=0.999)
    scheduler_g = optim.lr_scheduler.ExponentialLR(optim_gen, gamma=0.999) 

    fixed_z = torch.randn(args.batch_size, Z_dim).cuda()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in tqdm(range(epochs)):
        for batch_idx, (data, target) in enumerate(data_loader):
            if data.size()[0] != args.batch_size*batch_size_mult:
                print(f"data.size()[0]: {data.size()[0]}, args.batch_size*batch_size_mult: {args.batch_size*batch_size_mult}")
                continue
            data, target = data.cuda(), target.cuda()

            rand_class, rand_c_onehot = make_rand_class(args)
            samples = data[(target == rand_class).nonzero()].squeeze()
            bsize = samples.size(0)
            data_selected = samples.repeat((args.batch_size // bsize + 1, 1,1,1,1)).view(-1, channels, img_size, img_size)[:args.batch_size]

            # Discriminator 업데이트
            for _ in range(disc_iters):
                z = torch.randn(args.batch_size, Z_dim).cuda() # 아..여기서..우선 터지네

                optim_disc.zero_grad()
                optim_gen.zero_grad()

                disc_loss = (nn.ReLU()(1.0 - discriminator(data_selected, rand_c_onehot))).mean() + (nn.ReLU()(1.0 + discriminator(generator(z, rand_c_onehot[0]), rand_c_onehot))).mean()

                disc_loss.backward()
                optim_disc.step()

            z = Variable(torch.randn(args.batch_size, Z_dim).cuda())
            rand_class, rand_c_onehot = make_rand_class(args)
            # update generator
            optim_disc.zero_grad()
            optim_gen.zero_grad()

            gen_loss = -discriminator(generator(z, rand_c_onehot[0]), rand_c_onehot).mean()
            gen_loss.backward()
            optim_gen.step()

            # if batch_idx % 100 == 99:
            #     print('disc loss', disc_loss.data.item(), 'gen loss', gen_loss.data.item())

            if batch_idx % 30 == 29:
                print('disc loss', disc_loss.data.item(), 'gen loss', gen_loss.data.item())

        scheduler_d.step()
        scheduler_g.step()
        
        if epoch % 100 == 0:
            evaluate(epoch, args, fixed_z, generator)
            
            disc_save_path = os.path.join(args.checkpoint_dir, f"disc_{epoch}.pth")
            gen_save_path = os.path.join(args.checkpoint_dir, f"gen_{epoch}.pth")
            
            # torch.save(discriminator.state_dict(), disc_save_path)
            # torch.save(generator.state_dict(), gen_save_path)
            torch.save(discriminator, disc_save_path)
            torch.save(generator, gen_save_path)
        
        if epoch == 1998:
            generate_img_jebal(args, fixed_z, generator)


def generate_img_jebal(args, fixed_z, generator):
    classes = [0, 1]
    batch_size = 32
    neg_total_img = 16000
    pos_total_img = 2600
    neg_iter = neg_total_img // batch_size
    pos_iter = pos_total_img // batch_size
    
    neg_output_dir = r"/data/results/generation/sampling/sagan/neg"
    pos_output_dir = r"/data/results/generation/sampling/sagan/pos"
    
    for fixed_class in classes:
        fixed_c_onehot = torch.zeros(args.batch_size, args.num_classes).cuda()
        fixed_c_onehot.zero_()
        fixed_c_onehot[:, fixed_class] = 1
        
        class_name = "pos" if fixed_class == 1 else "neg"
        
        if class_name == "pos":
            print("pos_iter: ", pos_iter)
            for i in range(pos_iter):
                samples = generator(fixed_z, fixed_c_onehot[0]).expand(-1, 3, -1, -1).cpu().detach().numpy()[:64]
                # 배치를 분리하여 이미지 저장
                for j in range(samples.shape[0]):
                    img = samples[j]  # (3, 64, 64)
                    img = img.transpose((1, 2, 0))  # (64, 64, 3)
                    img = (img * 0.5 + 0.5) * 255  # Normalize to [0, 255]
                    img = img.astype(np.uint8)  # Convert to uint8
                    output_path = os.path.join(pos_output_dir, f'{class_name}_image_{i*batch_size+j}.png')
                    cv2.imwrite(output_path, img)
        
        elif class_name == "neg":
            print("neg_iter: ", neg_iter)
            for i in range(neg_iter):
                samples = generator(fixed_z, fixed_c_onehot[0]).expand(-1, 3, -1, -1).cpu().detach().numpy()[:64]
                # 배치를 분리하여 이미지 저장
                for j in range(samples.shape[0]):
                    img = samples[j]  # (3, 64, 64)
                    img = img.transpose((1, 2, 0))  # (64, 64, 3)
                    img = (img * 0.5 + 0.5) * 255  # Normalize to [0, 255]
                    img = img.astype(np.uint8)  # Convert to uint8
                    output_path = os.path.join(neg_output_dir, f'{class_name}_image_{i*batch_size+j}.png')
                    cv2.imwrite(output_path, img)


def make_rand_class(args):
    rand_class = np.random.randint(args.num_classes)
    rand_c_onehot = torch.zeros(args.batch_size, args.num_classes).cuda()
    rand_c_onehot.zero_()
    rand_c_onehot[:, rand_class] = 1    
    return (rand_class, rand_c_onehot)


def evaluate(epoch, args, fixed_z, generator):      
    for fixed_class in range(args.num_classes):
        fixed_c_onehot = torch.zeros(args.batch_size, args.num_classes).cuda()
        fixed_c_onehot.zero_()
        fixed_c_onehot[:, fixed_class] = 1

        samples = generator(fixed_z, fixed_c_onehot[0]).expand(-1, 3, -1, -1).cpu().detach().numpy()[:64]
        fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(8, 8)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.transpose((1,2,0)) * 0.5 + 0.5)

        if not os.path.exists(args.sampling_dir):
            os.makedirs(args.sampling_dir)
        
        plt.savefig(f"{args.sampling_dir}/{str(epoch).zfill(3)}_{str(fixed_class).zfill(2)}.png", bbox_inches='tight')
        plt.close(fig)


def main():
    utility.set_seed(990912)
    
    args = process_parser()
    print("args: ", args)
    
    train(args=args, epochs=2000)

if __name__ == "__main__":
    main()