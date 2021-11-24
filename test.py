import os
import math
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.models as models
import torch.backends.cudnn as cudnn
from data_loader import ImagerLoader, collate_fn
from args import get_parser
from trijoint import im2recipe
import utils
import sys

# =============================================================================
parser = get_parser()
opts = parser.parse_args()
N = opts.val_num
# =============================================================================

if not (torch.cuda.device_count()):
    device = torch.device(*('cpu', 0))
else:
    torch.cuda.manual_seed(opts.seed)
    device = torch.device(*('cuda', 0))


def main():
    model = im2recipe()
    model.visionMLP = torch.nn.DataParallel(model.visionMLP)
    # model = torch.nn.DataParallel(model)
    model.to(device)

    # define loss function (criterion) and optimizer
    # cosine similarity between embeddings -> input1, input2, target
    cosine_crit = nn.CosineEmbeddingLoss(0.1).to(device)
    criterion = cosine_crit

    # # creating different parameter groups
    vision_params = list(map(id, model.visionMLP.parameters()))
    base_params = filter(
        lambda p: id(p) not in vision_params,
        model.parameters())

    # optimizer - with lr initialized accordingly
    optimizer = torch.optim.Adam([
        {'params': base_params},
        {'params': model.visionMLP.parameters(), 'lr': opts.lr * opts.freeVision}
    ], lr=opts.lr * opts.freeRecipe)

    if opts.resume:
        if os.path.isfile(opts.resume):
            print("=> loading checkpoint '{}'".format(opts.resume))
            checkpoint = torch.load(opts.resume)
            opts.start_epoch = checkpoint['epoch']
            best_val = checkpoint['best_val']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(opts.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(opts.resume))
            best_val = float('-inf')
    else:
        best_val = float('-inf')

    # data preparation, loaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        ImagerLoader(opts.img_path,
                     transforms.Compose([
                         # rescale the image keeping the original aspect ratio
                         transforms.Scale(256),
                         # we get only the center of that rescaled
                         transforms.CenterCrop(224),
                         transforms.ToTensor(),
                         normalize,
                     ]), data_path=opts.data_path, partition='val'),
        num_workers=opts.workers, pin_memory=True,
        batch_size=100, shuffle=True, drop_last=True,
        collate_fn=collate_fn)
    # 51129 recipes
    print('Validation loader prepared, {} recipes'.format(len(val_loader.dataset)))

    res = np.zeros(11)
    for i in range(10):
        res += validate(val_loader, model, criterion)
    res /= 10
    i2t_info = "Average Image to text: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}".format(
        res[0], res[1], res[2], res[3], res[4])
    print(i2t_info)
    t2i_info = "Average Text to image: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}".format(
        res[5], res[6], res[7], res[8], res[9],)
    print(t2i_info)
    print('Average Sum Score: ', res[10])


def validate(val_loader, model, criterion):
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        input_visual = []
        input_text = []
        rec_id = []
        for i, (input, target) in enumerate(val_loader):
            img, instrs, itr_ln, ingrs, igr_ln = input
            input_visual.append(img)
            input_text.append([instrs, itr_ln, ingrs, igr_ln])
            for rec in target[-1]:
                rec_id.append(rec)
            if i >= (N // 100) - 1:
                break

        sim = 0
        sim_all = 0
        for i in range(len(input_visual)):
            for j in range(len(input_text)):
                input_v = input_visual[i]
                input_t = input_text[j]
                img = Variable(input_v).cuda()
                instrs = Variable(input_t[0]).cuda()
                itr_ln = Variable(input_t[1]).cuda()
                ingrs = Variable(input_t[2]).cuda()
                igr_ln = Variable(input_t[3]).cuda()
                scores = model(img, instrs, itr_ln, ingrs, igr_ln)
                scores = scores.cpu().data.numpy()
                if j == 0:
                    sim = scores
                else:
                    sim = np.concatenate((sim, scores), 1)
            if i == 0:
                sim_all = sim
            else:
                sim_all = np.concatenate((sim_all, sim), 0)

        (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t(sim_all)
        i2t_info = ("Image to text: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}").format(
            r1i, r5i, r10i, medri, meanri)

        (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_t2i(sim_all)
        t2i_info = ("Text to image: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}").format(
            r1t, r5t, r10t, medrt, meanrt)

        sum_score = r1i + r5i + r10i + r1t + r5t + r10t
        log_text = (
            '{}\n{}\nsum_score: {}\n'.format(
                i2t_info,
                t2i_info,
                sum_score))
        print(log_text)
        return np.array([
            r1i,
            r5i,
            r10i,
            medri,
            meanri,
            r1t,
            r5t,
            r10t,
            medrt,
            meanrt,
            sum_score])


if __name__ == '__main__':
    main()
