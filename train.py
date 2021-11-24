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
N = 1000
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
    base_params = filter(lambda p: id(p) not in vision_params, model.parameters())

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

        optimizer = torch.optim.AdamW([
            {'params': base_params},
            {'params': model.visionMLP.parameters()}
        ])
    optimizer.param_groups[0]['lr'] = opts.lr
    optimizer.param_groups[1]['lr'] = 0


    # models are save only when their loss obtains the best value in the validation
    valtrack = 0  # lr decay

    print('There are %d parameter groups' % len(optimizer.param_groups))
    print('Initial base params lr: {}'.format(optimizer.param_groups[0]['lr']))
    print('Initial vision params lr: {}'.format(optimizer.param_groups[1]['lr']))

    # data preparation, loaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    cudnn.benchmark = True

    # random_sampler = torch.utils.data.RandomSampler(range(100), replacement=True)
    # batch_sampler = torch.utils.data.BatchSampler(random_sampler, batch_size=opts.batch_size, drop_last=True)
    train_loader = torch.utils.data.DataLoader(
        ImagerLoader(opts.img_path,
                     transforms.Compose([
                         transforms.Resize(256),  # rescale the image keeping the original aspect ratio
                         transforms.CenterCrop(256),  # we get only the center of that rescaled
                         transforms.RandomCrop(224),  # random crop within the center crop
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         normalize,
                     ]), data_path=opts.data_path, partition= 'train', sem_reg=opts.semantic_reg),
        num_workers=opts.workers, pin_memory=True,
        batch_size=opts.batch_size, shuffle=True, drop_last=True,
        collate_fn=collate_fn)
    # 238459 recipes
    print('Training loader prepared, {} recipes'.format(len(train_loader.dataset)))

    # preparing validation loader

    val_loader = torch.utils.data.DataLoader(
        ImagerLoader(opts.img_path,
                     transforms.Compose([
                         transforms.Resize(256),  # rescale the image keeping the original aspect ratio
                         transforms.CenterCrop(224),  # we get only the center of that rescaled
                         transforms.ToTensor(),
                         normalize,
                     ]), data_path=opts.data_path, partition='val'),
        num_workers=opts.workers, pin_memory=True,
        batch_size=100, shuffle=True, drop_last=True,
        collate_fn=collate_fn)
    # 51129 recipes
    print('Validation loader prepared, {} recipes'.format(len(val_loader.dataset)))
    # run epochs
    torch.cuda.empty_cache()
    for epoch in range(opts.start_epoch, opts.epochs):

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        if (epoch + 1) % opts.valfreq == 0 and epoch != 0:
            #  sum_score = r1i + r5i + r10i + r1t + r5t + r10t
            sum_score = 0
            for i in range(3):
                sum_score += validate(val_loader, model, criterion)
            sum_score /= 3
            print('Average_score: {}\n'.format(sum_score))
            write_log('Average_score: {}\n'.format(sum_score))

            if sum_score < best_val:
                valtrack += 1
            if valtrack >= opts.patience:
                # we switch modalities
                opts.freeVision = opts.freeRecipe;
                # opts.freeRecipe = not (opts.freeVision)
                # change the learning rate accordingly
                adjust_learning_rate(optimizer, epoch, opts)
                valtrack = 0

            # save the best model
            last_name = opts.snapshots + '{}_lr{}_margin{}_e{}_v-{:.3f}.pth.tar'.format(opts.model,opts.lr, opts.margin, epoch, best_val)
            is_best = sum_score > best_val
            best_val = max(sum_score, best_val)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_val': best_val,
                'optimizer': optimizer.state_dict(),
                'valtrack': valtrack,
                'freeVision': opts.freeVision,
                'curr_val': sum_score,
                'lr': opts.lr,
                'margin': opts.margin,
                'last_name': last_name
            }, is_best)

            print('** Validation: Epoch: {}, Sum_scores: {}, Best: {}'.format(epoch, sum_score, best_val))


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # switch to train mode
    model.train()

    batch_num = 0
    end = time.time()
    semi_ratio = 0
    semi_ratio_average = 0

    for i, (input, target) in enumerate(train_loader):
        if i > 500:
            break
        # measure data loading time
        data_time.update(time.time() - end)

        input_var = list()
        for j in range(len(input)):
            input_var.append(input[j].to(device))

        if opts.neg == 'semi' or opts.neg == 'hardest':
            # compute output: [batch_size, batch_size]
            output = model(input_var[0], input_var[1], input_var[2], input_var[3], input_var[4])
            # compute loss
            margin = opts.margin
            loss, semi_num = utils.calcul_loss(output, margin, opts.neg)
            loss_value = loss.item()
            semi_ratio = semi_num / (opts.batch_size * opts.batch_size * 2)
            semi_ratio_average += semi_ratio
            if opts.neg == 'semi':
                if i % 100 == 0 and semi_ratio_average/(i+1e-12) < 0.015:
                    opts.neg = 'hardest'
                    print('Average semi ration {}, change triplet mode to hardest'.format(semi_ratio_average/i+1e-12))
                    break
            # compute gradient and do Adam step
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        elif opts.neg == 'ohnm':
            for bn in model.modules():
                if isinstance(bn,nn.BatchNorm2d) or isinstance(bn, nn.Dropout):
                    bn.eval()
            with torch.no_grad():
                input_var = list()
                for j in range(len(input)):
                    input_var.append(input[j].to(device))
                scores = model(input_var[0], input_var[1], input_var[2], input_var[3], input_var[4])
            s_index, im_index = utils.select_hard(scores, margin=opts.margin)
            loss_value = 0
            # fix batch_norm because of the small batch_size of hardest triplet
            num_index = len(s_index) + len(im_index)
            if num_index > 0:
                print("we select {} hardest triplet in {} anchors".format(num_index, opts.batch_size*2))
            num = 0
            for index in s_index:
                output = model(input_var[0], input_var[1], input_var[2], input_var[3], input_var[4], hard=True, hard_index=index)
                loss, semi_num = utils.calcul_loss(output, opts.margin, opts.neg)
                loss_value += loss.item()
                # compute gradient and do Adam step
                loss.backward()
            for index in im_index:
                output = model(input_var[0], input_var[1], input_var[2], input_var[3], input_var[4], hard=True, hard_index=index)
                output = output.t()
                loss, semi_num = utils.calcul_loss(output, opts.margin, opts.neg)
                loss_value += loss.item()
                # compute gradient and do Adam step
                loss.backward()
            semi_ratio = num_index / (opts.batch_size * 2)
            semi_ratio_average += semi_ratio
            optimizer.step()
            optimizer.zero_grad()

        batch_time.update(time.time() - end)
        batch_num += 1
        end = time.time()
        if i % 10 == 0:
            log_info = ('Epoch {}, batch {}, loss {:.3f}, ratio {:.3f}, base_lr {}, v_lr {}').format(epoch, i, loss_value, semi_ratio, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'])
            print(log_info)

    semi_ratio_average /= batch_num
    log_info = ('Epoch: {0}, Data_time {1:.2f}, Batch_time {2:.2f}, Loss {loss:.4f}, vision ({visionLR}) - recipe ({'
                'recipeLR}), neg {neg}, Semi-ratio {ratio:.3f}\n').format(
        epoch, data_time.sum, batch_time.sum, loss=loss_value, visionLR=optimizer.param_groups[1]['lr'],
        recipeLR=optimizer.param_groups[0]['lr'], neg=opts.neg, ratio=semi_ratio_average)

    write_log(log_info)
    print(log_info)


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
            if i >= 9:
                break
        print(len(input_visual))
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
        write_log(log_text)
        return sum_score


def write_log(text):
    filename = opts.log + '{}_lr{}_margin{}'.format(opts.model ,opts.lr, opts.margin)
    with open(filename, 'a') as f:
        f.write(text)


def save_checkpoint(state, is_best):
    filename = opts.snapshots + '{}_lr{}_margin{}_e{}_v-{:.3f}.pth.tar'.format (opts.model,state['lr'], state['margin'],state['epoch'], state['best_val'])
    if is_best:
        torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, opts):
    """Switching between modalities"""
    # parameters corresponding to the rest of the network
    decay = 0.5
    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * decay
    # parameters corresponding to visionMLP
    optimizer.param_groups[1]['lr'] = optimizer.param_groups[0]['lr']

    print('Decay learning rate!')
    print('Initial base params lr: {}'.format(optimizer.param_groups[0]['lr']))
    print('Initial vision lr: {}'.format(optimizer.param_groups[1]['lr']))

    # after first modality change we set patience to 3
    opts.patience = 1


if __name__ == '__main__':
    main()
