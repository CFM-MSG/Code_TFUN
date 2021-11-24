import torch
import numpy as np
import sys
import  math
from torch.autograd import Variable
from collections import OrderedDict
import torch.nn as nn


def calcul_loss(scores, margin, neg):
    # scores_size = x.view( batch_size_v, batch_size_t)
    # size = batch_size
    size = scores.size(0)
    diagonal = scores.diag().view(size, 1)  # S(Ip,Tp)
    d1 = diagonal.expand_as(scores) # S_t(ap)
    d2 = diagonal.t().expand_as(scores) #S_i(ap)

    # compare every diagonal score to scores in its column
    # loss = margin - s(ap) + s(an), select hardest s(an) > s(ap) - margin
    # scores: s(an)
    # caption retrieval
    cost_s = (margin + scores - d1).clamp(min=0)
    # compare every diagonal score to scores in its row
    # image retrieval
    cost_im = (margin + scores - d2).clamp(min=0)

    mask = torch.eye(scores.size(0)) > .5
    I = Variable(mask)
    if torch.cuda.is_available():
        I = I.cuda()
    cost_s = cost_s.masked_fill_(I, 0)
    cost_im = cost_im.masked_fill_(I, 0)
    if neg == 'hardest':
        return cost_s.sum() + cost_im.sum(), 2
    elif neg == 'semi':
        # semi-hardest, select s(an) < s(ap), so let s(an)>s(ap)=0
        mask_s = Variable(scores > d1)
        mask_im = Variable(scores > d2)
        cost_s = cost_s.masked_fill_(mask_s, 0)
        cost_im = cost_im.masked_fill_(mask_im, 0)
        return cost_s.sum() + cost_im.sum(), 2 # loss/=(batch_size*2)
    elif neg == 'ohnm':
        return scores[0][1] - scores[0][0] + margin, 2

def select_hard(scores, margin):
    scores = scores.cpu()
    size = scores.size(0)
    diagonal = scores.diag().view(size, 1)  # S(Ip,Tp)
    d1 = diagonal.expand_as(scores) # S_t(ap)
    d2 = diagonal.t().expand_as(scores) #S_i(ap)

    # loss = margin - s(ap) + s(an), select hardest s(an) > s(ap) - margin
    # scores: s(an)
    cost_s = (margin + scores - d1).clamp(min=0)
    cost_im = (margin + scores - d2).clamp(min=0)

    mask = torch.eye(scores.size(0)) > .5
    I = mask
    cost_s = cost_s.masked_fill_(I, 0)
    cost_im = cost_im.masked_fill_(I, 0)
    hard_list = []
    s_list = []
    im_list = []
    # find anchor image and hardest neg text
    for i in range(cost_s.size(0)):
        hard_idx = torch.where(cost_s[i]>0)[0]
        if hard_idx.nelement() > 0:
            # we only select no more than 3 hard negative each anchor
            # hard_list.append([i, hard_idx[0].item()])
            idx = cost_s[i].argmax().item()
            s_list.append([i, idx])
    # find anchor text and hardest neg image
    cost_im = cost_im.t()
    for i in range(cost_im.size(0)):
        hard_idx = torch.where(cost_im[i]>0)[0]
        if hard_idx.nelement() > 0:
            idx = cost_im[i].argmax().item()
            im_list.append([i, idx])
    return s_list, im_list  # hard_list[i][0] = img_idx, hard_list[i][1] = text_idx

def select_trihard(scores):
    # max_s(an) - min_s(ap) + margin
    # here we select max_s(an) for each ap
    scores = scores.cpu()
    mask = torch.eye(scores.size(0)) > .5
    I = mask
    cost_s = scores.masked_fill_(I, 0)
    cost_im = scores.masked_fill_(I, 0).t()
    it_list = []
    ti_list = []
    # find anchor image and hardest neg text
    for i in range(cost_s.size(0)):
        # we only select the hardest negative sample
        it_list.append([i, cost_s[i].argmax().item()])
    # find anchor text and hardest neg image
    for i in range(cost_im.size(0)):
        # we only select the hardest negative sample
        ti_list.append([i, cost_im[i].argmax().item()])

    return it_list, ti_list  # hard_list[i][0] = img_idx, hard_list[i][1] = text_idx


def select_msml(scores):
    # max_s(mn) - min_s(ap) + margin
    # so we select max_s(mn) and min_s(ap) here
    scores = scores.cpu()
    size = scores.size(0)
    diagonal = scores.diag().view(size, 1)  # S(Ip,Tp)

    mask = torch.eye(scores.size(0)) > .5
    I = mask
    scores_mn = scores.masked_fill(I, 0)
    hard_set = set()
    # find min ap
    hard_ap = diagonal.argmin().item()
    hard_set.add(hard_ap)
    # find max mn
    hard_mn = np.unravel_index(scores_mn.argmax(), scores.size())
    hard_set.add(hard_mn[0])
    hard_set.add(hard_mn[1])
    hard_list = list(hard_set)
    return hard_list


def acc_i2t(input):
    """Computes the precision@k for the specified values of k of i2t"""
    image_size = input.shape[0]
    ranks = np.zeros(image_size)
    top1 = np.zeros(image_size)

    for index in range(image_size):
        # index of  descending similarity
        inds = np.argsort(input[index])[::-1]
        rank = np.where(inds == index)[0][0]
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr), (ranks, top1)


def acc_t2i(input):
    """Computes the precision@k for the specified values of k of i2t"""
    image_size = input.shape[0]
    ranks = np.zeros(image_size)
    top1 = np.zeros(image_size)
    input = input.T

    for index in range(image_size):
        # index of  descending similarity
        inds = np.argsort(input[index])[::-1]
        rank = np.where(inds == index)[0][0]
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr), (ranks, top1)
