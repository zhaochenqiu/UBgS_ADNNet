import sys
import torch

sys.path.append("/home/cqzhao/projects/matrix/")


import imageio

import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

import time


from common_py.dataIO import loadFiles_plus
from common_py.dataIO import loadImgs_pytorch
from common_py.dataIO import checkCreateDir

from common_py.dataIO import saveImg


from common_py.dataIO import readImg_byFilesIdx_pytorch


from common_py.evaluationBS import evaluation_numpy_entry_torch
from common_py.evaluationBS import evaluation_numpy_entry
from common_py.evaluationBS import evaluation_numpy
from common_py.evaluationBS import evaluation_BS

from common_py.utils import setupSeed

from common_py.improcess import img2pos

# from unet_q import UNet

from torch import optim

from params_input.params_input import QParams

from bayesian.bayesian import bayesRefine_iterative_gpu




def getImXFg(pa_im, ft_im, pa_fg, ft_fg, pa_gt, ft_gt, idx):

    fs_im, fullfs_im = loadFiles_plus(pa_im, ft_im)
    fs_fg, fullfs_fg = loadFiles_plus(pa_fg, ft_fg)
    fs_gt, fullfs_gt = loadFiles_plus(pa_gt, ft_gt)


    im = torch.tensor(imageio.imread(fullfs_im[idx])).unsqueeze(0)
    fg = torch.tensor(imageio.imread(fullfs_fg[idx])).unsqueeze(0)
    gt = torch.tensor(imageio.imread(fullfs_gt[idx])).unsqueeze(0)


    imXfg = torch.cat((im, fg.unsqueeze(-1)) , dim = -1)


    return imXfg, gt




def getImXFgSeq(pa_im, ft_im, pa_fg, ft_fg, pa_gt, ft_gt, idx_list):

    imXfg, gt = getImXFg(pa_im, ft_im, pa_fg, ft_fg, pa_gt, ft_gt, idx_list[0])

    num_list = len(idx_list)

    for i in range(1, num_list):
        idx = idx_list[i]

        imXfg_tmp, gt_tmp = getImXFg(pa_im, ft_im, pa_fg, ft_fg, pa_gt, ft_gt, idx)

        imXfg = torch.cat((imXfg, imXfg_tmp), dim = 0)
        gt = torch.cat((gt, gt_tmp), dim = 0)

#        print("idx = ", idx)

#    imXfg = imXfg.permute(0, 3, 1, 2)
#    gt = gt.unsqueeze(-1).permute(0, 3, 1, 2)

#     idx = gt < 250
#     gt[idx] = 0

    return imXfg, gt


def test_unet(imgs, device, network, len_batch):

    NUM_DATA = imgs.shape[0]

    NUM_BATCH = round(NUM_DATA/len_batch - 0.5)



    left = 0
    i = 0

    imgs_batch = imgs[left:left + len_batch]
    imgs_batch = imgs_batch.to(device, dtype=torch.float32)

    re_mask = network(imgs_batch).detach().cpu()
    left += len_batch


    for i in range(1, NUM_BATCH):


        imgs_batch = imgs[left:left + len_batch]
        imgs_batch = imgs_batch.to(device, dtype=torch.float32)

        mask = network(imgs_batch).detach().cpu()

        re_mask = torch.cat((re_mask, mask), dim = 0)


        left += len_batch


    if left < NUM_DATA:
        imgs_batch = imgs[left:NUM_DATA]
        imgs_batch = imgs_batch.to(device, dtype=torch.float32)

#         print("left = ", left)
#         print("NUM_DATA = ", NUM_DATA)
        mask = network(imgs_batch).detach().cpu()

        re_mask = torch.cat((re_mask, mask), dim = 0)


    return re_mask




def test_unet_gpu(imgs, device, network, len_batch):

    NUM_DATA = imgs.shape[0]

    NUM_BATCH = round(NUM_DATA/len_batch - 0.5)

    imgs = imgs.to(device, dtype = torch.float32)



    left = 0
    i = 0

    imgs_batch = imgs[left:left + len_batch]
#    imgs_batch = imgs_batch.to(device, dtype=torch.float32)

    re_mask = network(imgs_batch)
    left += len_batch


    for i in range(1, NUM_BATCH):


        imgs_batch = imgs[left:left + len_batch]
#        imgs_batch = imgs_batch.to(device, dtype=torch.float32)

        mask = network(imgs_batch)
        #.detach().cpu()

        re_mask = torch.cat((re_mask, mask), dim = 0)


        left += len_batch


    if left < NUM_DATA:
        imgs_batch = imgs[left:NUM_DATA]
#        imgs_batch = imgs_batch.to(device, dtype=torch.float32)

#         print("left = ", left)
#         print("NUM_DATA = ", NUM_DATA)
        mask = network(imgs_batch)
        #.detach().cpu()

        re_mask = torch.cat((re_mask, mask), dim = 0)


    return re_mask.detach().cpu()




def train_unet(imgs, labs, device, network, optimizer, loss_func, len_batch, num_epoch, net_pa):


    NUM_DATA = imgs.shape[0]

    NUM_BATCH = round(NUM_DATA/len_batch - 0.5)


    loss_list = []
    #torch.tensor([])



    fig = plt.figure(figsize = (4, 4))



    for i in range(num_epoch):

        idx_data = torch.randperm(NUM_DATA)

        left = 0

        total_loss = 0

        for j in range(NUM_BATCH):

            idx_batch = idx_data[left: left + len_batch]



            imgs_batch = imgs[idx_batch]
            labs_batch = labs[idx_batch]

            imgs_batch = imgs_batch.to(device, dtype=torch.float32)
            labs_batch = labs_batch.to(device, dtype=torch.long)


            mask_pred = network(imgs_batch)


            loss = loss_func(mask_pred, labs_batch.squeeze())

            print("epoch:", i, "/", num_epoch, "   loss:", loss.detach().cpu().item())

            total_loss += loss.detach().cpu().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            left += len_batch


        if left != NUM_DATA:
            idx_batch =idx_data[left:NUM_DATA]

            imgs_batch = imgs[idx_batch]
            labs_batch = labs[idx_batch]

            imgs_batch = imgs_batch.to(device, dtype=torch.float32)
            labs_batch = labs_batch.to(device, dtype=torch.long)


            mask_pred = network(imgs_batch)


            loss = loss_func(mask_pred, labs_batch.squeeze())

            #print("loss:", loss.detach().cpu().item())
            print("epoch:", i, "/", num_epoch, "   loss:", loss.detach().cpu().item())


            total_loss += loss.detach().cpu().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



        # save model
        epoch = i
        if epoch % 20 == 0:
            name_net = net_pa + "network_dis_" + str(epoch).zfill(4) + '.pt'

            checkCreateDir(name_net)
            torch.save(network.state_dict(), name_net)

            print("\n\n save model completed")


        loss_list.append(total_loss)

        x = np.linspace(0, i, num = i + 1)
        y = loss_list

        plt.plot(x, y, '.:', color = [0.1, 0.1, 0.1])

        plt.pause(0.01)



    return network




def getPatches(im, center, radius):


    if len(im.shape) < 3:
        im = im.unsqueeze(-1)


    row_im      = im.shape[0]
    column_im   = im.shape[1]


    tl = center - radius
    tl[tl < 0] = 0


    rb = tl + 2*radius


    rb[rb[:, 0] > row_im, 0]    = row_im
    rb[rb[:, 1] > column_im, 1] = column_im

    tl = rb - 2*radius


    tl = tl.long()
    rb = rb.long()
    num_patch = tl.shape[0]


    i = 0

#     print(tl)
#     print(tl[i, 0])
#     print(rb[i, 0])
#     print(tl[i, 1])
#     print(rb[i, 1])

    re_patches = im[tl[i, 0]:rb[i, 0], tl[i, 1]:rb[i, 1], :].unsqueeze(0)


    for i in range(1, num_patch):

        patches = im[tl[i, 0]:rb[i, 0], tl[i, 1]:rb[i, 1], :].unsqueeze(0)

        re_patches = torch.cat( (re_patches, patches), dim = 0)


    return re_patches



def img2patch_plus(im, mask, num_pos, num_neg, radius):

    pos = img2pos(im)

    idx_pos = mask == 255
    idx_neg = mask == 0

    pos_pos = pos[idx_pos, :]
    pos_neg = pos[idx_neg, :]


    LEN_pos = pos_pos.shape[0]
    LEN_neg = pos_neg.shape[0]


    idx = torch.randperm(LEN_pos)
    pos_pos = pos_pos[idx, :]

    idx = torch.randperm(LEN_neg)
    pos_neg = pos_neg[idx, :]


    flag = 0

    if num_pos <= LEN_pos:
        center_pos = pos_pos[0:num_pos, :]

        patches_pos = getPatches(im, center_pos, radius)
        mask_pos = getPatches(mask, center_pos, radius)

        re_patches = patches_pos
        re_mask = mask_pos

        flag += 1



    if num_neg <= LEN_neg:
        center_neg = pos_neg[0:num_neg, :]

        patches_neg = getPatches(im, center_neg, radius)
        mask_neg = getPatches(mask, center_neg, radius)

        re_patches = patches_neg
        re_mask = mask_neg

        flag += 1

    if flag == 2:
        re_patches = torch.cat((patches_pos, patches_neg), dim = 0 )
        re_mask = torch.cat((mask_pos, mask_neg), dim = 0)


    return re_patches, re_mask





def img2patch(im, mask, num_pos, num_neg):
    pos = img2pos(im)

    idx_pos = mask == 255
    idx_neg = mask == 0

    pos_pos = pos[idx_pos, :]
    pos_neg = pos[idx_neg, :]


    LEN_pos = pos_pos.shape[0]
    LEN_neg = pos_neg.shape[0]


    idx = torch.randperm(LEN_pos)
    pos_pos = pos_pos[idx, :]

    idx = torch.randperm(LEN_neg)
    pos_neg = pos_neg[idx, :]


    center_pos = pos_pos[0:num_pos, :]
    center_neg = pos_neg[0:num_neg, :]


#     print("im.shape = ", im.shape)
#     print("center_pos = ", center_pos)
#     print("pos_pos = ", pos_pos)
#     print("num test = ", pos_pos.shape[0])

    patches_pos = getPatches(im, center_pos, 60)
    patches_neg = getPatches(im, center_neg, 60)

    mask_pos = getPatches(mask, center_pos, 60)
    mask_neg = getPatches(mask, center_neg, 60)


    re_patches = torch.cat((patches_pos, patches_neg), dim = 0 )
    re_mask = torch.cat((mask_pos, mask_neg), dim = 0)


    return re_patches, re_mask




def imgseq2patches(imXfg, gt, num_pos, num_neg, radius):

    patches, mask = img2patch_plus(imXfg[0], gt[0], num_pos, num_neg, radius)

    frames = imXfg.shape[0]


    for i in range(1,frames):
        patches_temp, mask_temp = img2patch_plus(imXfg[i], gt[i], num_pos, num_neg, radius)

        patches = torch.cat((patches, patches_temp), dim = 0)
        mask    = torch.cat((mask, mask_temp),       dim = 0)


    return patches, mask


def imgseq2patches_seedfill(list_imXfg, list_gt, radius):

#    starttime = time.time()
    lt_list_im, patches = img2patches_seedfill(list_imXfg[0], radius)
#    endtime = time.time()
#    print("time = ", endtime - starttime)

#     starttime = time.time()
#     lt_list_im1, patches1 = img2patches_seedfill_fast(list_imXfg[0], radius)
#     endtime = time.time()
#     print("time = ", endtime - starttime)

#     print("borderline ================================")
#
#     print(lt_list_im)
#     print(lt_list_im1)
#
#     print(torch.sum(lt_list_im - lt_list_im1))
#     print(torch.sum(patches - patches1))
#
#     print("borderline -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
#
#
#
#     stopins = input()


#    print(list_imXfg[0].shape)
#    print(list_gt[0].shape)

    lt_list_gt, mask = img2patches_seedfill(list_gt[0].unsqueeze(-1), radius)

    frames = list_imXfg.shape[0]


    for i in range(1,frames):

        lt_list_im, patches_temp = img2patches_seedfill(list_imXfg[i], radius)
        lt_list_gt, mask_temp = img2patches_seedfill(list_gt[i].unsqueeze(-1), radius)

        # patches_temp, mask_temp = img2patch_plus(imXfg[i], gt[i], num_pos, num_neg, radius)

        patches = torch.cat((patches, patches_temp), dim = 0)
        mask    = torch.cat((mask, mask_temp),       dim = 0)


    return patches, mask










def poscheck(r,c,row_im, column_im, radius):

    if r + radius > row_im:
        r = row_im - radius

    if c + radius > column_im:
        c = column_im - radius

    return r, c





def patches2img_seedfill(patches, lt_list, radius):

    row_im      = torch.max(lt_list[:, 0]) + radius
    column_im   = torch.max(lt_list[:, 1]) + radius


    byte_im = 1
    if len(patches.squeeze().shape) == 4:
        byte_im     = patches.shape[3]


    re_im = torch.zeros(row_im, column_im, byte_im).squeeze()

    num = lt_list.shape[0]


    for i in range(num):
        r = lt_list[i, 0]
        c = lt_list[i, 1]

        re_im[r:r + radius, c:c + radius] = patches[i].squeeze()


    return re_im





def img2patches_seedfill_slow(im, radius):

    row_im     = im.shape[0]
    column_im  = im.shape[1]


    canvas = torch.zeros(row_im, column_im)


    lt_list = torch.tensor([0, 0]).unsqueeze(0)


    canvas[0:radius, 0:radius] = 1
    re_patches = im[0:radius, 0:radius, :].unsqueeze(0)


    for r in range(row_im):
        for c in range(column_im):

            if canvas[r,c] == 0:
                r, c = poscheck(r, c, row_im, column_im, radius)

                canvas[r:r + radius, c:c + radius] = 1
                patches_tmp = im[r:r + radius, c:c + radius, :].unsqueeze(0)

                re_patches = torch.cat((re_patches, patches_tmp), dim = 0)


                pos = torch.tensor([r, c])
                lt_list = torch.cat( (lt_list, pos.unsqueeze(0)), dim = 0 )


    return lt_list, re_patches





def img2patches_seedfill(im, radius):

    row_im     = im.shape[0]
    column_im  = im.shape[1]


    len_r = round(row_im/radius + 0.5)
    len_c = round(column_im/radius + 0.5)


    for i in range(len_r):
        for j in range(len_c):
            if (i + j) == 0:
                r, c = poscheck(i * radius, j * radius, row_im, column_im, radius)

                pos = torch.tensor([r, c]).unsqueeze(0)
                lt_list = pos

                re_patches = im[0:radius, 0:radius, :].unsqueeze(0)

            else:

                r, c = poscheck(i * radius, j * radius, row_im, column_im, radius)

                pos = torch.tensor([r, c])
                lt_list = torch.cat( (lt_list, pos.unsqueeze(0)), dim = 0 )

                patches_tmp = im[r:r + radius, c:c + radius, :].unsqueeze(0)

                re_patches = torch.cat((re_patches, patches_tmp), dim = 0)

    return lt_list, re_patches


#     canvas = torch.zeros(row_im, column_im)
#
#
#
#
#     canvas[0:radius, 0:radius] = 1
#     re_patches = im[0:radius, 0:radius, :].unsqueeze(0)
#
#
#
#
#     r = radius
#     while r < row_im:
#
#         c = radius
#         while c < column_im:
#
#             if canvas[r,c] == 0:
#                 r, c = poscheck(r, c, row_im, column_im, radius)
#
#                 canvas[r:r + radius, c:c + radius] = 1
#                 patches_tmp = im[r:r + radius, c:c + radius, :].unsqueeze(0)
#
#                 re_patches = torch.cat((re_patches, patches_tmp), dim = 0)
#
#
#                 pos = torch.tensor([r, c])
#                 lt_list = torch.cat( (lt_list, pos.unsqueeze(0)), dim = 0 )
#
#             c += radius
#             print("c = ", c)
#
#
# #            c += 1
#
#         r += radius
#         print("r = ", r)
#
#     return lt_list, re_patches



#     for r in range(row_im):
#         for c in range(column_im):
#
#             if canvas[r,c] == 0:
#                 r, c = poscheck(r, c, row_im, column_im, radius)
#
#                 canvas[r:r + radius, c:c + radius] = 1
#                 patches_tmp = im[r:r + radius, c:c + radius, :].unsqueeze(0)
#
#                 re_patches = torch.cat((re_patches, patches_tmp), dim = 0)
#
#
#                 pos = torch.tensor([r, c])
#                 lt_list = torch.cat( (lt_list, pos.unsqueeze(0)), dim = 0 )
#
#
#     return lt_list, re_patches






def test_unet_ImXFg(network, device, pa_im, ft_im, pa_fg, ft_fg, pa_gt, ft_gt, idx, radius, len_batch):

    imXfg, gt = getImXFg(pa_im, ft_im, pa_fg, ft_fg, pa_gt, ft_gt, idx)
    imXfg = imXfg.squeeze()


    lt_list, patches = img2patches_seedfill(imXfg, radius)


    patches = patches.permute(0, 3, 1, 2)


    output = test_unet(patches, device, network, len_batch)

    output = output.argmax(dim = 1, keepdim = True).cpu().detach().squeeze()


    fgimg = patches2img_seedfill(output , lt_list, radius)

    return fgimg




def videos2patches(pa_im_list, pa_fg_list, pa_gt_list, idx_list, ft_im, ft_fg, ft_gt, radius, num_pos, num_neg):

    # imXfg_winterDriveway, gt_winterDriveway = getImXFgSeq(pa_im, ft_im, pa_fg, ft_fg, pa_gt, ft_gt, idx_list_winterDriveway)

    num = 0

    for pa_im in pa_im_list:
        num = num + 1


    i = 0
    pa_im = pa_im_list[i]
    pa_fg = pa_fg_list[i]
    pa_gt = pa_gt_list[i]

    idx = idx_list[i]

    imXfg, gt = getImXFgSeq(pa_im, ft_im, pa_fg, ft_fg, pa_gt, ft_gt, idx)
    re_patches, re_mask = imgseq2patches(imXfg, gt, num_pos, num_neg, radius)


    for i in range(1, num):

        pa_im = pa_im_list[i]
        pa_fg = pa_fg_list[i]
        pa_gt = pa_gt_list[i]

        idx = idx_list[i]

        imXfg, gt = getImXFgSeq(pa_im, ft_im, pa_fg, ft_fg, pa_gt, ft_gt, idx)

        patches, mask = imgseq2patches(imXfg, gt, num_pos, num_neg, radius)

        re_patches = torch.cat((re_patches, patches), dim = 0)
        re_mask = torch.cat((re_mask, mask), dim = 0)


    return re_patches, re_mask



def videos2patches_seedfill(pa_im_list, pa_fg_list, pa_gt_list, idx_list, ft_im, ft_fg, ft_gt, radius):

    # imXfg_winterDriveway, gt_winterDriveway = getImXFgSeq(pa_im, ft_im, pa_fg, ft_fg, pa_gt, ft_gt, idx_list_winterDriveway)

    num = 0

    for pa_im in pa_im_list:
        num = num + 1


    i = 0
    pa_im = pa_im_list[i]
    pa_fg = pa_fg_list[i]
    pa_gt = pa_gt_list[i]

    idx = idx_list[i]

    imXfg, gt = getImXFgSeq(pa_im, ft_im, pa_fg, ft_fg, pa_gt, ft_gt, idx)
    re_patches, re_mask = imgseq2patches_seedfill(imXfg, gt, radius)



    for i in range(1, num):

        pa_im = pa_im_list[i]
        pa_fg = pa_fg_list[i]
        pa_gt = pa_gt_list[i]

        idx = idx_list[i]

        imXfg, gt = getImXFgSeq(pa_im, ft_im, pa_fg, ft_fg, pa_gt, ft_gt, idx)
        patches, mask = imgseq2patches_seedfill(imXfg, gt, radius)

        re_patches = torch.cat((re_patches, patches), dim = 0)
        re_mask = torch.cat((re_mask, mask), dim = 0)


    return re_patches, re_mask





# def imgseq2patches_seedfill(list_imXfg, list_gt, radius):

def detectFgImg(pa_im, ft_im, pa_fg, ft_fg, pa_gt, ft_gt, radius, idx, network, device, len_batch):



#     idx_list = [    [1700, 1810, 1832, 1850, 1873, 1881],
#                     [718, 733, 840, 1125, 1129, 1160],
#                     [805, 843, 1426, 2764, 2814]]
#
#
#
#
#     i = 0
#     pa_im = pa_im_list[i]
#     pa_fg = pa_fg_list[i]
#     pa_gt = pa_gt_list[i]
#
#     idx = idx_list[i]

#     imXfg, gt = getImXFgSeq(pa_im, ft_im, pa_fg, ft_fg, pa_gt, ft_gt, idx)
#
#     print(imXfg[0].shape)
#
#
#     idx = 1700
    imXfg, gt = getImXFg(pa_im, ft_im, pa_fg, ft_fg, pa_gt, ft_gt, idx)


#    radius = 128
#    print(imXfg.shape)

    imXfg = imXfg.squeeze()
    gt = gt.squeeze()

    starttime = time.time()
    lt_list, patches = img2patches_seedfill(imXfg, radius)
    endtime = time.time()
    print("img2patches_seedfill:", endtime - starttime)


#    print(imXfg.shape)
    patches = patches.permute(0, 3, 1, 2)
    patches = patches.float()/255.0



#    len_batch = 5
    starttime = time.time()
    re_mask = test_unet_gpu(patches, device, network, len_batch)
    endtime = time.time()

    print("test_unet_gpu:", endtime - starttime)


    starttime = time.time()
    refinefg = patches2img_seedfill(re_mask.permute(0, 2, 3, 1), lt_list, radius)
    endtime = time.time()
    print("patches2img_seedfill:", endtime - starttime)

    # refinemask = re_mask.argmax(dim = 1, keepdim = True).cpu().detach().squeeze()
    refinemask = refinefg.argmax(dim = 2, keepdim = True).cpu().detach().squeeze()


    return refinemask, gt, imXfg, patches

#     print(re_mask.shape)
#     print(refinemask.shape)
#
#     plt.figure(figsize = (9,3))
#     plt.imshow(refinemask)
#     plt.show()


def checkTrainList(idx, list_idx):

    idx_t = torch.tensor(idx)
    list_idx_t = torch.tensor(list_idx)

    list_sub = list_idx_t - idx_t

    judge = torch.sum(list_sub == 0)

    return judge



def evaluateRefinement(pa_im, ft_im, pa_fg, ft_fg, pa_gt, ft_gt, radius, device, len_batch, list_train, flag_in, pa_out):

    sum_TP = 0
    sum_FP = 0
    sum_TN = 0
    sum_FN = 0

    sum_TP_rf = 0
    sum_FP_rf = 0
    sum_TN_rf = 0
    sum_FN_rf = 0

    radius = 2
    rate = 0.8
    iter_num = 20



    fs, fullfs = loadFiles_plus(pa_gt, ft_gt)
    frames = len(fullfs)

    plt.figure(figsize = (4,4))
    for i in range(frames):

        frame_idx_i = i


        starttime = time.time()

        mask_gt = torch.tensor( imageio.imread(fullfs[i]), dtype = torch.float)

        judge_flag = torch.sum(mask_gt == 255) + torch.sum(mask_gt == 0)


        judge_train = checkTrainList(i, list_train)

        if judge_train != 0 and flag_in == 1:
            judge_flag = 0
            print("This frame is included in training set:", i)
            print("list_train:", list_train)


        if judge_flag == 0:
            refinemask = mask_gt
            rfim = refinemask
        else:
#            starttime = time.time()

            im = readImg_byFilesIdx_pytorch(i, pa_im, ft_im)
            fgim = readImg_byFilesIdx_pytorch(i, pa_fg, ft_fg)
            gtim = readImg_byFilesIdx_pytorch(i, pa_gt, ft_gt)


            rfim = bayesRefine_iterative_gpu(im, fgim, radius, rate, iter_num, device)



#
#             refinemask, gt, imXfg, patches_tmp = detectFgImg(pa_im, ft_im, pa_fg, ft_fg, pa_gt, ft_gt, radius, i, network, device, len_batch)
# #            endtime = time.time()
#
# #            print("network processing time:", endtime - starttime)
#
#             srim = imXfg[:, :, 0:3].int()
#             fgim = imXfg[:, :, 3].int()
#             rfim = (refinemask*255).int()
#             gtim = gt.int()
#



            TP, FP, TN, FN = evaluation_numpy_entry_torch(fgim, gtim)



            Re = TP/max((TP + FN), 1)
            Pr = TP/max((TP + FP), 1)

            Fm = (2*Pr*Re)/max((Pr + Re), 0.0001)


            print("\n")
            print("==========================================================================================")
            print("pa_im:", pa_im)
            print(fullfs[i])
            print("frames:", i)
            print("current original:")
            print("Re:", Re)
            print("Pr:", Pr)
            print("Fm:", Fm)

            sum_TP += TP
            sum_FP += FP
            sum_TN += TN
            sum_FN += FN

            print("Current Entry:", TP, " ", FP, " ", TN, " ", FN)
            print("Current Sum Entry:", sum_TP, " ", sum_FP, " ", sum_TN, " ", sum_FN)

            TP, FP, TN, FN = evaluation_numpy_entry_torch(rfim.detach().cpu(), gtim.detach().cpu())



            #TP, FP, TN, FN = evaluation_numpy_entry(rfim.numpy(), gtim.numpy())


            Re = TP/max((TP + FN), 1)
            Pr = TP/max((TP + FP), 1)

            Fm = (2*Pr*Re)/max((Pr + Re), 0.0001)


            print("\n current refinefg:")
            print("Re:", Re)
            print("Pr:", Pr)
            print("Fm:", Fm)

            sum_TP_rf += TP
            sum_FP_rf += FP
            sum_TN_rf += TN
            sum_FN_rf += FN

            print("Current RF Entry:", TP, " ", FP, " ", TN, " ", FN)
            print("Current RF Sum Entry:", sum_TP_rf, " ", sum_FP_rf, " ", sum_TN_rf, " ", sum_FN_rf)


    #            endtime = time.time()
    #            print("evaluation time:", endtime - starttime)




            print("\n---------------------------------------------\n")
            Re_sum = sum_TP/max((sum_TP + sum_FN), 1)
            Pr_sum = sum_TP/max((sum_TP + sum_FP), 1)

            Fm_sum = (2*Pr_sum*Re_sum)/max((Pr_sum + Re_sum), 0.0001)

            print("accumulate original:")
            print("Re_sum:", Re_sum)
            print("Pr_sum:", Pr_sum)
            print("Fm_sum:", Fm_sum)

            Re_sum_rf = sum_TP_rf/max((sum_TP_rf + sum_FN_rf), 1)
            Pr_sum_rf = sum_TP_rf/max((sum_TP_rf + sum_FP_rf), 1)

            Fm_sum_rf = (2*Pr_sum_rf*Re_sum_rf)/max((Pr_sum_rf + Re_sum_rf), 0.0001)

            endtime = time.time()

            print("\n accumulate refinefg:")
            print("Re_sum_rf:", Re_sum_rf)
            print("Pr_sum_rf:", Pr_sum_rf)
            print("Fm_sum_rf:", Fm_sum_rf)

            print("==========================================================================================")
            print("total time:", endtime - starttime)
            print("\n\n")



        rfim = torch.tensor(rfim)
        filename = pa_out + fs[frame_idx_i]

        print("saving rfim:", filename)
        saveImg(filename, rfim.detach().cpu().numpy().astype(np.uint8))







def main(argc, argv):


    setupSeed(0)

    qparams = QParams()
    qparams.setParams(argc, argv)

    gpuid = qparams['gpuid']



    pa_out = qparams['pa_out']


    pa_out = pa_out
    #+ str_version + '/'



    ft_im = 'jpg'
    ft_fg = 'png'
    ft_gt = 'png'

    num_pos = 10
    num_neg = 10

    radius = 64


#     print(patches.shape)
#     print(patches_seed.shape)
#
#     print(gt.shape)
#     print(gt_seed.shape)




#    patches = patches_seed
#    gt = gt_seed





#    tempcommand = input("stop here")






    # training the network

    device = torch.device( ("cuda:" + str(gpuid)) if torch.cuda.is_available() else 'cpu')

#     network = UNet(n_channels=4, n_classes=2, bilinear=True).to(device)
#
#     filename = net_pa + "network_dis_" + str(int(idx_net)).zfill(4) + ".pt"
#     network.load_state_dict(torch.load(filename))


    len_batch = 10
    num_epoch = 60

#    fluidHighway/'

#    network = train_unet(patches, gt, device, network, optimizer, loss_func, len_batch, num_epoch, net_pa)





    print("running evaluateRefinement ...")
    ft_im = qparams['ft_im']
    ft_gt = qparams['ft_gt']


    pa_im = qparams['pa_im']
    pa_fg = qparams['pa_fg']
    pa_gt = qparams['pa_gt']

    list_train = qparams['imgs_idx']
    list_train[:] = [i - 1 for i in list_train]





    len_batch = 20

    evaluateRefinement(pa_im, ft_im, pa_fg, ft_fg, pa_gt, ft_gt, 2*radius, device, len_batch, list_train, 1, pa_out)









if __name__ == '__main__':

    argc = len(sys.argv)
    argv = sys.argv

    main(argc, argv)

