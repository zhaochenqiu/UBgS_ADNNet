import torch

from skimage import color
import torch.nn.functional as F



def img2pos(im):
    row_im, column_im, byte_im = im.shape

    re_pos = torch.abs(im - im)
    re_pos = re_pos[:, :, 0:2]


    pos_r = torch.linspace(0, row_im - 1, row_im ).expand(column_im, row_im).t()
    pos_c = torch.linspace(0, column_im - 1, column_im).expand(row_im, column_im)

    return torch.cat((pos_r.unsqueeze(-1), pos_c.unsqueeze(-1)), dim = 2)





def rgb2lab(im):

    im = im/255.0
    lab = color.rgb2lab(im)

    lab = torch.tensor( lab, dtype = torch.float )



    return lab


def isnan(x):
    return x != x




def imPatching(im, radius = 10):

    row_im, column_im, byte_im = im.shape
#    radius = 10

    exim = F.pad(im.permute(2, 0, 1).unsqueeze(0), (radius, radius, radius, radius), mode='replicate').squeeze()
    exim = exim.permute(1, 2, 0)

    imgs_vec = torch.cat([exim[i:i + row_im, j:j + column_im].reshape(row_im*column_im, byte_im).unsqueeze(-1) for i in range(radius*2 + 1) for j in range(radius*2 + 1)], dim = 2)

    nums, byte, cha = imgs_vec.shape
    imgs_vec = imgs_vec.reshape(row_im, column_im, byte, 2*radius + 1, 2*radius + 1)

    return imgs_vec.permute(0, 1, 3, 4, 2)



