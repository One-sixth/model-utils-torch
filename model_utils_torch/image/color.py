import torch


def RGB2Gray(x):
    '''
    RGB转灰度图，要求值域为0-255
    :param x: (N, 3, H, W)
    :return: (N, 1, H, W)
    '''
    R, G, B = torch.chunk(x, 3, 1)
    return 0.30 * R + 0.59 * G + 0.11 * B


def RGB2YUV(x):
    '''
    RGB转YUV，要求值域为0-255
    :param x: (N, 3, H, W)
    :return: (N, 3, H, W)
    '''
    R, G, B = torch.chunk(x, 3, 1)
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = 0.492 * (B - Y) + 128
    V = 0.877 * (R - Y) + 128
    return torch.cat([Y, U, V], dim=1)


def YUV2RGB(x):
    '''
    YUV转RGB，要求值域为0-255
    :param x:
    :return:
    '''
    Y, U, V = torch.chunk(x, 3, 1)
    R = Y + 1.140 * (V - 128)
    G = Y - 0.394 * (U - 128) - 0.581 * (V - 128)
    B = Y + 2.032 * (U - 128)
    return torch.cat([R, G, B], dim=1)


if __name__ == '__main__':
    import cv2
    import numpy as np

    im = cv2.imread('test_img1.jpg', 1)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    t1 = torch.from_numpy(im).permute(2, 0, 1).float()[None]
    t_gray = RGB2Gray(t1)
    t_yuv = RGB2YUV(t1)
    t_rgb = YUV2RGB(t_yuv)
    im_gray = t_gray.permute(0, 2, 3, 1)[0].numpy().astype(np.uint8)
    im_yuv = t_yuv.permute(0, 2, 3, 1)[0].numpy().astype(np.uint8)
    im_rgb = t_rgb.permute(0, 2, 3, 1)[0].numpy().astype(np.uint8)

    im_yuv = cv2.cvtColor(im_yuv, cv2.COLOR_YUV2BGR)
    im_rgb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2BGR)

    cv2.imshow('im_gray', im_gray)
    cv2.imshow('im_yuv', im_yuv)
    cv2.imshow('im_rgb', im_rgb)

    cv2.waitKey()
