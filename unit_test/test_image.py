'''
SSIM 和 MS_SSIM 的更多测试函数在 ssim.py 内
'''

import unittest
import cv2
import numpy as np
from model_utils_torch.image import *


class TestImage(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _check_grad_nan(self, a: torch.Tensor):
        self.assertTrue(torch.isnan(a.grad).sum().item() == 0, 'Found nan!')

    def test_color(self):
        im = cv2.imread('data/test_img1.jpg', 1)
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

        cv2.waitKey(15)
        cv2.destroyAllWindows()

    def test_ssim_simple(self):
        print('Simple Test')
        im = torch.randint(0, 255, (5, 3, 256, 256), dtype=torch.float, device='cuda')
        img1 = im / 255
        img2 = img1 * 0.5

        losser = SSIM(data_range=1.).cuda()
        loss = losser(img1, img2).mean()

        losser2 = MS_SSIM(data_range=1.).cuda()
        loss2 = losser2(img1, img2).mean()

        print(loss.item())
        print(loss2.item())

    def test_ssim_nan(self):
        print('Test nan')
        losser1 = SSIM(data_range=255.).cuda()
        losser2 = MS_SSIM(data_range=255.).cuda()
        losser3 = SSIM(data_range=1.).cuda()
        losser4 = MS_SSIM(data_range=1.).cuda()

        for _ in range(500):
            im1 = torch.randint(0, 255, (10, 3, 256, 256), dtype=torch.float32, device='cuda')
            im2 = torch.randint(0, 255, (10, 3, 256, 256), dtype=torch.float32, device='cuda')
            # im2 = torch.zeros(10, 3, 256, 256, dtype=torch.float32, device='cuda')
            im3 = im1 / 255.
            im4 = im2 / 255.
            im1.requires_grad = True
            im2.requires_grad = True
            im3.requires_grad = True
            im4.requires_grad = True

            # for data_range 255
            loss = losser1(im1, im2).mean()
            loss.backward()
            self.assertTrue(torch.isnan(loss).sum() == 0, 'Found nan!')
            self._check_grad_nan(im1)
            self._check_grad_nan(im2)
            del im1.grad
            del im2.grad

            loss = losser2(im1, im2).mean()
            loss.backward()
            self.assertTrue(torch.isnan(loss).sum() == 0, 'Found nan!')
            self._check_grad_nan(im1)
            self._check_grad_nan(im2)
            del im1.grad
            del im2.grad

            # for data_range 1.
            loss = losser3(im3, im4).mean()
            loss.backward()
            self.assertTrue(torch.isnan(loss).sum() == 0, 'Found nan!')
            self._check_grad_nan(im3)
            self._check_grad_nan(im4)
            del im3.grad
            del im4.grad

            loss = losser4(im3, im4).mean()
            loss.backward()
            self.assertTrue(torch.isnan(loss).sum() == 0, 'Found nan!')
            self._check_grad_nan(im3)
            self._check_grad_nan(im4)
            del im3.grad
            del im4.grad

    def test_ssim_training(self):
        print('Training Test')
        import cv2
        import torch.optim
        import numpy as np
        import imageio
        import time

        out_test_video = False
        # 最好不要直接输出gif图，会非常大，最好先输出mkv文件后用ffmpeg转换到GIF
        video_use_gif = False

        im = cv2.imread('data/test_img1.jpg', 1)
        t_im = torch.from_numpy(im).cuda().permute(2, 0, 1).float()[None] / 255.

        if out_test_video:
            if video_use_gif:
                fps = 0.5
                out_wh = (im.shape[1] // 2, im.shape[0] // 2)
                suffix = '.gif'
            else:
                fps = 5
                out_wh = (im.shape[1], im.shape[0])
                suffix = '.mkv'
            video_last_time = time.perf_counter()
            video = imageio.get_writer('ssim_test' + suffix, fps=fps)

        # 测试ssim
        print('Training SSIM')
        rand_im = torch.randint_like(t_im, 0, 255, dtype=torch.float32) / 255.
        rand_im.requires_grad = True
        optim = torch.optim.Adam([rand_im], 0.003, eps=1e-8)
        losser = SSIM(data_range=1., channel=t_im.shape[1]).cuda()
        ssim_score = 0
        while ssim_score < 0.999:
            optim.zero_grad()
            loss = losser(rand_im, t_im)
            (-loss).sum().backward()
            ssim_score = loss.item()
            optim.step()
            r_im = np.transpose(rand_im.detach().cpu().numpy().clip(0, 1) * 255, [0, 2, 3, 1]).astype(np.uint8)[0]
            r_im = cv2.putText(cv2.UMat(r_im), 'ssim %f' % ssim_score, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2).get()

            if out_test_video:
                if time.perf_counter() - video_last_time > 1. / fps:
                    video_last_time = time.perf_counter()
                    out_frame = cv2.cvtColor(r_im, cv2.COLOR_BGR2RGB)
                    out_frame = cv2.resize(out_frame, out_wh, interpolation=cv2.INTER_AREA)
                    if isinstance(out_frame, cv2.UMat):
                        out_frame = out_frame.get()
                    video.append_data(out_frame)

            cv2.imshow('ssim', r_im)
            cv2.setWindowTitle('ssim', 'ssim %f' % ssim_score)
            cv2.waitKey(1)

        if out_test_video:
            video.close()

        # 测试ms_ssim
        if out_test_video:
            if video_use_gif:
                fps = 0.5
                out_wh = (im.shape[1] // 2, im.shape[0] // 2)
                suffix = '.gif'
            else:
                fps = 5
                out_wh = (im.shape[1], im.shape[0])
                suffix = '.mkv'
            video_last_time = time.perf_counter()
            video = imageio.get_writer('ms_ssim_test' + suffix, fps=fps)

        print('Training MS_SSIM')
        rand_im = torch.randint_like(t_im, 0, 255, dtype=torch.float32) / 255.
        rand_im.requires_grad = True
        optim = torch.optim.Adam([rand_im], 0.003, eps=1e-8)
        losser = MS_SSIM(data_range=1., channel=t_im.shape[1]).cuda()
        ssim_score = 0
        while ssim_score < 0.999:
            optim.zero_grad()
            loss = losser(rand_im, t_im)
            (-loss).sum().backward()
            ssim_score = loss.item()
            optim.step()
            r_im = np.transpose(rand_im.detach().cpu().numpy().clip(0, 1) * 255, [0, 2, 3, 1]).astype(np.uint8)[0]
            r_im = cv2.putText(cv2.UMat(r_im), 'ms_ssim %f' % ssim_score, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2).get()

            if out_test_video:
                if time.perf_counter() - video_last_time > 1. / fps:
                    video_last_time = time.perf_counter()
                    out_frame = cv2.cvtColor(r_im, cv2.COLOR_BGR2RGB)
                    out_frame = cv2.resize(out_frame, out_wh, interpolation=cv2.INTER_AREA)
                    if isinstance(out_frame, cv2.UMat):
                        out_frame = out_frame.get()
                    video.append_data(out_frame)

            cv2.imshow('ms_ssim', r_im)
            cv2.setWindowTitle('ms_ssim', 'ms_ssim %f' % ssim_score)
            cv2.waitKey(1)

        if out_test_video:
            video.close()

        cv2.waitKey(15)
        cv2.destroyAllWindows()

    def test_ssim_perf(self):
        print('Performance Testing SSIM')
        import time
        s = SSIM(data_range=1.).cuda()

        a = torch.randint(0, 255, size=(20, 3, 256, 256), dtype=torch.float32).cuda() / 255.
        b = a * 0.5
        a.requires_grad = True
        b.requires_grad = True

        start_record = torch.cuda.Event(enable_timing=True)
        end_record = torch.cuda.Event(enable_timing=True)

        start_time = time.perf_counter()
        start_record.record()
        for _ in range(500):
            loss = s(a, b).mean()
            loss.backward()
        end_record.record()
        end_time = time.perf_counter()

        torch.cuda.synchronize()

        print('cuda time', start_record.elapsed_time(end_record))
        print('perf_counter time', end_time - start_time)

        print('Performance Testing MS_SSIM')
        import time
        s = MS_SSIM(data_range=1.).cuda()

        a = torch.randint(0, 255, size=(20, 3, 256, 256), dtype=torch.float32).cuda() / 255.
        b = a * 0.5
        a.requires_grad = True
        b.requires_grad = True

        start_record = torch.cuda.Event(enable_timing=True)
        end_record = torch.cuda.Event(enable_timing=True)

        start_time = time.perf_counter()
        start_record.record()
        for _ in range(500):
            loss = s(a, b).mean()
            loss.backward()
        end_record.record()
        end_time = time.perf_counter()

        torch.cuda.synchronize()

        print('cuda time', start_record.elapsed_time(end_record))
        print('perf_counter time', end_time - start_time)
