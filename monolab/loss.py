import torch
import torch.nn as nn
import torch.nn.functional as F


class MonodepthLoss(nn.modules.Module):
    def __init__(self, device, SSIM_w=0.85, disp_gradient_w=0.1, lr_w=1.0):
        super(MonodepthLoss, self).__init__()
        self.device = device
        self.SSIM_w = SSIM_w
        self.disp_gradient_w = disp_gradient_w
        self.lr_w = lr_w
        self.n = None

    def scale_pyramid(self, img, num_scales):
        """ Compute a pyramid of an image at different scales.
        If the original image dimension is (n,m), the i-th element of the pyramid is (n/2**i, m/2**i)

        Args:
            img: (n_batch, n_dim, nx, ny) the original scale image
            num_scales: number of scales

        Returns:
            A list of the image at decreasing sizes, starting with the original image.
        """
        scaled_imgs = [img]
        s = img.size()
        h = s[2]
        w = s[3]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(
                nn.functional.interpolate(
                    img, size=[nh, nw], mode="area", align_corners=None
                )
            )
        return scaled_imgs

    def gradient_x(self, img):
        """ Computes an image gradient in x direction

        Args:
            img: (n_batch, n_dim, nx, ny) input image

        Returns:
            the image gradient (same shape as input)
        """
        # Pad input to keep output size consistent
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
        return gx

    def gradient_y(self, img):
        """ Computes an image gradient in y direction

        Args:
            img: (n_batch, n_dim, nx, ny) input image

        Returns:
            the image gradient (same shape as input)
        """
        # Pad input to keep output size consistent
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
        return gy

    def apply_disparity(self, img, disp):
        """ Applies a disparity map to an image.

        Args:
            img: (n_batch, n_dim, nx, ny) input image
            disp: (n_batch, 1, nx, ny) disparity map to be applied

        Returns:
            the input image shifted by the disparity map (n_batch, n_dim, nx, ny)
        """
        batch_size, _, height, width = img.size()

        # Original coordinates of pixels
        x_base = (
            torch.linspace(0, 1, width)
            .repeat(batch_size, height, 1)
            .type_as(img)
            .to(self.device)
        )
        y_base = (
            torch.linspace(0, 1, height)
            .repeat(batch_size, width, 1)
            .transpose(1, 2)
            .type_as(img)
        ).to(self.device)

        # Apply shift in X direction
        x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
        flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
        # In grid_sample coordinates are assumed to be between -1 and 1
        output = F.grid_sample(
            img, 2 * flow_field - 1, mode="bilinear", padding_mode="zeros"
        )

        return output

    def generate_image_left(self, img, disp):
        """ Apply a left disparity map to a right image

        Args:
            img: (n_batch, n_dim, nx, ny) right input image
            disp: (n_batch, 1, nx, ny) disparity map to be applied

        Returns:
            the left image (n_batch, n_dim, nx, ny)
        """
        return self.apply_disparity(img, -disp)

    def generate_image_right(self, img, disp):
        """ Apply a left disparity map to a left image

        Args:
            img: (n_batch, n_dim, nx, ny) left input image
            disp: (n_batch, 1, nx, ny) disparity map to be applied

        Returns:
            the right image (n_batch, n_dim, nx, ny)
        """
        return self.apply_disparity(img, disp)

    def SSIM(self, x, y):
        """ Compute the structural similarity index between two images

        Args:
            x: (n_batch, n_dim, nx, ny) input image
            y: (n_batch, n_dim, nx, ny) input image

        Returns:
            (float) structural similarity measure
        """
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = nn.AvgPool2d(3, 1)(x)
        mu_y = nn.AvgPool2d(3, 1)(y)
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
        sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
        sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

        SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
        SSIM = SSIM_n / SSIM_d

        return torch.clamp((1 - SSIM) / 2, 0, 1)

    def disp_smoothness(self, disp, pyramid):
        """ Compute the smoothness of the disparity map (defined by its gradient),
            weighted by the neg. exp. of the image gradient.

        Args:
            disp: [disp1, disp2, disp3, disp4]
            pyramid: [img1, img2, img3, img4] images at different scales

        Returns:
            [smooth1, smooth2, smooth3, smooth4] smoothness maps
        """
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]

        weights_x = [
            torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True))
            for g in image_gradients_x
        ]
        weights_y = [
            torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True))
            for g in image_gradients_y
        ]

        smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(self.n)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(self.n)]

        return smoothness_x + smoothness_y


    def forward(self, input, target):
        """ Compute the loss, given disparity maps at 4 scales and left and right input images

        Args:
            input: [disp1, disp2, disp3, disp4], each (n_batch, 1, nx, ny)
            target: [left, right], each (n_batch, 3, nx, ny)

        Return:
            (float): The loss
        """

        ################
        # Preparations #
        ################
        self.n = len(input)

        left, right = target
        left_pyramid = self.scale_pyramid(left, self.n)
        right_pyramid = self.scale_pyramid(right, self.n)

        # Prepare disparities
        disp_left_est = [d[:, 0, :, :].unsqueeze(1) for d in input]
        disp_right_est = [d[:, 1, :, :].unsqueeze(1) for d in input]

        self.disp_left_est = disp_left_est
        self.disp_right_est = disp_right_est

        # Generate images by applying the left disparity map to the right image and vice versa (at each pyramid scale)
        left_est = [
            self.generate_image_left(right_pyramid[i], disp_left_est[i])
            for i in range(self.n)
        ]
        right_est = [
            self.generate_image_right(left_pyramid[i], disp_right_est[i])
            for i in range(self.n)
        ]
        self.left_est = left_est
        self.right_est = right_est

        ###################
        # L-R Consistency #
        ###################
        # Shift the right disparity map by the left disparity map and vice versa
        right_left_disp = [
            self.generate_image_left(disp_right_est[i], disp_left_est[i])
            for i in range(self.n)
        ]
        left_right_disp = [
            self.generate_image_right(disp_left_est[i], disp_right_est[i])
            for i in range(self.n)
        ]

        # Compute the difference between the estimated disparity map and the shifted ones
        lr_left_loss = [
            torch.mean(torch.abs(right_left_disp[i] - disp_left_est[i]))
            for i in range(self.n)
        ]
        lr_right_loss = [
            torch.mean(torch.abs(left_right_disp[i] - disp_right_est[i]))
            for i in range(self.n)
        ]
        lr_loss = sum(lr_left_loss + lr_right_loss)

        ##############
        # Smoothness #
        ##############
        disp_left_smoothness = self.disp_smoothness(disp_left_est, left_pyramid)
        disp_right_smoothness = self.disp_smoothness(disp_right_est, right_pyramid)

        # Weight the smoothness terms at different scales by 1 / 2**i
        disp_left_loss = [
            torch.mean(torch.abs(disp_left_smoothness[i])) / 2 ** i
            for i in range(self.n)
        ]
        disp_right_loss = [
            torch.mean(torch.abs(disp_right_smoothness[i])) / 2 ** i
            for i in range(self.n)
        ]
        disp_gradient_loss = sum(disp_left_loss + disp_right_loss)

        ########################
        # Image Reconstruction #
        ########################
        # consists of L1 norm and SSIM between estimated and input images
        # L1
        l1_left = [
            torch.mean(torch.abs(left_est[i] - left_pyramid[i])) for i in range(self.n)
        ]
        l1_right = [
            torch.mean(torch.abs(right_est[i] - right_pyramid[i]))
            for i in range(self.n)
        ]

        # SSIM
        ssim_left = [
            torch.mean(self.SSIM(left_est[i], left_pyramid[i])) for i in range(self.n)
        ]
        ssim_right = [
            torch.mean(self.SSIM(right_est[i], right_pyramid[i])) for i in range(self.n)
        ]

        image_loss_left = [
            self.SSIM_w * ssim_left[i] + (1 - self.SSIM_w) * l1_left[i]
            for i in range(self.n)
        ]
        image_loss_right = [
            self.SSIM_w * ssim_right[i] + (1 - self.SSIM_w) * l1_right[i]
            for i in range(self.n)
        ]
        image_loss = sum(image_loss_left + image_loss_right)

        ##############
        # Total loss #
        ##############
        loss = (
            image_loss + self.disp_gradient_w * disp_gradient_loss + self.lr_w * lr_loss
        )
        self.image_loss = image_loss
        self.disp_gradient_loss = disp_gradient_loss
        self.lr_loss = lr_loss
        return loss, image_loss, disp_gradient_loss, lr_loss
