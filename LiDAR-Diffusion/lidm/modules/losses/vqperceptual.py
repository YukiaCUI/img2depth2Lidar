import torch
from torch import nn
import os
import numpy as np

from . import weights_init, l1, l2, hinge_d_loss, vanilla_d_loss, measure_perplexity, square_dist_loss
from .geometric import GeoConverter
from .discriminator import NLayerDiscriminator, LiDARNLayerDiscriminator, LiDARNLayerDiscriminatorV2
from .perceptual import PerceptualLoss
from .curvature import Curvature

VERSION2DISC = {'v0': NLayerDiscriminator, 'v1': LiDARNLayerDiscriminator, 'v2': LiDARNLayerDiscriminatorV2}


class VQGeoLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_out_channels=1, disc_factor=1.0, disc_weight=1.0,
                 mask_factor=0.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge", n_classes=None, pixel_loss="l1", disc_version='v1',
                 geo_factor=1.0, curve_length=4, perceptual_factor=1.0, perceptual_type='rangenet_final',
                 dataset_config=dict()):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        assert pixel_loss in ["l1", "l2"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.mask_factor = mask_factor
        self.geo_factor = geo_factor

        # TODO:
        self.dataset_config = dataset_config

        # scale of reconstruction loss
        self.rec_scale = 1
        if mask_factor > 0:
            self.rec_scale += 1.
        if geo_factor > 0:
            self.rec_scale += 1.
        if perceptual_factor > 0:
            self.rec_scale += 1.

        if pixel_loss == "l1":
            self.pixel_loss = l1
        else:
            self.pixel_loss = l2

        self.perceptual_factor = perceptual_factor
        if perceptual_factor > 0.:
            print(f"{self.__class__.__name__}: Running with LPIPS.")
            self.perceptual_loss = PerceptualLoss(perceptual_type, dataset_config.depth_scale,
                                                  dataset_config.log_scale).eval()

        disc_cls = VERSION2DISC[disc_version]
        self.discriminator = disc_cls(input_nc=disc_in_channels,
                                      output_nc=disc_out_channels,
                                      n_layers=disc_num_layers,
                                      use_actnorm=use_actnorm,
                                      ndf=disc_ndf).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQGeoLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.n_classes = n_classes

        self.geometry_converter = GeoConverter(curve_length, False, dataset_config)  # force converting xyz output
        self.geo_loss = square_dist_loss

        # TODO: add curvature loss
        self.curvature_maker= None

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight
    
    def cur_loss(self, input_curvature, rec_curvature):
        # 确保输入张量的形状一致
        assert input_curvature.shape == rec_curvature.shape, "Input and reconstruction curvature must have the same shape."

        # 创建一个掩码，标记需要排除的点
        B, C, H, W = input_curvature.shape
        mask = torch.ones((B, C, H, W), dtype=torch.bool, device=input_curvature.device)

        # 排除前5和后5个元素
        mask[:, :, :, :5] = 0
        mask[:, :, :, -5:] = 0

        # 检查是否有无穷大值
        inf_mask = (input_curvature == float('inf')) | (rec_curvature == float('inf'))
        mask = mask & ~inf_mask  # 组合掩码，排除无穷大值的点

        # 计算有效的曲率差异
        valid_input_curvature = input_curvature[mask]
        valid_rec_curvature = rec_curvature[mask]

        # 计算损失，使用 L1 损失
        if valid_input_curvature.numel() > 0:  # 确保有效点存在
            loss = l1(valid_input_curvature, valid_rec_curvature)
            
        else:
            loss = torch.tensor(0.0, device=input_curvature.device)  # 如果没有有效点，损失为0, 没有 grad_fn 的张量, 不参与梯度传播

        return loss

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train", predicted_indices=None, masks=None):
        input_coord = self.geometry_converter(inputs)
        rec_coord = self.geometry_converter(reconstructions[:, 0:1].contiguous())

        ############# Reconstruction #############
        # pixel reconstruction loss
        if self.mask_factor > 0 and masks is not None:
            pixel_rec_loss = self.pixel_loss(inputs.contiguous(), reconstructions[:, 0:1].contiguous())
            mask_rec_loss = self.pixel_loss(masks.contiguous(), reconstructions[:, 1:2].contiguous()) * self.mask_factor
        else:
            pixel_rec_loss = self.pixel_loss(inputs.contiguous(), reconstructions.contiguous())
            mask_rec_loss = torch.tensor(0.0)

        # geometry reconstruction loss (bev)
        if self.geo_factor > 0:
            geo_rec_loss = self.geo_loss(input_coord[:, :2], rec_coord[:, :2]) * self.geo_factor
        else:
            geo_rec_loss = torch.tensor(0.0)

        # perceptual loss
        if self.perceptual_factor > 0:
            perceptual_loss = self.perceptual_loss((inputs.contiguous(), input_coord),
                                                   (reconstructions[:, 0:1].contiguous(), rec_coord)) * self.perceptual_factor
        else:
            perceptual_loss = torch.tensor(0.0)

        # Inputs shape: torch.Size([32, 1, 64, 1024])
        # Reconstructions shape: torch.Size([32, 1, 64, 1024])
        # TODO: add curve loss
        self.curvature_maker= Curvature(self.dataset_config) # 实例化

        input_curvature = self.curvature_maker(inputs)
        rec_curvature = self.curvature_maker(reconstructions[:, 0:1].contiguous())

        # TODO:
        # input_curvature_np = input_curvature.cpu().detach().numpy()
        # # 保存到 txt 文件
        # with open('curvature_values.txt', 'w') as f:
        #     f.write("Input Curvature:\n")
        #     for tensor_slice in input_curvature_np:
        #         f.write(f"{tensor_slice}\n")

        # print("Curvature values saved to curvature_values.txt")

 
        # TODO：
        threshold = 0.01
        valid_mask = torch.isfinite(input_curvature) & (input_curvature < threshold)    

        # 创建fake_mask，进行实验
        B, C, H, W = input_curvature.shape
        fake_mask = torch.hstack([torch.ones((H, int(W/2))), torch.zeros((H, int(W/2)))])
        fake_mask = fake_mask.repeat(B, C, 1, 1)
        filtered_input_curvature = input_curvature[fake_mask == 1] 
        filtered_rec_curvature = rec_curvature[fake_mask == 1]


        # filtered_input_curvature = input_curvature[valid_mask == 1]
        # filtered_rec_curvature = rec_curvature[valid_mask == 1]

        # 创建非零掩码，筛掉两个张量中都为 0 的值
        non_zero_mask = (filtered_input_curvature != 0) & (filtered_rec_curvature != 0)

        # 筛选非零值
        final_input_curvature = filtered_input_curvature[non_zero_mask]
        final_rec_curvature = filtered_rec_curvature[non_zero_mask]
        
        curve_loss = self.pixel_loss(final_input_curvature, final_rec_curvature)

        # curve_loss = self.cur_loss(filtered_input_curvature, filtered_rec_curvature)
        # valid_curve_loss = curve_loss[curve_loss != 0]

        if curve_loss.numel() > 0:
            curve_loss_mean = curve_loss.mean()
        else:
            curve_loss_mean = torch.tensor(0.0)

        # # 将 inputs 和 valid_mask 保存为 txt 文件
        # for i in range(inputs.shape[0]):  # 遍历每个样本
        #     input_data = inputs[i, 0].cpu().detach().numpy().reshape(64, 1024)  # 获取当前样本数据并转为 2D numpy 数组
        #     mask_data = valid_mask[i].cpu().detach().numpy().reshape(64, 1024)  # 获取当前样本的有效掩码并转为 2D numpy 数组
            
        #     # 保存输入数据
        #     input_file_path = f'input_{i}.txt'
        #     np.savetxt(input_file_path, input_data, fmt='%.6f')  # 保存输入数据为 txt 文件，保留小数点后六位
        #     print(f'Saved input data to: {input_file_path}')  # 输出保存位置

        #     # 保存有效掩码
        #     mask_file_path = f'mask_{i}.txt'
        #     np.savetxt(mask_file_path, mask_data, fmt='%d')  # 保存有效掩码为 txt 文件，数据格式为整数
        #     print(f'Saved mask data to: {mask_file_path}')  # 输出保存位置

        # overall reconstruction loss
        rec_loss = (pixel_rec_loss + mask_rec_loss + geo_rec_loss + perceptual_loss) / self.rec_scale
        nll_loss = rec_loss
        nll_loss = torch.mean(nll_loss)

        ############# GAN #############
        disc_factor = 0. if global_step > self.discriminator_iter_start else self.disc_factor
        # update generator (input: img, mask, coord, [cond])
        if optimizer_idx == 0:
            disc_recons = reconstructions.contiguous()
            if self.geo_factor > 0:
                disc_recons = torch.cat((disc_recons, rec_coord[:, :2].contiguous()), dim=1)
            if cond is not None and self.disc_conditional:
                disc_recons = torch.cat((disc_recons, cond), dim=1)
            logits_fake = self.discriminator(disc_recons)

            # adversarial loss
            g_loss = -torch.mean(logits_fake)

            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0) 

            loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean()

            # TODO: add curvature loss
            # if loss < 0.05:
            #     loss += curve_loss_mean
            loss += curve_loss_mean


            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/pix_rec_loss".format(split): pixel_rec_loss.detach().mean(),
                   "{}/geo_rec_loss".format(split): geo_rec_loss.detach().mean(),
                   "{}/mask_rec_loss".format(split): mask_rec_loss.detach().mean(),
                   "{}/perceptual_loss".format(split): perceptual_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   "{}/curve_loss".format(split): curve_loss.detach().mean()
                   }

            if predicted_indices is not None:
                assert self.n_classes is not None
                with torch.no_grad():
                    perplexity, cluster_usage = measure_perplexity(predicted_indices, self.n_classes)
                log[f"{split}/perplexity"] = perplexity
                log[f"{split}/cluster_usage"] = cluster_usage
            return loss, log

        # update discriminator (input: img, mask, coord, [cond])
        if optimizer_idx == 1:
            disc_inputs, disc_recons = inputs.contiguous().detach(), reconstructions.contiguous().detach()
            if self.mask_factor > 0 and masks is not None:
                disc_inputs = torch.cat((disc_inputs, masks.contiguous().detach()), dim=1)
            if self.geo_factor > 0:
                disc_inputs = torch.cat((disc_inputs, input_coord[:, :2].contiguous()), dim=1)
                disc_recons = torch.cat((disc_recons, rec_coord[:, :2].contiguous()), dim=1)
            if cond is not None:
                disc_inputs = torch.cat((disc_inputs, cond), dim=1)
                disc_recons = torch.cat((disc_recons, cond), dim=1)
            logits_real = self.discriminator(disc_inputs)
            logits_fake = self.discriminator(disc_recons)

            # gan loss
            d_loss = self.disc_loss(logits_real, logits_fake) * disc_factor

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()}

            return d_loss, log
