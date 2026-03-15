import math

import torch
import torch.nn.functional as F
from torch import profiler

from models.common.model.layers import ssim


def compute_errors_l1ssim(img0, img1, mask=None):
    n, pc, h, w, nv, c = img0.shape
    img1 = img1.expand(img0.shape)
    img0 = img0.permute(0, 1, 4, 5, 2, 3).reshape(-1, c, h, w)
    img1 = img1.permute(0, 1, 4, 5, 2, 3).reshape(-1, c, h, w)
    errors = .85 * torch.mean(ssim(img0, img1, pad_reflection=False, gaussian_average=True, comp_mode=True), dim=1) + .15 * torch.mean(torch.abs(img0 - img1), dim=1)
    errors = errors.view(n, pc, nv, h, w).permute(0, 1, 3, 4, 2).unsqueeze(-1)
    if mask is not None: return errors, mask
    else: return errors


def edge_aware_smoothness(gt_img, depth, mask=None):
    n, pc, h, w = depth.shape
    gt_img = gt_img.permute(0, 1, 4, 5, 2, 3).reshape(-1, 3, h, w)
    depth = 1 / depth.reshape(-1, 1, h, w).clamp(1e-3, 80)
    depth = depth / torch.mean(depth, dim=[2, 3], keepdim=True)

    gt_img = F.interpolate(gt_img, (h, w))

    d_dx = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
    d_dy = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])

    i_dx = torch.mean(torch.abs(gt_img[:, :, :, :-1] - gt_img[:, :, :, 1:]), 1, keepdim=True)
    i_dy = torch.mean(torch.abs(gt_img[:, :, :-1, :] - gt_img[:, :, 1:, :]), 1, keepdim=True)

    d_dx *= torch.exp(-i_dx)
    d_dy *= torch.exp(-i_dy)

    errors = F.pad(d_dx, pad=(0, 1), mode='constant', value=0) + F.pad(d_dy, pad=(0, 0, 0, 1), mode='constant', value=0)
    errors = errors.view(n, pc, h, w)
    return errors

# 定义 compute_scale_and_shift 函数
def compute_scale_and_shift(prediction, target, mask):
    mask = mask.float()
    a_00 = torch.sum(mask * prediction * prediction, dim=(1, 2))
    a_01 = torch.sum(mask * prediction, dim=(1, 2))
    a_11 = torch.sum(mask, dim=(1, 2))

    b_0 = torch.sum(mask * prediction * target, dim=(1, 2))
    b_1 = torch.sum(mask * target, dim=(1, 2))

    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det != 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1

# 定义 gradient_loss 函数
def gradient_loss(prediction, target, mask, reduction=torch.mean):
    M = torch.sum(mask, dim=(1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, dim=(1, 2)) + torch.sum(grad_y, dim=(1, 2))

    return reduction(image_loss / M)

def get_normal_loss(normal_pred, normal_gt):
    # 正则化预测和GT法线
    normal_gt = F.normalize(normal_gt, p=2, dim=-1)
    normal_pred = F.normalize(normal_pred, p=2, dim=-1)

    # 计算 L1 损失
    l1 = torch.abs(normal_pred - normal_gt).sum(dim=-1).mean()

    # 计算余弦损失
    cos = (1.0 - torch.sum(normal_pred * normal_gt, dim=-1)).mean()

    return l1, cos


class ReconstructionLoss:
    def __init__(self, config, use_automasking=False) -> None:
        super().__init__()
        self.criterion_str = config.get("criterion", "l2")
        if self.criterion_str == "l2":
            self.rgb_coarse_crit = torch.nn.MSELoss(reduction="none")
            self.rgb_fine_crit = torch.nn.MSELoss(reduction="none")
        elif self.criterion_str == "l1":
            self.rgb_coarse_crit = torch.nn.L1Loss(reduction="none")
            self.rgb_fine_crit = torch.nn.L1Loss(reduction="none")
        elif self.criterion_str == "l1+ssim":
            self.rgb_coarse_crit = compute_errors_l1ssim
            self.rgb_fine_crit = compute_errors_l1ssim
        self.invalid_policy = config.get("invalid_policy", "strict")  # weight guided
        assert self.invalid_policy in ["strict", "weight_guided", "weight_guided_diverse", None, "none"]
        self.ignore_invalid = self.invalid_policy is not None and self.invalid_policy != "none"
        self.lambda_coarse = config.get("lambda_coarse", 1)
        self.lambda_fine = config.get("lambda_fine", 1)
        self.lambda_segmentation = config.get("lambda_segmentation", 1)
        self.lambda_depth = config.get("lambda_depth", 1)
        self.lambda_normal_l1 = config.get("lambda_normal_l1", 1)
        self.lambda_normal_cos = config.get("lambda_normal_cos", 1)
        self.segmentation_class_weights = config.get("segmentation_class_weights", None)

        if self.segmentation_class_weights is not None:
            self.segmentation_class_weights = torch.tensor(list(config.get("segmentation_class_weights", None).values()))

        self.use_automasking = use_automasking

        self.lambda_entropy = config.get("lambda_entropy", 0)
        self.lambda_density_entropy = config.get("lambda_density_entropy", 0)
        self.lambda_depth_reg = config.get("lambda_depth_reg", 0)
        self.lambda_alpha_reg = config.get("lambda_alpha_reg", 0)
        self.lambda_surfaceness_reg = config.get("lambda_surfaceness_reg", 0)
        self.lambda_edge_aware_smoothness = config.get("lambda_edge_aware_smoothness", 0)
        self.lambda_depth_smoothness = config.get("lambda_depth_smoothness", 0)

        self.median_thresholding = config.get("median_thresholding", False)

        self.alpha_reg_reduction = config.get("alpha_reg_reduction", "ray")
        self.alpha_reg_fraction = config.get("alpha_reg_fraction", 1/8)

        if self.alpha_reg_reduction not in ("ray", "slice"):
            raise ValueError(f"Unknown reduction for alpha regularization: {self.alpha_reg_reduction}")

    @staticmethod
    def get_loss_metric_names():
        return ["loss", "loss_rgb_coarse", "loss_rgb_fine", "loss_ray_entropy", "loss_depth_reg"]

    def __call__(self, data):
        with profiler.record_function("loss_computation"):
            n_scales = len(data["coarse"])

            loss_dict = {}

            loss_coarse_all = 0
            loss_fine_all = 0
            loss_segmentation = 0
            loss = 0

            coarse_0 = data["coarse"][0]  # raw_coarse
            fine_0 = data["fine"][0]  # raw_coarse
            segmentation_0 = data["segmentation"][0]  #
            invalid_coarse = coarse_0["invalid"]
            invalid_fine = fine_0["invalid"]
            invalid_segmentation = segmentation_0["invalid"]
            invalid_depth = segmentation_0["invalid"]
            invalid_normal = segmentation_0["invalid"]

            weights_coarse = coarse_0["weights"]
            weights_fine = fine_0["weights"]
            weights_segmentation = segmentation_0["weights"]

            if self.invalid_policy == "strict":
                # Consider all rays invalid where there is at least one invalidly sampled color
                invalid_coarse = torch.all(torch.any(invalid_coarse > .5, dim=-2), dim=-1).unsqueeze(-1)
                invalid_fine = torch.all(torch.any(invalid_fine > .5, dim=-2), dim=-1).unsqueeze(-1)
                invalid_segmentation = torch.all(torch.any(invalid_segmentation > .5, dim=-2), dim=-1).unsqueeze(-1)

            elif self.invalid_policy == "weight_guided":
                # Integrate invalid indicator function over the weights. It is invalid if > 90% of the mass is invalid. (Arbitrary threshold)
                invalid_coarse = torch.all((invalid_coarse.to(torch.float32) * weights_coarse.unsqueeze(-1)).sum(-2) > .9, dim=-1, keepdim=True)
                invalid_fine = torch.all((invalid_fine.to(torch.float32) * weights_fine.unsqueeze(-1)).sum(-2) > .9, dim=-1, keepdim=True)
                invalid_segmentation = torch.all((invalid_segmentation.to(torch.float32) * weights_segmentation.unsqueeze(-1)).sum(-2) > .9,
                                         dim=-1, keepdim=True)

            elif self.invalid_policy == "weight_guided_diverse":
                # We now also consider, whether there is enough variance in the ray colors to give a meaningful supervision signal.
                rgb_samps_c = coarse_0["rgb_samps"]
                rgb_samps_f = fine_0["rgb_samps"]
                ray_std_c = torch.std(rgb_samps_c, dim=-3).mean(-1)
                ray_std_f = torch.std(rgb_samps_f, dim=-3).mean(-1)

                # Integrate invalid indicator function over the weights. It is invalid if > 90% of the mass is invalid. (Arbitrary threshold)
                invalid_coarse = torch.all(((invalid_coarse.to(torch.float32) * weights_coarse.unsqueeze(-1)).sum(-2) > .9) | (ray_std_c < 0.01), dim=-1, keepdim=True)
                invalid_fine = torch.all(((invalid_fine.to(torch.float32) * weights_fine.unsqueeze(-1)).sum(-2) > .9) | (ray_std_f < 0.01), dim=-1, keepdim=True)

                # for now we just do the weight guided invalids for the segmentation
                invalid_segmentation = torch.all(
                    (invalid_segmentation.to(torch.float32) * weights_segmentation.unsqueeze(-1)).sum(-2) > .9,
                    dim=-1, keepdim=True)
            elif self.invalid_policy == "none":
                invalid_coarse = torch.zeros_like(torch.all(torch.any(invalid_coarse > .5, dim=-2), dim=-1).unsqueeze(-1), dtype=torch.bool)
                invalid_fine = torch.zeros_like(torch.all(torch.any(invalid_fine > .5, dim=-2), dim=-1).unsqueeze(-1), dtype=torch.bool)
                invalid_segmentation = torch.zeros_like(torch.all(torch.any(invalid_segmentation > .5, dim=-2), dim=-1).unsqueeze(-1),
                                                dtype=torch.bool)
            else:
                raise NotImplementedError

            loss_depth_reg = torch.tensor(0.0, device=invalid_fine.device)
            loss_alpha_reg = torch.tensor(0.0, device=invalid_fine.device)
            loss_surfaceness_reg = torch.tensor(0.0, device=invalid_fine.device)
            loss_eas = torch.tensor(0.0, device=invalid_fine.device)
            loss_depth_smoothness = torch.tensor(0.0, device=invalid_fine.device)

            for scale in range(n_scales):
                coarse = data["coarse"][scale]
                fine = data["fine"][scale]
                segmentation = data["segmentation"][scale]

                rgb_coarse = coarse["rgb"]
                rgb_fine = fine["rgb"]
                rgb_gt = data["rgb_gt"]
                segmentation_gt = data["segmentation_gt"].permute(0, 4, 1, 2, 3).squeeze(1) #(batch_size, n_patches, h, w)
                depth_gt = data["depth_gt"]
                normal_gt = data["normal_gt"]
                bs, n_patch, ph, pw, n_classes = segmentation["segs"].shape
                segmentation_gt = segmentation_gt.view(-1, ph, pw)

                # do cross entropy loss
                self.segmentation_class_weights = self.segmentation_class_weights.to(segmentation_gt.device).float()
                cp_loss_fn = torch.nn.NLLLoss(weight=self.segmentation_class_weights)
                # log_segmentation = torch.log(segmentation["segs"] + 1e-5).permute(0, 4, 1, 2, 3) #(batch_size, n_classes, n_patches, h, w)
                patch_to_image = data["patch_to_image"]
                front_indices = patch_to_image <= 4
                side_indices = patch_to_image > 4

                log_segmentation = torch.log(segmentation["segs"] + 1e-5).reshape(-1, ph, pw, n_classes).permute(0, 3, 1, 2)

                # Account for the invalids
                # TODO: Adjust the mean so that we don't have a low loss just because we have a lot of invalids
                invalid_segmentation = invalid_segmentation.squeeze(-1).to(torch.float32).reshape(-1, ph, pw)

                cp_loss = cp_loss_fn(
                    ((1 - invalid_segmentation.contiguous()).unsqueeze(1) * log_segmentation.contiguous()).float(),
                    ((1 - invalid_segmentation.contiguous()) * segmentation_gt.contiguous()).long())

                loss_segmentation += self.lambda_segmentation * cp_loss.item()

                loss += self.lambda_segmentation * cp_loss

                if self.use_automasking:
                    thresh_gt = rgb_gt[..., -1:]
                    rgb_coarse = rgb_coarse[..., :-1]
                    rgb_fine = rgb_fine[..., :-1]
                    rgb_gt = rgb_gt[..., :-1]

                rgb_coarse = rgb_coarse
                rgb_fine = rgb_fine
                rgb_gt = rgb_gt.unsqueeze(-2)

                using_fine = len(fine) > 0

                b, pc, h, w, nv, c = rgb_coarse.shape

                # Take minimum across all reconstructed views
                rgb_loss = self.rgb_coarse_crit(rgb_coarse, rgb_gt)
                rgb_loss = rgb_loss.amin(-2)

                if self.use_automasking:
                    rgb_loss = torch.min(rgb_loss, thresh_gt)

                if self.ignore_invalid:
                    rgb_loss = rgb_loss * (1 - invalid_coarse.to(torch.float32))

                if self.median_thresholding:
                    threshold = torch.median(rgb_loss.view(b, -1), dim=-1)[0].view(-1, 1, 1, 1, 1)
                    rgb_loss = rgb_loss[rgb_loss <= threshold]

                rgb_loss = rgb_loss.mean()

                loss_coarse_all += rgb_loss.item() * self.lambda_coarse
                if using_fine:
                    fine_loss = self.rgb_fine_crit(rgb_fine, rgb_gt)
                    fine_loss = fine_loss.amin(-2)

                    if self.use_automasking:
                        fine_loss = torch.min(fine_loss, thresh_gt)

                    if self.ignore_invalid:
                        fine_loss = fine_loss * (1 - invalid_fine.to(torch.float32))

                    if self.median_thresholding:
                        threshold = torch.median(fine_loss.view(b, -1), dim=-1)[0].view(-1, 1, 1, 1, 1)
                        fine_loss = fine_loss[fine_loss <= threshold]

                    fine_loss = fine_loss.mean()
                    rgb_loss = rgb_loss * self.lambda_coarse + fine_loss * self.lambda_fine
                    loss_fine_all += fine_loss.item() * self.lambda_fine
                else:
                    loss_dict["loss_rgb_fine"] = 0

                loss += rgb_loss

                # calculate the depth
                depth_pre = segmentation["depth"].reshape(-1, ph, pw)
                depth_gt = depth_gt.reshape(-1, ph, pw)
                mask = (depth_pre > 0.01).float()
                depth_scale, depth_shift = compute_scale_and_shift((1 - invalid_segmentation.contiguous()) * depth_pre,
                                                       (1 - invalid_segmentation.contiguous()) * depth_gt, mask)
                depth_pre_corr = depth_scale[:, None, None] * depth_pre + depth_shift[:, None, None]

                depth_mse_loss = F.mse_loss((depth_pre_corr * (1 - invalid_segmentation.contiguous())).float() , (depth_gt * (1 - invalid_segmentation.contiguous())).float(), reduction="mean")
                depth_grad_loss = gradient_loss((depth_pre_corr * (1 - invalid_segmentation.contiguous())).float(), (depth_gt * (1 - invalid_segmentation.contiguous())).float(), mask)

                loss += self.lambda_depth * (depth_mse_loss + depth_grad_loss)

                # calculate the normal
                normal_pre = segmentation["nors"].reshape(-1, ph, pw, 3)
                normal_gt = normal_gt.reshape(-1, ph, pw, 3)

                nor_l1_loss, nor_cos_loss = get_normal_loss(normal_pre * (1 - invalid_segmentation.contiguous()).unsqueeze(-1), normal_gt * (1 - invalid_segmentation.contiguous()).unsqueeze(-1))

                loss += (self.lambda_normal_l1 * nor_l1_loss + self.lambda_normal_cos * nor_cos_loss)



                if self.lambda_depth_reg > 0:
                    depths = coarse["depth"]
                    diffs_x = depths[:, :, 1:, :] - depths[:, :, :-1, :]
                    diffs_y = depths[:, :, :, 1:] - depths[:, :, :, :-1]
                    loss_depth_reg_s = (diffs_x ** 2).mean() + (diffs_y ** 2).mean()
                    loss_depth_reg += loss_depth_reg_s # * self.lambda_depth_reg
                    loss += loss_depth_reg_s * self.lambda_depth_reg

                if self.lambda_alpha_reg > 0:
                    alphas = coarse["alphas"]
                    n_smps = alphas.shape[-1]

                    # alphas = alphas[..., :-1].sum(-1)
                    # loss_alpha_reg_s = (alphas - (n_smps * self.alpha_reg_fraction)).clamp_min(0)
                    # if self.ignore_invalid:
                    #     loss_alpha_reg_s = loss_alpha_reg_s * (1 - invalid_coarse.squeeze(-1).to(torch.float32))

                    alpha_sum = alphas[..., :-1].sum(-1)
                    min_cap = torch.ones_like(alpha_sum) * (n_smps * self.alpha_reg_fraction)

                    if self.ignore_invalid:
                        alpha_sum = alpha_sum * (1 - invalid_coarse.squeeze(-1).to(torch.float32))
                        min_cap = min_cap * (1 - invalid_coarse.squeeze(-1).to(torch.float32))

                    if self.alpha_reg_reduction == "ray":
                        loss_alpha_reg_s = (alpha_sum - min_cap).clamp_min(0)
                    elif self.alpha_reg_reduction == "slice":
                        loss_alpha_reg_s = (alpha_sum.sum(dim=-1) - min_cap.sum(dim=-1)).clamp_min(0) / alpha_sum.shape[-1]

                    # alphas = alphas[..., :-n_smps//16]
                    # alpha_deltas = alphas[..., 1:] - alphas[..., :-1]
                    # The sum of deltas should be zero. This means that the number of peaks (ie objects) is not limited, but there needs to be free space afterwards again.
                    # We don't consider the last 1/16 samples. They are likely background.
                    # loss_alpha_reg_s = alpha_deltas.sum(-1).clamp_min(0)

                    loss_alpha_reg_s = loss_alpha_reg_s.mean()

                    loss_alpha_reg += loss_alpha_reg_s
                    loss += loss_alpha_reg_s * self.lambda_alpha_reg

                if self.lambda_surfaceness_reg > 0:
                    alphas = coarse["alphas"]
                    n_smps = alphas.shape[-1]

                    p = -torch.log(torch.exp(-alphas.abs()) + torch.exp(-(1 - alphas).abs()))
                    p = p.mean(-1)

                    if self.ignore_invalid:
                        p = p * (1 - invalid_coarse.squeeze(-1).to(torch.float32))

                    loss_surfaceness_reg_s = p.mean()

                    loss_surfaceness_reg += loss_surfaceness_reg_s
                    loss += loss_surfaceness_reg_s * self.lambda_surfaceness_reg

                if self.lambda_edge_aware_smoothness > 0:
                    gt_img = rgb_gt
                    depths = coarse["depth"]
                    loss_eas_s = edge_aware_smoothness(gt_img, depths)

                    if self.ignore_invalid:
                        invalid_scale = torch.ceil(F.interpolate(invalid_coarse.squeeze(-1).to(torch.float32), size=(depths.shape[-2:])))
                        loss_eas_s = loss_eas_s * (1 - invalid_scale)

                    loss_eas_s = loss_eas_s.mean()

                    loss_eas += loss_eas_s
                    loss += loss_eas_s * self.lambda_edge_aware_smoothness / (2 ** scale)

                if self.lambda_depth_smoothness > 0:
                    depths = coarse["depth"]
                    loss_depth_smoothness_s = ((depths[..., :-1, :] - depths[..., 1:, :]) ** 2).mean() + ((depths[..., :, :-1] - depths[..., :, 1:]) ** 2).mean()

                    loss_depth_smoothness += loss_depth_smoothness_s
                    loss += loss_depth_smoothness_s * self.lambda_depth_smoothness


            loss = loss / n_scales

            loss_ray_entropy = torch.tensor(0.0, device=loss.device)
            if self.lambda_entropy > 0:
                alphas = coarse_0["alphas"]
                alphas = alphas + 1e-5

                ray_density = alphas / alphas.sum(dim=-1, keepdim=True)
                ray_entropy = -(ray_density * torch.log(ray_density)).sum(-1) / (math.log2(alphas.shape[-1]))
                ray_entropy = ray_entropy * (1 - invalid_coarse.squeeze(-1).to(torch.float32))
                loss_ray_entropy = ray_entropy.mean()

            loss = loss + loss_ray_entropy * self.lambda_entropy

            # add density entropy loss
            loss_density_entropy = torch.tensor(0.0, device=loss.device)

            if self.lambda_density_entropy > 0:
                alphas = coarse_0["alphas"]
                alphas = alphas + 1e-5
                density_entropy = (1 - alphas)*alphas
                loss_density_entropy = torch.mean(density_entropy) * self.lambda_density_entropy

            loss = loss + loss_density_entropy

        loss_dict["loss_rgb_coarse"] = loss_coarse_all
        loss_dict["loss_rgb_fine"] = loss_fine_all
        loss_dict["loss_segmentation"] = loss_segmentation
        loss_dict["loss_ray_entropy"] = loss_ray_entropy.item()
        loss_dict["loss_density_entropy"] = loss_density_entropy.item()
        loss_dict["loss_depth_reg"] = loss_depth_reg.item()
        loss_dict["loss_alpha_reg"] = loss_alpha_reg.item()
        loss_dict["loss_eas"] = loss_eas.item()
        loss_dict["loss_depth_smoothness"] = loss_depth_smoothness.item()
        loss_dict["loss_invalid_ratio"] = invalid_coarse.float().mean().item()
        loss_dict["loss"] = loss.item()

        return loss, loss_dict
