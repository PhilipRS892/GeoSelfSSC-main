
import cv2

import logging
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os

import PIL.Image as pil

import numpy as np
from tqdm import tqdm

import torch

import torch.nn.functional as F


DRY_RUN = False


def compute_normals(depth_map):
    # 计算法线
    dx = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=5)  # x方向的梯度
    dy = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=5)  # y方向的梯度

    # 法线向量
    normals = np.dstack((-dx, -dy, np.ones_like(depth_map)))
    normals /= np.linalg.norm(normals, axis=2, keepdims=True)  # 归一化法线向量

    return normals


def get_points_coordinate(depth, instrinsic_inv):
    B, height, width, C = depth.size()
    # 创建坐标网格
    y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=depth.device),
                           torch.arange(0, width, dtype=torch.float32, device=depth.device)])
    y, x = y.contiguous(), x.contiguous()
    # 将2维扁平化为1维
    y, x = y.view(height * width), x.view(height * width)
    # 创建3D坐标
    xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
    xyz = torch.unsqueeze(xyz, 0).repeat(B, 1, 1)  # [B, 3, H*W] broadcast
    # 将2D像素坐标转换为3D世界坐标
    xyz = torch.matmul(instrinsic_inv, xyz) # [B, 3, H*W]
    # 调整3D点的实际坐标
    depth_xyz = xyz * depth.view(B, 1, -1)  # [B, 3, Ndepth, H*W]

    return depth_xyz.view(B, 3, height, width)

def depth2normal(depth_torch, intrinsic_torch):

    # load depth & intrinsic

    depth_torch = depth_torch.unsqueeze(-1) # (B, h, w, 1)
    B, H, W, _ = depth_torch.shape

    intrinsic_inv_torch = torch.inverse(intrinsic_torch) # (B, 3, 3)

    ## step.2 compute matrix A
    # compute 3D points xyz
    points = get_points_coordinate(depth_torch, intrinsic_inv_torch)
    # 获取邻域信息 通过展开炒作获取每个点周围的5x5的点
    point_matrix = F.unfold(points, kernel_size=5, stride=1, padding=4, dilation=2)

    # An = b
    matrix_a = point_matrix.view(B, 3, 25, H, W)  # (B, 3, 25, HxW)
    matrix_a = matrix_a.permute(0, 3, 4, 2, 1) # (B, HxW, 25, 3)
    matrix_a_trans = matrix_a.transpose(3, 4)
    matrix_b = torch.ones([B, H, W, 25, 1], device=depth_torch.device)

    # dot(A.T, A)  计算矩阵的行列式 并判断它是否可逆
    point_multi = torch.matmul(matrix_a_trans, matrix_a)
    matrix_deter = torch.det(point_multi)

    # make inversible
    inverse_condition = torch.ge(matrix_deter, 1e-5)
    inverse_condition = inverse_condition.unsqueeze(-1).unsqueeze(-1)
    inverse_condition_all = inverse_condition.repeat(1, 1, 1, 3, 3)
    # diag matrix to update uninverse  使用对角矩阵来处理不可逆的情况
    diag_constant = torch.ones([3], dtype=torch.float32, device=depth_torch.device)
    diag_element = torch.diag(diag_constant)
    diag_element = diag_element.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    diag_matrix = diag_element.repeat(1, H, W, 1, 1)
    # inversible matrix
    inversible_matrix = torch.where(inverse_condition_all, point_multi, diag_matrix)
    inv_matrix = torch.inverse(inversible_matrix)

    ## step.3 compute normal vector use least square
    # n = (A.T A)^-1 A.T b // || (A.T A)^-1 A.T b ||2
    generated_norm = torch.matmul(torch.matmul(inv_matrix, matrix_a_trans), matrix_b)
    norm_normalize = F.normalize(generated_norm.squeeze(-1), p=2, dim=3).permute(0, 3, 1, 2) #b 3 h w

    return norm_normalize


def load_depth_image(depth_path):
    """加载深度图"""
    # depth_raw = Image.open(depth_path)  # 使用PIL加载PNG图像
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) / 500.0

    # depth_array = np.array(depth_raw, dtype=np.uint16)

    # depth_array_float = depth_raw.astype(np.float32)
    #
    # depth_image = torch.tensor(depth_array_float)  # 转换为 PyTorch 张量
    return depth_raw


def save_normal(path: str, filename: str, normals):
    """将归一化的法线图保存为图像文件"""

    # 转换为可视化的格式
    height, width, _ = normals.shape
    normals_visual = ((normals + 1) / 2 * 255).astype(np.uint8)  # 将法线值从 [-1, 1] 转换到 [0, 255]

    # 转换为 PIL 图像并保存
    pil_image = pil.fromarray(normals_visual)  # 转换为 PIL 图像
    pil_image.save(os.path.join(path, filename))  # 保存图像

    # 可视化法线
    # plt.imshow(normals_visual)
    # plt.axis('off')
    # plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    # plt.show()

def visualize_normal_map(norm_normalize):
    """可视化法线图"""
    # norm_normalize 的形状为 (B, 3, H, W)
    norm_image = norm_normalize[0].detach().cpu()  # 选择第一个样本并移到CPU

    # 将法线图转换为可视化格式
    norm_image = norm_image.permute(1, 2, 0)  # (H, W, 3)
    norm_image = (norm_image * 0.5 + 0.5)  # 将值映射到 [0, 1]

    # 使用 Matplotlib 可视化
    import matplotlib.pyplot as plt

    plt.imshow(norm_image.numpy())
    plt.axis('off')  # 不显示坐标轴
    plt.show()

def convert_sequence(input_dir, output_dir, seq: str, intrinsic_torch):
    # image_00
    convert_folder(
        folder_path=os.path.join(input_dir, seq, 'image_00', 'data_192x640'),
        output_path=os.path.join(output_dir, seq, 'image_00', 'data_192x640'),
        intrinsic_torch = intrinsic_torch
    )

    # image_01
    convert_folder(
        folder_path=os.path.join(input_dir, seq, 'image_01', 'data_192x640'),
        output_path=os.path.join(output_dir, seq, 'image_01', 'data_192x640'),
        intrinsic_torch = intrinsic_torch
    )

    # image_02
    convert_folder(
        folder_path=os.path.join(input_dir, seq, 'image_02', 'data_192x640_0x-15'),
        output_path=os.path.join(output_dir, seq, 'image_02', 'data_192x640_0x-15'),
        intrinsic_torch = intrinsic_torch
    )

    # image_03
    convert_folder(
        folder_path=os.path.join(input_dir, seq, 'image_03', 'data_192x640_0x-15'),
        output_path=os.path.join(output_dir, seq, 'image_03', 'data_192x640_0x-15'),
        intrinsic_torch = intrinsic_torch
    )


def convert_folder(folder_path, output_path, intrinsic_torch):
    logging.info(f"Converting folder at {folder_path}")

    if not os.path.exists(output_path):
        logging.info(f"Output directory {output_path} does not exist and has to be created!")
        os.makedirs(output_path)

    for filename in tqdm(sorted(os.listdir(folder_path))):
        # _pred = predict_image(os.path.join(folder_path, filename), model=model, device=device)
        depth_map = load_depth_image(os.path.join(folder_path, filename))
        # depth_raw = depth_raw.unsqueeze(0) # (B H W)
        normals = compute_normals(depth_map)
        save_normal(output_path, filename, normals)

    logging.info("Conversion finished")



def main():

    input_dir = "/data/GPT/s4c-main/data_depth"  # KITTI-360数据集所在路径
    output_dir = "/data/GPT/s4c-main/data_normal2"  # 保存预测深度图的目标路径
    # process_dataset(input_folder, output_folder)
    intrinsic_torch = torch.tensor([[0.78488, 0, -0.03118],
                                    [0, 2.93912, 0.27005],
                                    [0, 0, 1]], dtype=torch.float32)

    with torch.no_grad():
        for seq in os.listdir(input_dir):
            convert_sequence(input_dir, output_dir, seq, intrinsic_torch)


if __name__ == '__main__':

    main()
