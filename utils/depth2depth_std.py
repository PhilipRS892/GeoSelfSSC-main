

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


def calculate_depth_std(depth_map, beta=0.0, window_size=3):
    """
    Calculate the confidence level of a depth map based on gradient magnitudes.

    Parameters:
    - depth_map: 2D tensor representing the depth map.
    - beta: scaling factor.
    - window_size: size of the sliding window (e.g., 3 for 3x3, 5 for 5x5).

    Returns:
    - depth_std: 2D tensor of the calculated standard deviations.
    """
    # Ensure depth_map is a float tensor
    depth_map = depth_map.float()

    # Step 1: Compute the gradients in x and y directions
    gradient_x = F.conv2d(depth_map.unsqueeze(0).unsqueeze(0),
                          weight=torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
                                              dtype=torch.float32).unsqueeze(1),
                          stride=1, padding=1)

    gradient_y = F.conv2d(depth_map.unsqueeze(0).unsqueeze(0),
                          weight=torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],
                                              dtype=torch.float32).unsqueeze(1),
                          stride=1, padding=1)

    # Step 2: Compute the gradient magnitudes
    gradient_magnitude = torch.sqrt(gradient_x ** 2 + gradient_y ** 2)

    # Step 3: Compute the mean gradient magnitude in a sliding window
    mean_grad = F.avg_pool2d(gradient_magnitude, kernel_size=window_size, stride=1, padding=window_size // 2)

    # Step 4: Compute the squared difference from the mean in the sliding window
    squared_diff = (gradient_magnitude - mean_grad) ** 2
    local_variance = F.avg_pool2d(squared_diff, kernel_size=window_size, stride=1, padding=window_size // 2)

    # Step 5: Compute the local standard deviation (std)
    local_std = torch.sqrt(local_variance)

    # Step 6: Scale by |beta - 1| to get the depth confidence map
    depth_std = torch.abs(torch.tensor(beta - 1)) * local_std.squeeze()

    return depth_std



DRY_RUN = False


def load_depth_image(depth_path):
    """加载深度图"""
    # depth_raw = Image.open(depth_path)  # 使用PIL加载PNG图像
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) / 500.0

    # depth_array = np.array(depth_raw, dtype=np.uint16)

    depth_array_float = depth_raw.astype(np.float32)

    depth_image = torch.tensor(depth_array_float)  # 转换为 PyTorch 张量
    return depth_image


def save_std(path: str, filename: str, std):

    # 转换为可视化的格式
    # height, width = std.shape
    # std = np.asarray(std)
    # std = std.astype(np.uint8)

    std_np = std.detach().numpy()
    std = (std_np * 500).astype('uint16')
    # 转换为 PIL 图像并保存
    pil_image = pil.fromarray(std)  # 转换为 PIL 图像
    pil_image.save(os.path.join(path, filename))  # 保存图像



def convert_sequence(input_dir, output_dir, seq: str):
    # image_00
    convert_folder(
        folder_path=os.path.join(input_dir, seq, 'image_00', 'data_192x640'),
        output_path=os.path.join(output_dir, seq, 'image_00', 'data_192x640')
    )

    # image_01
    convert_folder(
        folder_path=os.path.join(input_dir, seq, 'image_01', 'data_192x640'),
        output_path=os.path.join(output_dir, seq, 'image_01', 'data_192x640')
    )

    # image_02
    convert_folder(
        folder_path=os.path.join(input_dir, seq, 'image_02', 'data_192x640_0x-15'),
        output_path=os.path.join(output_dir, seq, 'image_02', 'data_192x640_0x-15')
    )

    # image_03
    convert_folder(
        folder_path=os.path.join(input_dir, seq, 'image_03', 'data_192x640_0x-15'),
        output_path=os.path.join(output_dir, seq, 'image_03', 'data_192x640_0x-15')
    )


def convert_folder(folder_path, output_path):
    logging.info(f"Converting folder at {folder_path}")

    if not os.path.exists(output_path):
        logging.info(f"Output directory {output_path} does not exist and has to be created!")
        os.makedirs(output_path)

    for filename in tqdm(sorted(os.listdir(folder_path))):
        # _pred = predict_image(os.path.join(folder_path, filename), model=model, device=device)
        depth_map = load_depth_image(os.path.join(folder_path, filename))
        # depth_raw = depth_raw.unsqueeze(0) # (B H W)
        std = calculate_depth_std(depth_map)
        save_std(output_path, filename, std)

    logging.info("Conversion finished")



def main():

    # 使用示例
    input_dir = "/data/GPT/s4c-main/data_depth"  # KITTI-360数据集所在路径
    output_dir = "/data/GPT/s4c-main/data_std"  # 保存预测深度图的目标路径


    with torch.no_grad():
        for seq in os.listdir(input_dir):
            convert_sequence(input_dir, output_dir, seq)


if __name__ == '__main__':

    main()
