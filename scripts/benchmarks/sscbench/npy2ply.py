import os
import numpy as np
import yaml
from scripts.voxel.gen_voxelgrid_npy import save_as_voxel_ply
from tqdm import tqdm
import torch

# 定义文件夹路径
folder_path = "/data/GPT/实验结果/raw_result/sscnet"  # 输入文件夹路径
SAVE_PATH = "/data/GPT/实验结果/map_result/sscnet"  # 保存文件夹路径
PLY_SIZES = [51.2]

def convert_voxels(arr, map_dict):
    f = np.vectorize(map_dict.__getitem__)
    return f(arr)

# 定义处理逻辑
def process_npy(file_path,file_name,label_maps):
    """
    加载并处理 .npy 文件
    :param file_path: .npy 文件路径
    :return: 处理结果
    """
    # try:
    # 加载 .npy 文件
    data = np.load(file_path)

    # 将类别映射到sscbench
    segs = data.copy()
    # segs = convert_voxels(segs, label_maps["cityscapes_to_label"])

    is_occupied_seg = torch.tensor(segs>0)

    file_name_ply = file_name.replace(".npy", ".ply")

    # 保存成ply文件
    for size in PLY_SIZES:
        num_voxels = int(size // 0.2)
        save_as_voxel_ply(os.path.join(SAVE_PATH, file_name_ply),
                          is_occupied_seg[: num_voxels, (128 - num_voxels // 2): (128 + num_voxels // 2), :],
                          classes=torch.tensor(
                              segs[: num_voxels, (128 - num_voxels // 2): (128 + num_voxels // 2), :]))

    # except Exception as e:
    #     print(f"处理文件 {file_path} 时出错: {e}")
    #     return None


# 遍历文件夹并调用处理 .npy 文件的程序
def process_folder(folder_path,label_maps):
    """
    遍历文件夹并处理所有 .npy 文件
    :param folder_path: 文件夹路径
    """
    if not os.path.exists(folder_path):
        print(f"路径 {folder_path} 不存在！")
        return
    file_list = sorted(os.listdir(folder_path))

    for file_name in tqdm(file_list, desc="处理文件",unit="个文件"):
        # 获取完整路径
        file_path = os.path.join(folder_path, file_name)
        # 检查是否为 .npy 文件
        if file_name.endswith(".npy"):
            process_npy(file_path,file_name,label_maps)



with open("label_maps.yaml", "r") as f:
    label_maps = yaml.safe_load(f)
# 调用函数
process_folder(folder_path,label_maps)
