# python data_process/compute_normalization.py

import numpy as np
import argparse
import os

def compute_normalization(data_path, save_path=None):
    print(f"Loading dataset from {data_path}")
    data = np.load(data_path)
    states = data["states"]
    actions = data["actions"]

    print(f"states shape: {states.shape}, actions shape: {actions.shape}")

    obs_min = states.min(axis=0)
    obs_max = states.max(axis=0)
    action_min = actions.min(axis=0)
    action_max = actions.max(axis=0)

    if save_path is None:
        save_dir = os.path.dirname(data_path)
        save_path = os.path.join(save_dir, "normalization.npz")

    np.savez_compressed(
        save_path,
        obs_min=obs_min,
        obs_max=obs_max,
        action_min=action_min,
        action_max=action_max,
    )

    print(f"Saved normalization file to: {save_path}")
    print(f"obs_min: {obs_min.shape}, obs_max: {obs_max.shape}, action_min: {action_min.shape}, action_max: {action_max.shape}")
    return save_path


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--data_path", type=str, required=True, help="输入 npz 数据路径（如 train.npz）")
    # parser.add_argument("--save_path", type=str, default=None, help="输出 normalization 文件路径（可选）")
    # args = parser.parse_args()

    # compute_normalization(args.data_path, args.save_path)

    data_path = "/home/ani/ReinFlow/data/book/train.npz"

    compute_normalization(data_path)
