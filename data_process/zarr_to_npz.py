# python data_process/zarr_to_npz.py

import os
import zarr
import numpy as np
import cv2
import argparse


def load_zarr(zarr_path):
    print(f"Loading zarr dataset from {zarr_path}")
    root = zarr.open(zarr_path, mode='r')
    data = root['data']
    meta = root['meta']

    wrist_color = data['wrist_color']
    global_color = data['global_color']
    eef_pose = data['eef_pose']
    episode_ends = np.array(meta['episode_ends'])

    return wrist_color, global_color, eef_pose, episode_ends


def process_episode(wrist_color, global_color, eef_pose, start, end, resize=(96, 96)):
    n = end - start

    wrist_imgs = np.array([cv2.resize(img, resize) for img in wrist_color[start:end]])
    global_imgs = np.array([cv2.resize(img, resize) for img in global_color[start:end]])

    wrist_imgs = np.transpose(wrist_imgs, (0, 3, 1, 2))
    global_imgs = np.transpose(global_imgs, (0, 3, 1, 2))
    imgs = np.concatenate([wrist_imgs, global_imgs], axis=1)

    # eef_pose
    eef = np.array(eef_pose[start:end])
    # states = eef[t], actions = eef[t+1]
    states = eef[:-1]
    actions = eef[1:]
    imgs = imgs[:-1]

    return states, actions, imgs


def convert_zarr_to_npz(zarr_path, save_dir):
    wrist_color, global_color, eef_pose, episode_ends = load_zarr(zarr_path)

    os.makedirs(save_dir, exist_ok=True)
    # save_path = os.path.join(save_dir, os.path.basename(zarr_path).replace('.zarr', '.npz'))
    save_path = os.path.join(save_dir, "train.npz")

    all_states, all_actions, all_images, traj_lengths = [], [], [], []
    start = 0
    for i, end in enumerate(episode_ends):
        print(f"Processing episode {i}: frames {start} → {end - 1}")
        states, actions, images = process_episode(wrist_color, global_color, eef_pose, start, end)

        all_states.append(states)
        all_actions.append(actions)
        all_images.append(images)
        traj_lengths.append(len(states))

        start = end

    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    images = np.concatenate(all_images, axis=0)
    traj_lengths = np.array(traj_lengths, dtype=np.int64)

    np.savez_compressed(
        save_path,
        states=states,
        actions=actions,
        images=images,
        traj_lengths=traj_lengths
    )

    print(f"Saved npz dataset to: {save_path}")
    print(f"states: {states.shape}, actions: {actions.shape}, images: {images.shape}, episodes: {len(traj_lengths)}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--zarr_path", type=str, required=True, help="输入 Zarr 文件路径")
    # parser.add_argument("--save_dir", type=str, default="./data/robomimic", help="输出 npz 存放路径")
    # args = parser.parse_args()

    # convert_zarr_to_npz(args.zarr_path, args.save_dir)

    zarr_path = "/home/ani/ReinFlow/data/book30.zarr"
    save_dir = "./data/book"
    convert_zarr_to_npz(zarr_path, save_dir)
