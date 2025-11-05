# python data_process/vis_zarr.py

import os
import cv2
import numpy as np
import zarr
import matplotlib.pyplot as plt


# =================== 配置 ===================
zarr_path = "/home/ani/ReinFlow/data/book30.zarr"  # Zarr路径
episode_id = 10   # 想查看的episode编号（从0开始）
fps = 20         # 播放速度（帧/秒）


# =================== 加载函数 ===================
def load_zarr_episode(zarr_path, episode_id):
    print(f"🔍 Opening Zarr dataset: {zarr_path}")

    z_root = zarr.open(zarr_path, mode='r')
    z_data = z_root['data']
    z_meta = z_root['meta']

    # 加载 datasets
    global_color = z_data['global_color']
    wrist_color = z_data['wrist_color']
    eef_pose = z_data['eef_pose']
    joint = z_data['joint']
    episode_ends = np.array(z_meta['episode_ends'])

    # 计算 episode 起止索引
    if episode_id == 0:
        start_idx = 0
    else:
        start_idx = episode_ends[episode_id - 1]
    end_idx = episode_ends[episode_id]

    print(f"🎬 Loading episode_{episode_id}: frames {start_idx} → {end_idx - 1}")
    print(f"📦 Total frames: {end_idx - start_idx}")

    # 读取该 episode 的数据
    global_imgs = np.array(global_color[start_idx:end_idx])
    wrist_imgs = np.array(wrist_color[start_idx:end_idx])
    eef_pose_array = np.array(eef_pose[start_idx:end_idx])
    joint_array = np.array(joint[start_idx:end_idx])

    # ===== 修改开始：将图像缩放为 96x96 =====
    global_imgs = np.array([cv2.resize(img, (96, 96)) for img in global_imgs])
    wrist_imgs = np.array([cv2.resize(img, (96, 96)) for img in wrist_imgs])
    # ===== 修改结束 =====

    print(f"✅ Loaded episode {episode_id} successfully.\n")
    return wrist_imgs, global_imgs, eef_pose_array, joint_array


# =================== 可视化函数 ===================
def visualize_episode(wrist_imgs, global_imgs, eef_pose, joint, fps=20, episode_id=0):
    num_frames = len(wrist_imgs)
    print(f"🖼️  Playing {num_frames} frames (press ESC to quit, SPACE to pause)...")

    delay = int(1000 / fps)
    paused = False
    i = 0

    while True:
        if not paused:
            wrist = wrist_imgs[i]
            global_ = global_imgs[i]

            # 拼接 + 显示
            combined = np.hstack((
                cv2.putText(wrist.copy(), "Wrist", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2),
                cv2.putText(global_.copy(), "Global", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            ))

            cv2.imshow(f"Episode Player (Episode {episode_id})", combined)

            # 打印前几帧的姿态数据
            if i < 3:
                print(f"Frame {i}: eef_pose = {np.round(eef_pose[i], 4)}, joint = {np.round(joint[i], 4)}")

            i += 1
            if i >= num_frames:
                print("🎞️  Reached end of episode.")
                break

        key = cv2.waitKey(delay)
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            paused = not paused

    cv2.destroyAllWindows()
    print("✅ Done.\n")


# =================== 主函数 ===================
def main():
    wrist_imgs, global_imgs, eef_pose, joint = load_zarr_episode(zarr_path, episode_id)
    visualize_episode(wrist_imgs, global_imgs, eef_pose, joint, fps=fps, episode_id=episode_id)


if __name__ == "__main__":
    main()
