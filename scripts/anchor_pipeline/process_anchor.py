"""Anchor keyframe extraction script.

Purpose:
- Read the dataset JSON index and organize samples in trajectory order.
- Extract kinematic keyframes based on state changes and gripper changes.
- Extract semantic keyframes based on DINOv2 visual features.
- Write the results to a JSON file for downstream training or data processing.

Usage:
    python process_anchor.py <json_file> <output_file_name>
"""

import os
import sys
import re
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel # 或者 import timm (用于DINOv2)

# ==================== 配置区 ====================
GRIPPER_THRESHOLD = 0.1                # 夹爪状态变化的阈值
VELOCITY_THRESHOLD = 0.2             # 速度接近0的阈值
DINO_SIM_THRESHOLD = 0.2             # 视觉相似度跌破该值时记为新Anchor
BATCH_SIZE = 32                       # 视觉提取时的Batch大小
DATA_DIR = "/root/datasets"

if hasattr(torch, "npu") and torch.npu.is_available():
    DEVICE = "npu"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
# ===============================================


def load_dataset_json_list(json_file):
    """串联或按轨迹排序读取所有的json文件"""
    with open(json_file, 'r') as file:
        dataset = json.load(file)
        
    sorted_data = sorted(dataset, key=lambda x: (int(x["images"][-1].split('_')[-3]), int(x["images"][-1].split('_')[-2])))
    return sorted_data


def _get_image_path(entry, data_dir=None):
    """从样本中提取图片路径，兼容旧版字段名。"""
    if data_dir is None:
        data_dir = DATA_DIR

    if isinstance(entry.get('images'), list) and entry['images']:
        image_path = entry['images'][0]
    elif entry.get('image_path'):
        image_path = entry['image_path']
    elif entry.get('json_filename'):
        image_path = entry['json_filename']
    else:
        image_path = None

    if not image_path:
        return None
    if os.path.isabs(image_path):
        return image_path
    return os.path.join(data_dir, image_path)

def get_episode_path(image_path):
    # 匹配 "episode_" 后面紧跟的连续数字
    match = re.search(r"episode_(\d+)", image_path)   
    if match:
        episode_number = match.group(1)
        return episode_number
    return -1


def _get_state_vector(entry):
    """从样本中取出状态向量，兼容旧版 current_velocity / states。"""
    if isinstance(entry.get('states'), list) and entry['states']:
        values = np.asarray(entry['states'], dtype=np.float32)
        if len(values) >= 6:
            return values[:6]
        return values
    if isinstance(entry.get('state'), list) and entry['state']:
        values = np.asarray(entry['state'], dtype=np.float32)
        if len(values) >= 6:
            return values[:6]
        return values
    if isinstance(entry.get('current_velocity'), list) and entry['current_velocity']:
        return np.asarray(entry['current_velocity'], dtype=np.float32)
    return None


def _get_gripper_state(entry):
    """提取夹爪状态，优先从 states 的最后一个值。"""
    if isinstance(entry.get('states'), list) and entry['states']:
        return float(entry['states'][-1])
    if isinstance(entry.get('state'), list) and entry['state']:
        return float(entry['state'][-1])
    if 'gripper_state' in entry:
        return float(entry['gripper_state'])
    return None


def extract_kinematic_keyframes(dataset, data_dir=None):
    """
    第一级：基于当前 JSON 的状态向量和夹爪状态进行提取。
    """
    if data_dir is None:
        data_dir = DATA_DIR

    kinematic_keyframes = []

    for i in range(len(dataset)):
        current = dataset[i]
        is_keyframe = False
        reason = ""

        curr_state = _get_state_vector(current)
        curr_gripper = _get_gripper_state(current)
        filename = _get_image_path(current, data_dir)

        if i > 0:
            prev_state = _get_state_vector(dataset[i - 1])
            prev_gripper = _get_gripper_state(dataset[i - 1])

            if prev_gripper is not None and curr_gripper is not None and abs(curr_gripper - prev_gripper) > GRIPPER_THRESHOLD:
                is_keyframe = True
                reason = f"gripper_changed_to_{curr_gripper:.2f}_from_{prev_gripper:.2f}"
                print(f'检测到夹爪状态变化的帧: {os.path.basename(filename)}, 夹爪状态从 {prev_gripper:.2f} 变为 {curr_gripper:.2f}，标记为关键帧。')
            elif curr_state is not None and prev_state is not None:
                delta = curr_state - prev_state
                vel_mag = np.linalg.norm(delta)

                if vel_mag < VELOCITY_THRESHOLD and i < len(dataset) - 1:
                #     print(f"检测到速度接近零的帧: {i}, 速度幅值: {vel_mag:.4f}，检查下一帧状态变化...")
                # if i < len(dataset) - 1:
                    next_state = _get_state_vector(dataset[i + 1])
                    if next_state is not None:
                        next_delta = next_state - curr_state
                        if np.dot(delta, next_delta) < 0:
                            print(f"检测到状态变化方向反转的帧: {os.path.basename(filename)}, 当前速度幅值: {vel_mag:.4f}，标记为关键帧。")
                            is_keyframe = True
                            reason = "state_bottleneck"

        if is_keyframe:
            kinematic_keyframes.append({
                "index": i,
                "filename": filename,
                "reason": reason
            })

    return kinematic_keyframes

@torch.no_grad()
def extract_semantic_keyframes(dataset, processor, model, data_dir=None, ):
    """
    第二级：使用 DINOv2 对每条样本的第一张图做视觉特征提取。
    兼容当前 bridge JSON 的 images 字段结构。
    """
    if data_dir is None:
        data_dir = DATA_DIR

    semantic_keyframes = []

    anchor_feat = None

    # 记录第一帧作为初始 Anchor
    initial_filename = _get_image_path(dataset[0], data_dir) if dataset else ""
    semantic_keyframes.append({"index": 0, "filename": initial_filename, "reason": "initial_frame"})

    similarity_scores = []
    for i in range(len(dataset)):
        img_path = _get_image_path(dataset[i], data_dir)
        if not img_path or not os.path.exists(img_path):
            continue

        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)

        outputs = model(**inputs)
        curr_feat = outputs.last_hidden_state[:, 0, :]
        curr_feat = curr_feat / curr_feat.norm(dim=-1, keepdim=True)

        if anchor_feat is None:
            anchor_feat = curr_feat
            continue

        similarity = torch.dot(anchor_feat.squeeze(), curr_feat.squeeze()).item()
        similarity_scores.append(similarity)

        if similarity < DINO_SIM_THRESHOLD:
            semantic_keyframes.append({
                "index": i,
                "filename": img_path,
                "reason": f"visual_drift_sim_{similarity:.3f}"
            })
            print(f"视觉相似度跌破阈值: {similarity:.3f}，标记为关键帧: {os.path.basename(img_path)}")
            anchor_feat = curr_feat
    return semantic_keyframes

def resolve_output_path(json_file, output_file_name):
    if os.path.isabs(output_file_name):
        return output_file_name

    data_root = os.path.dirname(os.path.dirname(os.path.abspath(json_file)))
    if not os.path.exists(data_root):
        data_root = DATA_DIR
    return os.path.join(data_root, output_file_name)


def _extract_episode_frame(image_path):
    """Extract episode and frame numbers from an image path."""
    if not image_path:
        return None, None

    match = re.search(r"episode_(\d+)_(\d+)", str(image_path))
    if match:
        return int(match.group(1)), int(match.group(2))

    match = re.search(r"episode_(\d+)", str(image_path))
    if match:
        return int(match.group(1)), None

    return None, None


def check_in_order(dataset, max_samples=10):
    """Check whether the first few samples are already in order and print a warning if not."""
    if not dataset:
        print("WARNING: input dataset is empty; order check skipped.")
        return False

    sample_count = min(max_samples, len(dataset))
    in_order = True
    checked_items = []

    for i in range(1, sample_count):
        prev_entry = dataset[i - 1]
        curr_entry = dataset[i]

        prev_images = prev_entry.get("images") or []
        curr_images = curr_entry.get("images") or []
        prev_path = prev_images[-1] if isinstance(prev_images, list) and prev_images else str(prev_images)
        curr_path = curr_images[-1] if isinstance(curr_images, list) and curr_images else str(curr_images)

        prev_episode, prev_frame = _extract_episode_frame(prev_path)
        curr_episode, curr_frame = _extract_episode_frame(curr_path)

        if prev_episode is not None and curr_episode is not None:
            if prev_episode > curr_episode:
                in_order = False
                break
            if prev_episode == curr_episode and prev_frame is not None and curr_frame is not None and prev_frame > curr_frame:
                in_order = False
                break

        checked_items.append((prev_path, curr_path, prev_episode, prev_frame, curr_episode, curr_frame))

    print("Order check for first {} samples:".format(sample_count))
    for idx, (prev_path, curr_path, prev_episode, prev_frame, curr_episode, curr_frame) in enumerate(checked_items, start=1):
        prev_label = f"ep{prev_episode}" if prev_episode is not None else str(prev_path)
        curr_label = f"ep{curr_episode}" if curr_episode is not None else str(curr_path)
        if prev_frame is not None:
            prev_label += f"/frame{prev_frame}"
        if curr_frame is not None:
            curr_label += f"/frame{curr_frame}"
        print(f"  [{idx}] {prev_label} -> {curr_label}")

    if in_order:
        print("INFO: The first {} samples appear to be in order.".format(sample_count))
    else:
        print("WARNING: The first {} samples are not in order. Please ensure the input JSON is already sorted in the required order before running this pipeline.".format(sample_count))

    print("WARNING: This pipeline assumes the input JSON is already in the required order; otherwise anchor extraction results may be incorrect.")
    return in_order


def main(json_file, output_file_name):
    print("步骤 1: 正在从磁盘加载 JSON 索引...")
    dataset = load_dataset_json_list(json_file)
    check_in_order(dataset)

    print(f"成功加载 {len(dataset)} 帧数据。")

    data_root = os.path.dirname(os.path.dirname(os.path.abspath(json_file)))
    if not os.path.exists(data_root):
        data_root = DATA_DIR
    
    # 聚合汇总
    output_metadata = {
        "dataset_total_frames": len(dataset),
        "kinematic_anchors": [],
        "semantic_anchors": []
    }

    print("正在加载 DINOv2 模型...")
    processor = AutoProcessor.from_pretrained("facebook/dinov2-base", use_fast=True)
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE).eval()

    total_len = len(dataset)
    current_idx = 0
    while current_idx < total_len:
        current_episode = get_episode_path(dataset[current_idx].get("images")[-1])
        end_idx = current_idx + 1
        while end_idx < total_len and get_episode_path(dataset[end_idx].get("images")[-1]) == current_episode:
            end_idx += 1
        print(f"\n处理 episode_{current_episode}，帧范围: {current_idx} - {end_idx - 1}，共 {end_idx - current_idx} 帧。")
        k_keyframes = extract_kinematic_keyframes(dataset[current_idx:end_idx], data_dir=data_root)
        print(f"提取到运动关键帧共: {len(k_keyframes)} 帧。")
        output_metadata["kinematic_anchors"].extend(k_keyframes)
        
        s_keyframes = extract_semantic_keyframes(dataset[current_idx:end_idx], processor, model, data_dir=data_root)
        print(f"提取到视觉关键帧共: {len(s_keyframes)} 帧。")
        output_metadata["semantic_anchors"].extend(s_keyframes)
        current_idx = end_idx
    
    
    out_path = resolve_output_path(json_file, output_file_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output_metadata, f, indent=4, ensure_ascii=False)
        
    print(f"\n🎉 运行成功！Anchor 标记文件已保存至: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python process_anchor.py <json_file> <output_file_name>")
        sys.exit(1)
    json_file = sys.argv[1]
    output_file_name = sys.argv[2]
    main(json_file, output_file_name)