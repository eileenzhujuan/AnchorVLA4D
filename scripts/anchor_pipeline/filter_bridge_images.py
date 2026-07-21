"""Post-processing script for limiting anchor usage to a single anchor.

This script is intended to run after add_anchor.py. Its purpose is to
restrict the total number of anchors attached to each sample to one by
compressing or trimming the image list and message placeholders.

Usage:
    python filter_bridge_images.py <input_json> [output_json]
"""

import json
import os
import sys
import re


def normalize_images_field(entry):
    images = entry.get("images")
    if images is None:
        return []
    if isinstance(images, str):
        return [images]
    if isinstance(images, list):
        return images
    return []


def normalize_message_content(entry):
    messages = entry.get("messages")
    if not isinstance(messages, list):
        return entry

    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue

        # 将多个连续的 <image> / <video> 占位符压缩为一个
        content = re.sub(r"(<image>)+", "<image>", content)
        content = re.sub(r"(<video>)+", "<video>", content)

        # 将类似 "given initial anchors <image><image>" 这类连续占位符也压缩
        content = re.sub(r"(<image>)(\s*<image>)+", "<image>", content)
        content = re.sub(r"(<video>)(\s*<video>)+", "<video>", content)

        message["content"] = content

    return entry


def filter_images(entries):
    filtered = []
    for entry in entries:
        images = normalize_images_field(entry)
        length = len(images)
        if length <= 1:
            continue

        if length > 2:
            entry["images"] = images[-2:]
        else:
            entry["images"] = images

        entry = normalize_message_content(entry)
        filtered.append(entry)
    return filtered


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main(src_path, dst_path=None):
    if dst_path is None:
        base, ext = os.path.splitext(src_path)
        dst_path = f"{base}_filtered{ext}"

    dataset = load_json(src_path)
    if not isinstance(dataset, list):
        raise ValueError("Expected input JSON to be a list of entries.")

    filtered = filter_images(dataset)
    save_json(dst_path, filtered)
    print(f"Filtered {len(dataset)} entries to {len(filtered)} entries.")
    print(f"Saved filtered JSON to: {dst_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python filter_bridge_images.py <input_json> [output_json]")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
