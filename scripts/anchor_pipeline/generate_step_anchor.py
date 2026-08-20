"""Generate a step-anchor JSON file from an existing dataset JSON.

For each sample the anchors are images of the same episode whose frame_idx is
``frame_idx - k*step`` for k=1,2,... ; when the past frame index would go below
0 it falls back to frame 0. ``frame_idx`` is the second-to-last ``_``-separated
number in the image file name, e.g. ``episode_1_13_0.jpg`` -> 13.

``--max_images`` caps the total number of images kept per sample (1 current
image + up to ``max_images - 1`` anchors); omit it to keep all anchors.

Usage:
    python generate_step_anchor.py <input_json> --step 10 [--max_images 2] [--output out.json]
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_key(img: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    """Return (episode_id, frame_idx) parsed from an image path."""
    if not img:
        return None, None
    parts = os.path.basename(str(img)).rsplit(".", 1)[0].split("_")
    if len(parts) < 2:
        return None, None
    ep = re.search(r"episode_(\d+)", str(img))
    try:
        return (int(ep.group(1)) if ep else None), int(parts[-2])
    except (TypeError, ValueError):
        return None, None


def current_image(sample: Dict[str, Any]) -> Optional[str]:
    """The current image is the last one in ``images`` (anchors are prepended)."""
    imgs = sample.get("images")
    if isinstance(imgs, str):
        return imgs
    return imgs[-1] if imgs else None


def fill_placeholders(content: Any, n: int) -> Any:
    """Ensure content has n anchor + 1 current ``<image>`` placeholders.

    If the content already carries a ``given previous anchors`` prefix the
    missing placeholders are inserted before the first ``<image>``; otherwise
    the prefix is prepended.
    """
    if not isinstance(content, str) or n <= 0:
        return content
    count = content.count("<image>")
    if count >= n + 1:
        return content
    if "given previous anchors" in content:
        i = content.index("<image>")
        return content[:i] + "<image>" * (n + 1 - count) + content[i:]
    text = content.lstrip()
    if text.startswith("given current image"):
        text = text.replace("given current image", "current image", 1)
    return "given previous anchors " + "<image>" * n + ", and " + text


def build_anchor_path(img: str, af: int) -> str:
    """Build the anchor image path by replacing the frame number in the file
    name with ``af`` (the frame is the second-to-last ``_``-separated number)."""
    directory = os.path.dirname(str(img))
    base, ext = os.path.splitext(os.path.basename(str(img)))
    parts = base.split("_")
    parts[-2] = str(af)
    new_base = "_".join(parts) + ext
    return os.path.join(directory, new_base) if directory else new_base


def generate_step_anchors(dataset: List[Dict[str, Any]], step: int, max_images: Optional[int]):
    max_anchors = max_images - 1 if max_images else None  # images cap minus the current image
    results, missing, trajs = [], 0, set()

    for s in dataset:
        cur = current_image(s)
        ep, fr = parse_key(cur)
        if ep is not None:
            trajs.add(ep)
        if fr is None:
            results.append(s)
            continue

        anchors, k = [], 1
        while max_anchors is None or len(anchors) < max_anchors:
            af = max(0, fr - k * step)  # past frame idx < 0 -> fall back to 0
            img = build_anchor_path(cur, af)
            if img == cur or img in anchors:
                break
            anchors.append(img)
            if af == 0:
                break
            k += 1

        if not anchors:
            missing += 1
            results.append(s)
            continue

        s = dict(s)
        imgs = s["images"]
        s["images"] = anchors + ([imgs] if isinstance(imgs, str) else list(imgs))
        msg = (s.get("messages") or [{}])[0]
        if msg.get("role") == "user":
            msg = dict(msg)
            msg["content"] = fill_placeholders(msg.get("content"), len(anchors))
            s["messages"] = [msg] + s.get("messages", [])[1:]
        results.append(s)

    return results, missing, len(trajs)


def build_output_path(input_path: str, step: int, max_images: Optional[int]) -> str:
    p = Path(input_path)
    tag = f"step{step}" + (f"_max{max_images}" if max_images else "")
    return str(p.with_name(f"{p.stem}_{tag}_anchor{p.suffix}"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate step-anchor JSON from a dataset JSON.")
    parser.add_argument("input_json", help="Source dataset json (a list of samples)")
    parser.add_argument("--step", type=int, required=True, help="Anchor frame = current frame - k*step")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Max images per sample (1 current + anchors), e.g. 2 keeps at most 1 anchor")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.step <= 0:
        parser.error("--step must be a positive integer")

    with open(args.input_json, encoding="utf-8") as f:
        dataset = json.load(f)
    if not isinstance(dataset, list):
        raise ValueError("Input JSON must be a list of samples.")

    new_dataset, missing, n_traj = generate_step_anchors(dataset, args.step, args.max_images)
    output = args.output or build_output_path(args.input_json, args.step, args.max_images)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(new_dataset, f, ensure_ascii=False, indent=2)

    print(f"samples: {len(dataset)}, with_anchor: {len(dataset) - missing}, missing: {missing}")
    print(f"trajectories: {n_traj}")
    print(f"wrote: {output}")


if __name__ == "__main__":
    main()
