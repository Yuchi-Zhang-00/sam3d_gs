#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
prompt + image -> SAM3 多物体分割 mask 的完整脚本：

1. 用 Qwen3-VL-8B-Thinking 看图，生成若干条英文物体描述 prompt_list
2. 对每条 prompt 调用 SAM3 分割：
   - 输出到 agent_output_multi/obj_i/*.json
   - json 里包含 pred_masks（RLE）、overlay 图路径等
3. 将所有 obj_i/*.json 里的 pred_masks 解码为 PNG 二值 mask：
   - 保存到 agent_output_multi/masks/obj_i/<json_name>/mask_k.png

之后，你的 run_sam3d_multi.py 里的 --mask-root
可以直接指向 agent_output_multi/masks。
"""

import os
import ast
import json
import argparse
from functools import partial
from typing import Optional

import numpy as np
import torch
from PIL import Image
import pycocotools.mask as mask_util

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.agent.client_llm import send_generate_request as send_generate_request_orig
from sam3.agent.client_sam3 import call_sam_service as call_sam_service_orig


# =========================
# 0. 环境变量（可按需精简）
# =========================



# =========================
# 1. LLM 配置（Qwen3-VL）
# =========================

LLM_CONFIGS = {
    # vLLM-served models
    "qwen3_vl_8b_thinking": {
        "provider": "vllm",
        # model 不再写死，在 build_llm_config 时通过参数传入
        "model": None,
    },
}


def build_llm_config(
    name: str = "qwen3_vl_8b_thinking",
    model_id: Optional[str] = None,
):
    """
    构建 LLM config：
    - name: 在 LLM_CONFIGS 里的 key
    - model_id: 要发给 vLLM 的模型名称（需与 --served-model-name 一致）
    """
    cfg = LLM_CONFIGS[name].copy()
    cfg["name"] = name
    cfg["api_key"] = "LOCAL_VLLM"

    if model_id is not None:
        cfg["model"] = model_id
    elif cfg.get("model") is None:
        raise ValueError(
            "LLM model id is not set. Please pass --llm-model-id to match vLLM --served-model-name."
        )

    if cfg["provider"] == "vllm":
        server_url = "http://127.0.0.1:8001/v1"
    else:
        server_url = cfg["base_url"]

    return cfg, server_url


# =========================
# 2. SAM3 模型构建
# =========================

def build_sam3_processor() -> Sam3Processor:
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(bpe_path=bpe_path)
    processor = Sam3Processor(model, confidence_threshold=0.5)
    return processor


# =========================
# 3. Qwen 生成场景 prompt_list
# =========================

def generate_scene_prompts_with_qwen(
    image_path: str,
    send_generate_request,
    llm_config: dict,
    max_prompts: int = 12,
    system_prompt_path: str = "examples/system_prompt_scene_prompts.txt",
):
    """
    1. 调 Qwen3-VL-8B-Thinking，看图生成可分割对象的英文短 prompt 列表。
    2. 更鲁棒地解析 <prompt_list>...[...]...</prompt_list>，在缺少 closing tag 时也能工作。
    3. 自动清洗掉 </think> 等无效内容。
    """

    # 1) 读取 system prompt
    if not os.path.exists(system_prompt_path):
        raise FileNotFoundError(f"system prompt file not found: {system_prompt_path}")

    with open(system_prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    # 2) 构造 messages（带 image_url）
    image_path = os.path.abspath(image_path)
    image_url = f"file://{image_path}"

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are given the image above. "
                        "Follow the instructions in the system prompt to analyze the scene, "
                        "then output both <analysis>...</analysis> and <prompt_list>...</prompt_list>. "
                        "Do NOT omit the <prompt_list> block. The <prompt_list> block must be a valid Python list of strings."
                    ),
                },
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]

    # 3) 调用 vLLM / Qwen
    resp = send_generate_request(messages=messages)

    # 4) 统一拿到 raw_text
    if isinstance(resp, str):
        raw_text = resp
    elif isinstance(resp, dict):
        try:
            raw_text = resp["choices"][0]["message"]["content"]
        except Exception:
            raw_text = str(resp)
    else:
        try:
            raw_text = resp.choices[0].message.content
        except Exception:
            raw_text = str(resp)

    raw_text = raw_text.strip()

    # ---------------------------
    # 5) 尝试从 <prompt_list> 中抽取“[...]”这段
    # ---------------------------
    list_block = raw_text

    # 先截掉 <prompt_list> 前面的分析内容
    if "<prompt_list>" in raw_text:
        after_tag = raw_text.split("<prompt_list>", 1)[1]
        list_block = after_tag
    # 如果有 closing tag，再截掉后面
    if "</prompt_list>" in list_block:
        list_block = list_block.split("</prompt_list>", 1)[0]

    # 从 list_block 中找第一个 '[' 和最后一个 ']'，尽量拿到一个完整的 Python list 字符串
    inner = None
    start = list_block.find("[")
    end = list_block.rfind("]")
    if start != -1 and end != -1 and end > start:
        inner = list_block[start : end + 1].strip()

    # 如果还是没拿到，就 fallback：把整个 list_block 当作 inner
    if inner is None:
        inner = list_block.strip()

    # ---------------------------
    # 6) 解析 inner -> Python list[str]
    # ---------------------------
    prompt_list: list[str] = []

    # 优先 literal_eval
    try:
        data = ast.literal_eval(inner)
        if isinstance(data, list):
            prompt_list = [
                s.strip()
                for s in data
                if isinstance(s, str) and s.strip()
            ]
        else:
            raise ValueError("parsed object is not a list")
    except Exception:
        # fallback：行级解析（更严格一点，只收“看起来像短 prompt”的行）
        lines = [l.strip() for l in inner.splitlines() if l.strip()]
        tmp: list[str] = []
        for l in lines:
            # 跳过明显是 tag 或分析段落的行
            if l.startswith("<") and l.endswith(">"):
                continue
            if l in ("<think>", "</think>"):
                continue

            # 如果是形如 1. xxx / 2) xxx
            if l[0].isdigit():
                parts = l.split(maxsplit=1)
                if len(parts) == 2:
                    candidate = parts[1].lstrip(".)").strip()
                else:
                    candidate = l
            else:
                candidate = l

            # 简单过滤掉过长的整段分析（比如一个大段落 > 200 字符）
            if len(candidate) > 200:
                continue

            if candidate:
                tmp.append(candidate)

        prompt_list = tmp

    # ---------------------------
    # 7) 最后再清洗一遍 prompt_list
    # ---------------------------
    cleaned: list[str] = []
    for s in prompt_list:
        s = s.strip()
        if not s:
            continue
        # 丢掉残余的 tag / think
        if s.startswith("<") and s.endswith(">"):
            continue
        if s in ("<think>", "</think>"):
            continue
        cleaned.append(s)

    prompt_list = cleaned[:max_prompts]
    return raw_text, prompt_list


# =========================
# 4. JSON → PNG mask 工具
# =========================

def safe_name(name: str) -> str:
    """简单处理一下名字中的空格，避免路径问题。"""
    return name.replace(" ", "_")


def decode_rle_mask(counts: str, h: int, w: int) -> np.ndarray:
    """将 SAM3/COCO RLE 字符串解码为 (h, w) 的 0/1 uint8 mask。"""
    rle = {"counts": counts.encode("utf-8"), "size": [h, w]}
    mask = mask_util.decode(rle)   # (h, w, 1) 或 (h, w)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.uint8)


def convert_agent_json_to_masks(agent_root: str):
    """
    遍历 agent_root 下的 obj_*/ 目录，
    把所有 json 里的 pred_masks 解码为 PNG mask。

    输出结构：
      agent_root/masks/obj_i/<json_basename>/mask_k.png
    """
    agent_root = os.path.abspath(agent_root)
    mask_root = os.path.join(agent_root, "masks")
    os.makedirs(mask_root, exist_ok=True)

    print(f"[INFO] Converting JSON → PNG masks under: {agent_root}")
    print(f"[INFO] Masks will be saved to: {mask_root}")

    for obj_name in os.listdir(agent_root):
        obj_dir = os.path.join(agent_root, obj_name)
        if not os.path.isdir(obj_dir):
            continue
        if os.path.abspath(obj_dir) == os.path.abspath(mask_root):
            continue

        safe_obj_name = safe_name(obj_name)
        obj_mask_root = os.path.join(mask_root, safe_obj_name)
        os.makedirs(obj_mask_root, exist_ok=True)

        print(f"\n=== Scanning folder: {obj_dir} → {obj_mask_root} ===")

        for root, _, files in os.walk(obj_dir):
            for fname in files:
                if not fname.endswith(".json"):
                    continue

                json_path = os.path.join(root, fname)

                try:
                    with open(json_path, "r") as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"  [SKIP] Failed to load {json_path}: {e}")
                    continue

                # 某些是 list（history log），直接跳过
                if not isinstance(data, dict):
                    print(f"  [SKIP] {json_path}: json is list, not mask dict")
                    continue

                pred_masks = data.get("pred_masks")
                if not pred_masks:
                    print(f"  [SKIP] {json_path}: no pred_masks")
                    continue

                h = data.get("orig_img_h")
                w = data.get("orig_img_w")
                if h is None or w is None:
                    print(f"  [SKIP] {json_path}: missing height/width")
                    continue

                json_basename = os.path.splitext(os.path.basename(json_path))[0]
                safe_json_basename = safe_name(json_basename)

                out_dir = os.path.join(obj_mask_root, safe_json_basename)
                os.makedirs(out_dir, exist_ok=True)

                print(f"  [OK] {json_path}: {len(pred_masks)} masks → {out_dir}")

                scores = data.get("pred_scores", [])
                for i, counts in enumerate(pred_masks):
                    mask = decode_rle_mask(counts, h, w)

                    mask_save_path = os.path.join(out_dir, f"mask_{i+1}.png")
                    Image.fromarray(mask * 255).save(mask_save_path)

                    score_str = f", score={scores[i]:.3f}" if i < len(scores) else ""
                    print(f"    saved mask_{i+1}.png{score_str}")


# =========================
# 5. 主流程：prompt + img -> mask
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-VL + SAM3: prompt+image -> multi-object masks"
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default="sam3/assets/img.jpg",
        help="输入图片路径",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="sam3/agent_output_multi",
        help="SAM3 多物体输出根目录（内部会建 obj_1, obj_2, ...）",
    )
    parser.add_argument(
        "--system-prompt-path",
        type=str,
        default="sam3/examples/system_prompt_scene_prompts.txt",
        help="Qwen 用的 system prompt 文本路径",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=12,
        help="最多保留多少个物体 prompt",
    )
    parser.add_argument(
        "--skip-first",
        action="store_true",
        help="是否丢弃 prompt_list 的第一个元素（如果它更像场景描述而不是具体物体）",
    )
    parser.add_argument(
        "--llm-model-id",
        type=str,
        default="sam3/models",
        help="发送给 LLM 服务的模型名称（需与 vLLM --served-model-name 一致）",
    )

    args = parser.parse_args()


    # 构建 LLM & SAM3
    llm_config, llm_server_url = build_llm_config(
        name="qwen3_vl_8b_thinking",
        model_id=args.llm_model_id,
    )
    processor = build_sam3_processor()

    send_generate_request = partial(
        send_generate_request_orig,
        server_url=llm_server_url,
        model=llm_config["model"],
        api_key=llm_config["api_key"],
    )
    call_sam_service = partial(call_sam_service_orig, sam3_processor=processor)

    image = os.path.abspath(args.image_path)
    output_root = os.path.abspath(args.output_root)
    os.makedirs(output_root, exist_ok=True)

    # 1) Qwen 生成场景 prompt_list
    print(f"[INFO] Generating prompts for image: {image}")
    raw_text, prompt_list = generate_scene_prompts_with_qwen(
        image_path=image,
        send_generate_request=send_generate_request,
        llm_config=llm_config,
        max_prompts=args.max_prompts,
        system_prompt_path=args.system_prompt_path,
    )

    print("\n====== 原始 Qwen 输出（raw_text，截断开头 800 字） ======")
    print(raw_text[:800])
    print("......\n")

    if args.skip_first and len(prompt_list) > 1:
        prompt_list = prompt_list[1:]

    print("====== 解析后的 prompt_list ======")
    for i, p in enumerate(prompt_list, start=1):
        print(f"{i}. {p}")

    # 2) 逐个 prompt 调用 SAM3，写入 json
    for i, prompt in enumerate(prompt_list, start=1):
        print(f"\n================ [Prompt {i}] {prompt} ================\n")

        this_output_dir = os.path.join(output_root, f"obj_{i}")
        os.makedirs(this_output_dir, exist_ok=True)

        json_path = call_sam_service(
            image_path=image,
            text_prompt=prompt,
            output_folder_path=this_output_dir,
        )
        print(f"[OK] SAM3 output json: {json_path}")

    # 3) 把所有 json 里的 pred_masks 解码为 PNG mask
    convert_agent_json_to_masks(output_root)

    print("\n✅ All done. Masks are under:")
    print(f"   {os.path.join(output_root, 'masks')}")


if __name__ == "__main__":
    main()
