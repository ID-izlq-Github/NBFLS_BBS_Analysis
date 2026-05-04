#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取 data/text/processed_v2.jsonl，调用 DeepSeek API 解析帖子，
实时写入 data/feature_data/extracted_posts.csv。
"""

import asyncio
import aiohttp
import json
import os
import re
import csv
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("extract_features.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------- 配置 ----------------------
API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-api-key")
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL = "deepseek-ai/DeepSeek-V4-Flash"
MAX_CONCURRENT = 40
MAX_RETRIES = 3
BATCH_WRITE_SIZE = 40  # 每处理多少张图片写一次 CSV

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "text" / "processed_v2.jsonl"
OUTPUT_FILE = BASE_DIR / "data" / "feature_data" / "extracted_posts.csv"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

CLASS_PATTERN = re.compile(r"^([SJ])(\d{2})(\d{2})$")


def infer_department(class_str: Optional[str]) -> Optional[str]:
    if not class_str:
        return None
    m = CLASS_PATTERN.match(class_str.strip().upper())
    if not m:
        return None
    prefix, year, num = m.group(1), int(m.group(2)), int(m.group(3))
    if prefix == "J":
        return "J"
    if year in (23, 24):
        return "I"
    if year == 25:
        return "S" if num <= 6 else "I"
    if year == 26:
        return "S" if num <= 8 else "I"
    if year == 27:
        return "S" if num <= 10 else "I"
    # 28届同27届
    if year == 28:
        return "S" if num <= 10 else "I"
    logger.warning(f"未知毕业年份的班级: {class_str}")
    return None


def adjust_sentiment(post: Dict[str, Any]) -> str:
    sentiment = post.get("sentiment", "中性")
    purpose = post.get("purpose", "")
    text = post.get("post_text", "")
    if purpose in ("表白", "祝福"):
        return "积极"
    negative_words = ["挂人", "恶心", "烦死了", "讨厌", "滚", "垃圾"]
    if any(w in text for w in negative_words):
        return "消极"
    return sentiment


SYSTEM_PROMPT = """你是一个校园墙帖子解析器。输入是一张图片中识别出的全部文本，可能包含多个帖子。请完成以下任务：

1. 将文本分割为独立的帖子，分割依据包括但不限于：换行、时间戳、系统消息（如“解除关系 加为好友”）、“谢谢墙”“墙墙”“墙！”等常见结束语。
2. 对每个帖子提取指定字段，最终返回一个 JSON 数组。

每个帖子对象需包含以下字段：

- post_text (string): 该帖子对应的原始文本片段，尽量完整保留。
- author_name (string|null): 发帖人的自称姓名、昵称或代号。剔除明显非人名的符号、ocr 错误（如“国凯”“画凯”等），多个人名用空格分隔。
- author_class (string|null): 发帖人班级，必须为“1位字母 + 4位数字”格式（如 S2602, J2210）。若原文为四位数字（如“2403”），或其他不规范形式（如“初二三班”）请根据发帖时间、毕业年份和学部规则推断出完整班级（可参考班级→学部映射规则有关班级名称的格式介绍）；出现多个保留出现次数最多的，次数相同则取首次出现的；无法推断则填 null。
- author_department (string|null): 发帖人学部，仅允许单个字母：I（国际部）、S（普高部）、J（初中部）。若原文明确提到部门则直接映射；否则按班级→学部规则推断。若出现多个部门字母，保留出现次数最多的，次数相同则取首次出现的。
- is_anonymous (boolean): 是否匿名。“匿了”“匿名”“匿”“不实名”等 → true；“不匿”“不腻”“不匿名”“实名”“本人”等 → false。若文中无明显匿名/实名标识，默认为 true（匿名）。
- purpose (string): 发帖目的，从 ["表白","寻人","寻物","祝福","吐槽","推广","提问","其他"] 中选择。
- target_name (string|null): 目标对象姓名或昵称，多人则用空格分隔。
- target_class (string|null): 目标对象班级，格式同 author_class。
- target_department (string|null): 目标对象学部，规则同 author_department。
- description (string|null): 对目标对象的完全标签化概括（如“帅”“唱歌好听”“绿衣服”），不必留空，可基于原文推断，空格分割标签
- sentiment (string): 情感倾向，从 ["积极","消极","中性"] 中选择。

【班级→学部映射规则】
班级格式为“前缀(大写S/J) + 两位毕业年份 + 两位班级序号”。
- 若前缀为 J → 学部为 J（初中部）。
- 若前缀为 S：
  - 22届：序号 01-06 → S（普高部）；07及以上 → I（国际部）
  - 23、24届：全部 → I
  - 25届：01-06 → S；07-09 → I
  - 26届：01-08 → S；09-11 → I
  - 27届：01-10 → S；11-13 → I
  - 28届：同27届规则
  - 其他未知届数或无法解析时，填 null。

【注意】
- 严格区分发帖人和目标对象，切勿混淆。
- “国凯”“画凯”等常为 OCR 错误，多出现在文本首尾，请甄别并丢弃。
- 尽量从上下文推断字段值，不轻易置 null；确实无信息才用 null。
- 所有帖子必须填充 post_text；即使只识别到一个帖子，也要返回包含一个对象的数组。
- 最终输出格式：{"posts": [ { ... }, { ... } ]}
"""


def build_user_prompt(image_id: str, date: str, plain_text: str) -> str:
    return f'图片ID: {image_id}\n日期: {date}\n文本:\n"""\n{plain_text}\n"""'


async def fetch_posts(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    image_id: str,
    date: str,
    plain_text: str,
    retries: int = MAX_RETRIES,
) -> List[Dict[str, Any]]:
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(image_id, date, plain_text)},
        ],
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
        "temperature": 0.5,
        "max_tokens": 2000,
    }

    async with semaphore:
        for attempt in range(1, retries + 1):
            try:
                async with session.post(
                    API_URL, headers=headers, json=payload, timeout=30
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = data["choices"][0]["message"]["content"]
                        parsed = json.loads(content)
                        return parsed.get("posts", [])
                    elif resp.status == 429:
                        retry_after = resp.headers.get("Retry-After", "5")
                        logger.warning(f"429 限流 {image_id}, 等待 {retry_after}s")
                        await asyncio.sleep(float(retry_after))
                    elif resp.status == 503:
                        logger.warning(f"503 服务不可用 {image_id}, 重试")
                        await asyncio.sleep(2**attempt)
                    else:
                        text = await resp.text()
                        logger.error(f"API {resp.status} for {image_id}: {text[:200]}")
                        return []
            except asyncio.TimeoutError:
                logger.error(f"超时 {image_id} (尝试 {attempt}/{retries})")
                if attempt < retries:
                    await asyncio.sleep(2**attempt)
            except aiohttp.ClientError as e:
                logger.error(f"网络错误 {image_id}: {e} (尝试 {attempt}/{retries})")
                if attempt < retries:
                    await asyncio.sleep(2**attempt)
            except json.JSONDecodeError as e:
                logger.error(f"JSON解析失败 {image_id}: {e}")
                return []
            except KeyError as e:
                logger.error(f"响应字段缺失 {image_id}: {e}")
                return []
            except Exception as e:
                logger.error(f"未知异常 {image_id}: {type(e).__name__}: {e}")
                return []
        return []


def process_posts(
    image_id: str, date: str, posts: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    rows = []
    for idx, post in enumerate(posts):
        # 可选：后处理部门覆盖（目前保留 LLM 输出，若需要规则校准可在此覆盖）
        # post["author_department"] = infer_department(post.get("author_class"))
        # post["target_department"] = infer_department(post.get("target_class"))
        # post["sentiment"] = adjust_sentiment(post)

        row = {
            "image_id": image_id,
            "post_id": f"{image_id}_{idx+1:02d}",
            "date": date,
            "author_name": post.get("author_name"),
            "author_class": post.get("author_class"),
            "author_department": post.get("author_department"),
            "is_anonymous": post.get("is_anonymous", False),
            "purpose": post.get("purpose"),
            "target_name": post.get("target_name"),
            "target_class": post.get("target_class"),
            "target_department": post.get("target_department"),
            "description": post.get("description"),
            "sentiment": post.get("sentiment", "中性"),
            "post_text": post.get("post_text", ""),
        }
        rows.append(row)
    return rows


CSV_COLUMNS = [
    "image_id",
    "post_id",
    "date",
    "author_name",
    "author_class",
    "author_department",
    "is_anonymous",
    "purpose",
    "target_name",
    "target_class",
    "target_department",
    "description",
    "sentiment",
    "post_text",
]


# ！！！关键修复：改为普通函数
def write_rows_to_csv(rows: List[Dict[str, Any]]):
    """同步写入 CSV，可由 asyncio.to_thread 调用"""
    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writerows(rows)


async def main():
    if API_KEY == "your-api-key":
        logger.error("请设置环境变量 DEEPSEEK_API_KEY")
        return

    if not INPUT_FILE.exists():
        logger.error(f"输入文件不存在: {INPUT_FILE}")
        return

    tasks_data = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("is_valid") and obj.get("plain_text"):
                    tasks_data.append(
                        {
                            "image_id": obj["image_id"],
                            "date": obj["date"],
                            "plain_text": obj["plain_text"],
                        }
                    )
            except json.JSONDecodeError as e:
                logger.error(f"跳过无效 JSON 行: {e}")

    logger.info(f"共 {len(tasks_data)} 条有效图片待处理")

    # 初始化 CSV（覆盖）
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    total_posts = 0
    processed_images = 0

    async with aiohttp.ClientSession() as session:
        # 分批处理，每 BATCH_WRITE_SIZE 张图片写一次
        for i in range(0, len(tasks_data), BATCH_WRITE_SIZE):
            batch = tasks_data[i : i + BATCH_WRITE_SIZE]
            tasks = [
                fetch_posts(
                    session, semaphore, d["image_id"], d["date"], d["plain_text"]
                )
                for d in batch
            ]
            results = await asyncio.gather(*tasks)

            batch_rows = []
            for data, posts in zip(batch, results):
                processed_images += 1
                if posts:
                    rows = process_posts(data["image_id"], data["date"], posts)
                    batch_rows.extend(rows)
                else:
                    logger.info(
                        f"图片 {data['image_id']} 未提取到帖子（可能无内容或API失败）"
                    )
            if batch_rows:
                # 在线程中执行同步写入
                await asyncio.to_thread(write_rows_to_csv, batch_rows)
                total_posts += len(batch_rows)
            logger.info(
                f"进度: {processed_images}/{len(tasks_data)} 图片, 目前已提取 {total_posts} 个帖子"
            )

    logger.info(f"处理完成，共提取 {total_posts} 个帖子，保存至 {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
