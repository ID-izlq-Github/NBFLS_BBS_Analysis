# scripts/01_clean_text.py
import json
import re
import os
from typing import Dict, List, Optional

# ==================== 黑话映射表（根据实际数据扩充，保持上下文完整性） ====================
SLANG_MAP = {
    # 原有
    "下蛋": "下单",
    "腻了": "匿名了",
    "阳体": "体育活动",
    "表自": "表白",
    "扩列": "加好友",
    "捉嘤湖": "濯缨湖",
    "匿": "匿名",
    "不匿": "不匿名",
    # 新增（从提供片段中发现）
    "墙墙": "墙",  # 保留一个"墙"字，而不是完全删除
    "不腻": "不匿名",
    "腻死": "匿名",
    "栓Q": "谢谢",
    "蟹蟹": "谢谢",
    "捞": "寻找",
    "随舞": "随机舞蹈",
    "随唱": "随机合唱",
    "腻": "匿名",
    "腻了谢谢": "匿名谢谢",
    "不腻了": "不匿名",
    "不腻了谢谢": "不匿名谢谢",
    "腻死谢谢": "匿名谢谢",
}

# ==================== UI 垃圾正则模式（大幅增强） ====================
UI_GARBAGE_PATTERNS = [
    # ---- 系统消息 ----
    r"对方已添加你为好友.*?(?=。|$)",
    r"你设置了允许任何人加你为好友.*?(?=。|$)",
    r"对方撤回了一条消息",
    r"解除关系\s*加为好友",
    r"解除关系",
    r"加为好友",
    r"手机在线\S*",
    r"字体大小设置",
    r"分辨率\d+×\d+",
    r"\d{1,2}:\d{2}\s*(?:AM|PM|上午|下午)?",  # 时间戳
    r"请求添加你为好友",
    r"显示图片过期了",
    # ---- 键盘布局 ----
    r"@\s*[WwEeRrTtYyUuIiOoPp]+\s*",
    r"[Aa]\s*[Ss]\s*[Dd]\s*[Ff]\s*[Gg]\s*[Hh]\s*[Jj]\s*[Kk]\s*[Ll]\s*",
    r"[Zz]\s*[Xx]\s*[Cc]\s*[Vv]\s*[Bb]\s*[Nn]\s*[Mm]\s*",
    r"123\s*空格\s*发送",
    r"空格\s*发送",
    r"我你好那是嗯他这不",
    r"我你好这在不是一今天",
    r"我你好他那嗯是哦不",
    r"我你他好在这不一",
    # ---- UI 符号/标签 ----
    r"<[^>]*>",  # HTML 标签风格
    r"[-—–_=]{3,}",  # 分割线
    r"\[[^]]*\]",  # [用摄影解锁新视野] 等
    r"[①②③④⑤⑥⑦⑧⑨⑩⚫⚪◉◎]",  # 特殊序号
    r"@\s*[圆回国心加]",  # 常见 UI 按钮
    # ---- 文件相关 ----
    r"MP3文件",
    r"\d+\.mp3",
    r"\s+MB",
    # ---- 手机状态栏/电量 ----
    r"\d{1,3}%\s*手机电量",
    r"手机在线\S*",
    r"WiFi\S*",
    # ---- 特殊字符 ----
    r"[^\w\s\u4e00-\u9fa5.,!?;:()（）【】\"'‘’" "“”\n\r，。！？；：]",
]

# 额外的纯文本替换（直接删除的短语）
DIRECT_REMOVE_PATTERNS = [
    # 完整匹配的短语
    r"(?<!\w)(墙墙|国|圃|圆|回|早|道|加|画)(?!\w)",  # 单独的字符，避免在词语中间删除
    r"(?<!\w)(腻了|腻死|腻|不腻)(?!\w)",  # 匿名相关的单独词
    r"(?<!\w)(下单|匿名|不匿名|谢谢墙|辛苦墙|墙)(?!\w)",  # 其他功能词
]


def clean_text(raw: str) -> str:
    """
    清洗OCR文本，保留有意义内容，去除噪音

    Args:
        raw: 原始OCR文本

    Returns:
        清洗后的文本
    """
    if not raw or not isinstance(raw, str):
        return ""

    text = raw

    # 1. 保存原始长度用于后续验证
    original_length = len(text)

    # 2. 先用正则批量清除 UI 垃圾（与之前一致）
    for pat in UI_GARBAGE_PATTERNS:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE | re.MULTILINE)

    # 3. 换行转空格，压缩空白
    text = text.replace("\n", "").replace("\r", "")
    text = re.sub(r"\s+", " ", text)

    # 4. 黑话替换（优先级高，先做映射）
    # 按长度降序排列，避免短词先替换导致长词无法匹配
    sorted_slang_items = sorted(
        SLANG_MAP.items(), key=lambda x: len(x[0]), reverse=True
    )
    for old, new in sorted_slang_items:
        if new is None:  # 如果值为None，则删除该词
            text = re.sub(rf"(?<!\w){re.escape(old)}(?!\w)", " ", text)
        else:
            text = text.replace(old, new)

    # 5. 直接删除无意义的词/符号
    for pattern in DIRECT_REMOVE_PATTERNS:
        text = re.sub(pattern, " ", text)

    # 6. 清理多余空格与乱码
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"^[^a-zA-Z0-9\u4e00-\u9fa5]+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fa5]+$", "", text)
    text = re.sub(r"[^\w\s\u4e00-\u9fa5,.!?;:()（）【】\"'‘’" "“”]+", " ", text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"\n", "", text)

    # 7. 保留至少一个字符，如果全部被清除则返回原内容
    cleaned_text = text.strip()
    if not cleaned_text and original_length > 0:
        # 如果清洗后为空但原文不为空，可能过度清洗，返回部分原内容
        # 只保留中文、英文、数字
        fallback = re.sub(r"[^\w\s\u4e00-\u9fa5]", " ", raw)
        fallback = re.sub(r"\s+", " ", fallback).strip()
        return fallback

    return cleaned_text


def validate_content(text: str) -> bool:
    """
    验证清洗后的内容是否有意义

    Args:
        text: 清洗后的文本

    Returns:
        是否有意义
    """
    if not text or len(text.strip()) <= 5:
        return False

    # 检查是否包含有意义的字符（中文、英文、数字）
    meaningful_chars = re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]", text)
    if len(meaningful_chars) < 2:
        return False

    # 检查是否全是重复字符
    unique_chars = set(text.replace(" ", ""))
    if len(unique_chars) <= 3 and len(text) > 10:
        return False

    return True


def process_record(record: Dict) -> Optional[Dict]:
    """
    处理单个记录

    Args:
        record: 原始记录字典

    Returns:
        处理后的记录，如果无效则返回None
    """
    try:
        raw_text = record.get("text", "")
        if not raw_text:
            return None

        # 清洗文本
        plain = clean_text(raw_text)

        # 创建新的记录副本
        processed_record = record.copy()
        processed_record["plain_text"] = plain
        processed_record["original_text"] = raw_text
        processed_record["text_length"] = len(plain)
        processed_record["is_valid"] = validate_content(plain)
        processed_record["clean_ratio"] = len(plain) / max(len(raw_text), 1)

        # 移除原始text字段
        del processed_record["text"]

        return processed_record
    except Exception as e:
        print(f"处理记录时出错: {e}, 记录: {record}")
        return None


def main():
    data_dir = "/home/z-lq/personal_projects/project_nbfls_bbs/data/"
    input_path = os.path.join(data_dir, "text/ocr_result.jsonl")
    output_dir = os.path.join(data_dir, "text")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "processed_v2.jsonl")

    # 1. 读取所有记录
    records = []
    with open(input_path, "r", encoding="utf-8") as f_in:
        for line_num, line in enumerate(f_in, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError as e:
                print(f"第{line_num}行JSON格式错误: {e}")
                continue

    print(f"读取 {len(records)} 条记录")

    # 2. 按 image_id 排序（转为整数排序）
    #    注意：image_id 是字符串，如 "1778"，转换为 int 排序更自然
    try:
        records.sort(key=lambda x: int(x.get("image_id", "0")))
    except ValueError:
        # 如果有非数字的 image_id，退回字符串排序
        records.sort(key=lambda x: x.get("image_id", ""))

    # 3. 清洗每条文本并添加字段
    valid_records = []
    invalid_count = 0

    with open(output_path, "w", encoding="utf-8") as f_out:
        for i, rec in enumerate(records, 1):
            processed_rec = process_record(rec)

            if processed_rec is not None:
                f_out.write(json.dumps(processed_rec, ensure_ascii=False) + "\n")

                if processed_rec["is_valid"]:
                    valid_records.append(processed_rec)
                else:
                    invalid_count += 1

            if i % 1000 == 0:
                print(f"已处理 {i}/{len(records)} 条记录")

    print(f"清洗完成！")
    print(f"总记录数: {len(records)}")
    print(f"有效记录: {len(valid_records)}")
    print(f"无效记录: {invalid_count}")
    print(f"输出文件: {output_path}")


if __name__ == "__main__":
    main()
