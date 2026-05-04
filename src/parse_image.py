import os
import json
import time
import cv2
import numpy as np
from paddleocr import PaddleOCR
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


# =====================
# 初始化 OCR（每个线程独立实例）
# =====================
def create_ocr_instance():
    """创建独立的 OCR 实例（用于多线程）"""
    return PaddleOCR(
        lang="ch",
        use_textline_orientation=False,
        det_db_thresh=0.1,
        det_db_box_thresh=0.3,
        det_db_unclip_ratio=1.8,
        use_doc_preprocessor=False,
        max_side_len=2000,
    )


# =====================
# 图片预处理
# =====================
def preprocess_image(image_path):
    """对图片进行预处理，增强文字识别效果"""
    img = cv2.imread(image_path)
    if img is None:
        return image_path

    # 如果图片太大，适当缩放
    h, w = img.shape[:2]
    max_size = 2000
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 自适应阈值二值化
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # 保存临时文件
    temp_path = image_path.rsplit(".", 1)[0] + "_processed.jpg"
    cv2.imwrite(temp_path, binary)

    return temp_path


# =====================
# 工具函数
# =====================
def clean_text(text: str) -> str:
    return text.strip().replace("\u3000", " ")


def extract_meta_from_filename(filename: str):
    name = os.path.splitext(filename)[0]
    parts = name.split("_")
    image_id = parts[0] if len(parts) > 0 else ""
    date = parts[1] if len(parts) > 1 else ""
    return image_id, date


# =====================
# 解析和排序功能
# =====================
def parse_ocr_result(result, sort_by_position=True):
    if not result or not isinstance(result, list) or len(result) == 0:
        return ""

    ocr_res = result[0]
    rec_texts = ocr_res.get("rec_texts", [])
    rec_polys = ocr_res.get("rec_polys", [])
    rec_scores = ocr_res.get("rec_scores", [])

    if not rec_texts:
        return ""

    filtered_items = []
    ui_keywords = {
        "复制",
        "划重点",
        "朗读",
        "装扮",
        "翻译",
        "分享",
        "收藏",
        "点赞",
        "评论",
    }

    for text, poly, score in zip(rec_texts, rec_polys, rec_scores):
        if not text or not text.strip():
            continue
        if text.strip() in ui_keywords:
            continue
        if score < 0.3:
            continue

        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        x_min = min(xs)
        y_min = min(ys)
        y_max = max(ys)
        h = y_max - y_min

        filtered_items.append({"text": text.strip(), "x": x_min, "y": y_min, "h": h})

    if not filtered_items:
        return ""

    if sort_by_position:
        filtered_items.sort(key=lambda x: x["y"])
        avg_h = sum(i["h"] for i in filtered_items) / len(filtered_items)
        y_threshold = avg_h * 0.5

        lines = []
        current_line = []

        for item in filtered_items:
            if not current_line:
                current_line.append(item)
            elif abs(item["y"] - current_line[0]["y"]) < y_threshold:
                current_line.append(item)
            else:
                lines.append(current_line)
                current_line = [item]

        if current_line:
            lines.append(current_line)

        sorted_texts = []
        for line in lines:
            line.sort(key=lambda x: x["x"])
            line_text = " ".join(item["text"] for item in line)
            sorted_texts.append(line_text)

        return "\n".join(sorted_texts)
    else:
        return "\n".join([item["text"] for item in filtered_items])


# =====================
# 单张图片处理函数（用于多线程）
# =====================
def process_single_image(image_path, filename, ocr_instance=None):
    """处理单张图片，返回结果字典"""
    if ocr_instance is None:
        ocr_instance = create_ocr_instance()

    try:
        # 尝试预处理
        processed_path = preprocess_image(image_path)
        result = ocr_instance.predict(processed_path)
        text = parse_ocr_result(result, sort_by_position=True)

        # 如果预处理效果不好，尝试原图
        if not text:
            result = ocr_instance.predict(image_path)
            text = parse_ocr_result(result, sort_by_position=True)

        # 清理临时文件
        if os.path.exists(processed_path):
            os.remove(processed_path)

        image_id, date = extract_meta_from_filename(filename)
        return {
            "image_id": image_id,
            "date": date,
            "text": text,
            "filename": filename,
            "success": True,
        }

    except Exception as e:
        image_id, date = extract_meta_from_filename(filename)
        return {
            "image_id": image_id,
            "date": date,
            "text": "",
            "filename": filename,
            "error": str(e),
            "success": False,
        }


# =====================
# 主流程（多线程版本）
# =====================
def main_multithread(num_workers=4, batch_size=10):
    """
    多线程批量处理
    :param num_workers: 线程数（建议 2-4，太多会导致显存不足）
    :param batch_size: 每批处理多少张图片后写入文件
    """
    data_dir = "/home/z-lq/personal_projects/project_nbfls_bbs/data/"
    image_dir = os.path.join(data_dir, "raw_image")
    output_json = os.path.join(data_dir, "raw_text/ocr_result.json")

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    output_jsonl = output_json.replace(".json", ".jsonl")

    image_files = [
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"共找到 {len(image_files)} 张图片")
    print(f"使用 {num_workers} 个线程并行处理")
    print("-" * 50)

    # 准备任务列表
    tasks = [(os.path.join(image_dir, f), f) for f in image_files]

    processed_count = 0
    success_count = 0
    start_time = time.time()

    # 文件写入锁
    file_lock = Lock()

    with open(output_jsonl, "w", encoding="utf-8") as f:
        # 使用线程池
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 为每个线程创建独立的 OCR 实例
            ocr_instances = [create_ocr_instance() for _ in range(num_workers)]

            # 提交所有任务
            future_to_task = {}
            for i, (img_path, filename) in enumerate(tasks):
                # 轮询分配 OCR 实例
                ocr_idx = i % num_workers
                future = executor.submit(
                    process_single_image, img_path, filename, ocr_instances[ocr_idx]
                )
                future_to_task[future] = (img_path, filename)

            # 批量收集结果并写入
            results_buffer = []
            for future in as_completed(future_to_task):
                img_path, filename = future_to_task[future]

                try:
                    result = future.result()
                    results_buffer.append(result)
                    processed_count += 1

                    if result.get("success"):
                        success_count += 1
                        status = "✓"
                        preview = (
                            result["text"][:40].replace("\n", " ")
                            if result["text"]
                            else "无文本"
                        )
                    else:
                        status = "✗"
                        preview = f"错误: {result.get('error', '未知')}"

                    print(
                        f"{status} [{processed_count:4d}/{len(image_files)}] {filename}: {preview}..."
                    )

                    # 批量写入（每 batch_size 张写一次）
                    if len(results_buffer) >= batch_size:
                        with file_lock:
                            for res in results_buffer:
                                # 移除 success 字段（不需要保存到文件）
                                res_copy = {
                                    k: v for k, v in res.items() if k != "success"
                                }
                                f.write(json.dumps(res_copy, ensure_ascii=False) + "\n")
                            f.flush()
                        results_buffer.clear()

                except Exception as e:
                    print(f"✗ [线程错误] {filename}: {e}")
                    processed_count += 1

            # 写入剩余的结果
            if results_buffer:
                with file_lock:
                    for res in results_buffer:
                        res_copy = {k: v for k, v in res.items() if k != "success"}
                        f.write(json.dumps(res_copy, ensure_ascii=False) + "\n")
                    f.flush()

    # 统计信息
    total_time = time.time() - start_time
    print("\n" + "=" * 50)
    print(f"处理完成！")
    print(f"总耗时: {total_time/60:.1f} 分钟 ({total_time:.2f}秒)")
    print(f"平均每张: {total_time/len(image_files):.2f}秒")
    print(f"成功: {success_count}/{len(image_files)} 张")
    print(f"失败: {len(image_files)-success_count} 张")
    print(f"输出文件: {output_jsonl}")
    print("=" * 50)


# =====================
# 调试函数
# =====================
def debug_single_image(image_path):
    """详细调试单张图片"""
    print(f"\n{'='*60}")
    print(f"调试图片: {os.path.basename(image_path)}")
    print("=" * 60)

    ocr_instance = create_ocr_instance()
    result = process_single_image(
        image_path, os.path.basename(image_path), ocr_instance
    )

    print(f"\n结果:")
    print(f"  成功: {result['success']}")
    print(f"  文本长度: {len(result['text'])}")
    print(f"  文本内容:\n{result['text'] if result['text'] else '(无文本)'}")

    return result


# =====================
# 主入口
# =====================
if __name__ == "__main__":
    import sys

    data_dir = "/home/z-lq/personal_projects/project_nbfls_bbs/data/"
    image_dir = os.path.join(data_dir, "raw_image")

    if len(sys.argv) > 1:
        if sys.argv[1] == "debug":
            # 调试模式
            image_files = [
                f
                for f in os.listdir(image_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            if image_files:
                first_image = os.path.join(image_dir, image_files[0])
                debug_single_image(first_image)
        elif sys.argv[1] == "test":
            # 测试模式（单线程处理前5张）
            image_files = [
                f
                for f in os.listdir(image_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ][:5]
            print(f"测试模式：处理前 {len(image_files)} 张图片")
            ocr_instance = create_ocr_instance()
            for filename in image_files:
                img_path = os.path.join(image_dir, filename)
                result = process_single_image(img_path, filename, ocr_instance)
                print(
                    f"{'✓' if result['success'] else '✗'} {filename}: {result['text'][:50] if result['text'] else '无文本'}"
                )
        else:
            # 指定线程数
            workers = int(sys.argv[1]) if sys.argv[1].isdigit() else 4
            main_multithread(num_workers=workers)
    else:
        # 默认：4线程
        main_multithread(num_workers=4)
