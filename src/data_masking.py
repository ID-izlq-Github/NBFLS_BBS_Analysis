from pathlib import Path
import pandas as pd
import re
from pypinyin import pinyin, Style

BASE_DIR = Path(__file__).resolve().parent.parent

original_data: list = [
    BASE_DIR / "notebook" / "input" / "extracted_posts_v1.csv",
    BASE_DIR / "notebook" / "input" / "extracted_posts_v2.csv",
    BASE_DIR / "data" / "feature_data" / "extracted_posts_v2.csv",
]


def hide_raw_information(df: pd.DataFrame):
    try:
        df.drop(columns=["author_name", "target_name", "post_text"], inplace=True)
    except KeyError as e:
        print(f"文件损坏或无需处理{e}")
        exit()


def to_name_initial(name: str) -> str:
    eng_pattern = re.compile("\\w+", re.ASCII)
    if eng_pattern.fullmatch(name):
        return name
    else:
        initials = pinyin(name, style=Style.FIRST_LETTER)
        return "".join([item[0] for item in initials])


for file_path in original_data:
    if file_path.exists():
        file = pd.read_csv(file_path)

        mask = file["target_name"].notnull()
        file.loc[mask, "target_name_initial"] = file.loc[mask, "target_name"].map(to_name_initial)

        hide_raw_information(file)

        new_filepath = file_path.parent / f"{file_path.stem}_public{file_path.suffix}"
        file.to_csv(new_filepath, index=False)

    else:
        print(f"文档不存在: {file_path}")
