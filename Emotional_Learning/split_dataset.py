"""
将 TapTap 评论数据集划分为训练集、验证集和测试集。
支持 taptap_review_ready.xlsx 或 taptap_review.csv 作为数据源。
"""

import os
import random
from pathlib import Path

import pandas as pd

# 划分比例：训练 70% / 验证 15% / 测试 15%
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

DATA_DIR = Path(__file__).parent
OUTPUT_DIR = DATA_DIR / "data"


def load_dataset() -> pd.DataFrame:
    xlsx_path = DATA_DIR / "taptap_review_ready.xlsx"
    csv_path = DATA_DIR / "taptap_review.csv"

    if xlsx_path.exists():
        df = pd.read_excel(xlsx_path, sheet_name="Sheet1")
        df.columns = ["review", "sentiment"]
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            "未找到 taptap_review_ready.xlsx 或 taptap_review.csv，请将数据集放入项目根目录。"
        )

    df = df[["review", "sentiment"]].copy()
    df["review"] = df["review"].astype(str).str.strip()
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce").astype("Int64")

    before = len(df)
    df = df.dropna(subset=["review", "sentiment"])
    df = df[df["review"].str.len() > 0]
    df["sentiment"] = df["sentiment"].astype(int)

    if len(df) < before:
        print(f"已移除 {before - len(df)} 条无效样本")

    return df.reset_index(drop=True)


def stratified_split(df: pd.DataFrame, train_ratio: float, val_ratio: float, seed: int):
    """按标签分层划分，保证各类别比例一致。"""
    random.seed(seed)
    train_idx, val_idx, test_idx = [], [], []

    for label in sorted(df["sentiment"].unique()):
        indices = df.index[df["sentiment"] == label].tolist()
        random.shuffle(indices)
        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train : n_train + n_val])
        test_idx.extend(indices[n_train + n_val :])

    return (
        df.loc[train_idx].sample(frac=1, random_state=seed).reset_index(drop=True),
        df.loc[val_idx].sample(frac=1, random_state=seed).reset_index(drop=True),
        df.loc[test_idx].sample(frac=1, random_state=seed).reset_index(drop=True),
    )


def print_stats(name: str, split_df: pd.DataFrame):
    total = len(split_df)
    pos = (split_df["sentiment"] == 1).sum()
    neg = (split_df["sentiment"] == 0).sum()
    print(f"  {name}: {total} 条 | 满意(1): {pos} ({pos/total:.1%}) | 不满意(0): {neg} ({neg/total:.1%})")


def main():
    assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6

    df = load_dataset()
    print(f"数据集总量: {len(df)} 条")
    print_stats("整体", df)

    train_df, val_df, test_df = stratified_split(df, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED)

    OUTPUT_DIR.mkdir(exist_ok=True)
    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(OUTPUT_DIR / "val.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False, encoding="utf-8-sig")

    print(f"\n划分完成 (seed={RANDOM_SEED}, 比例 {TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%}):")
    print_stats("训练集", train_df)
    print_stats("验证集", val_df)
    print_stats("测试集", test_df)
    print(f"\n已保存至: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
