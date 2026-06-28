"""
Bi-LSTM 中文评论情感分类训练脚本。

默认使用 conda 环境 test4 (Python 3.12.9):
    conda activate test4
    python train.py
    python train.py --profile balanced    # 推荐：折中配置（平衡容量与正则）
    python train.py --profile optimized   # 强正则，缓解过拟合
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ReviewDataset, Vocab, collate_batch, load_csv, save_data_config
from model import BiLSTMSentimentClassifier


PROFILE_PRESETS = {
    "baseline": {},
    "balanced": {
        # 在 baseline 与 optimized 之间折中：
        # 保留较长序列与足够模型容量，同时引入适度正则与 attention 池化
        "output_dir": "checkpoints_balanced",
        "max_len": 256,
        "min_freq": 2,
        "embed_dim": 128,
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.5,
        "embed_dropout": 0.2,
        "pooling": "attention",
        "batch_size": 64,
        "epochs": 25,
        "lr": 7e-4,
        "weight_decay": 5e-4,
        "label_smoothing": 0.05,
        "patience": 4,
        "lr_scheduler": "plateau",
        "lr_factor": 0.5,
        "lr_patience": 2,
    },
    "optimized": {
        "output_dir": "checkpoints_optimized",
        "max_len": 128,
        "min_freq": 3,
        "embed_dim": 128,
        "hidden_dim": 96,
        "num_layers": 1,
        "dropout": 0.6,
        "embed_dropout": 0.3,
        "pooling": "attention",
        "batch_size": 64,
        "epochs": 30,
        "lr": 5e-4,
        "weight_decay": 1e-3,
        "label_smoothing": 0.1,
        "patience": 4,
        "lr_scheduler": "plateau",
        "lr_factor": 0.5,
        "lr_patience": 2,
    },
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(model, loader, criterion, optimizer, device, train: bool, scheduler=None):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for input_ids, lengths, labels in tqdm(loader, leave=False):
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1, all_labels, all_preds


def apply_profile(args: argparse.Namespace) -> argparse.Namespace:
    if args.profile == "custom":
        return args
    preset = PROFILE_PRESETS.get(args.profile, {})
    for key, value in preset.items():
        setattr(args, key, value)
    return args


def parse_args():
    parser = argparse.ArgumentParser(description="Bi-LSTM 中文评论情感分类")
    parser.add_argument(
        "--profile",
        type=str,
        default="baseline",
        choices=["baseline", "balanced", "optimized", "custom"],
        help="baseline=原始配置, balanced=折中配置, optimized=强正则配置, custom=完全手动指定",
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-vocab", type=int, default=50000)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--embed-dropout", type=float, default=0.0)
    parser.add_argument(
        "--pooling",
        type=str,
        default="last",
        choices=["last", "attention"],
        help="句子表示: last=最后隐状态, attention=注意力池化",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--lr-scheduler", type=str, default="none", choices=["none", "plateau"])
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5, help="早停耐心值")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    return apply_profile(args)


def main():
    args = parse_args()
    set_seed(args.seed)

    project_dir = Path(__file__).parent
    data_dir = project_dir / args.data_dir
    output_dir = project_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"使用设备: {device}")
    print(f"训练配置 profile={args.profile}")

    train_df = load_csv(data_dir / "train.csv")
    val_df = load_csv(data_dir / "val.csv")
    test_df = load_csv(data_dir / "test.csv")
    print(f"训练集: {len(train_df)} | 验证集: {len(val_df)} | 测试集: {len(test_df)}")

    vocab = Vocab()
    vocab.build(train_df["review"].tolist(), min_freq=args.min_freq, max_size=args.max_vocab)
    print(f"词表大小: {len(vocab)}")

    train_loader = DataLoader(
        ReviewDataset(train_df, vocab, args.max_len),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        ReviewDataset(val_df, vocab, args.max_len),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        ReviewDataset(test_df, vocab, args.max_len),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
    )

    model = BiLSTMSentimentClassifier(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        embed_dropout=args.embed_dropout,
        pooling=args.pooling,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.lr_factor,
            patience=args.lr_patience,
        )

    best_val_f1 = 0.0
    patience_counter = 0
    history = []

    print("\n开始训练...")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc, train_f1, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc, val_f1, _, _ = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        elapsed = time.time() - start

        if scheduler is not None:
            scheduler.step(val_f1)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f}, acc {train_acc:.4f}, f1 {train_f1:.4f} | "
            f"val loss {val_loss:.4f}, acc {val_acc:.4f}, f1 {val_f1:.4f} | "
            f"lr {optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "vocab_size": len(vocab),
                },
                output_dir / "best_model.pt",
            )
            print(f"  -> 保存最佳模型 (val_f1={val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"验证集 F1 连续 {args.patience} 轮未提升，提前停止训练。")
                break

    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, test_f1, test_labels, test_preds = run_epoch(
        model, test_loader, criterion, optimizer, device, train=False
    )
    print("\n测试集结果:")
    print(f"  loss: {test_loss:.4f}")
    print(f"  acc : {test_acc:.4f}")
    print(f"  f1  : {test_f1:.4f}")
    print("\n分类报告:")
    print(classification_report(test_labels, test_preds, target_names=["不满意(0)", "满意(1)"]))

    vocab.save(output_dir / "vocab.pkl")
    save_data_config(output_dir / "config.json", vars(args))
    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    test_result = {
        "profile": args.profile,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "classification_report": classification_report(
            test_labels,
            test_preds,
            target_names=["不满意(0)", "满意(1)"],
            output_dict=True,
        ),
    }
    with open(output_dir / "test_result.json", "w", encoding="utf-8") as f:
        json.dump(test_result, f, ensure_ascii=False, indent=2)

    print(f"\n模型与词表已保存至: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
