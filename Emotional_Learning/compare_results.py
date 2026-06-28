"""对比 baseline / balanced / optimized 三次实验的测试集结果。"""

import json
from pathlib import Path


def load_result(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    project_dir = Path(__file__).parent
    experiments = [
        ("baseline", project_dir / "checkpoints" / "test_result.json"),
        ("balanced", project_dir / "checkpoints_balanced" / "test_result.json"),
        ("optimized", project_dir / "checkpoints_optimized" / "test_result.json"),
    ]

    print(f"{'实验':<12} {'Acc':>8} {'F1':>8} {'Loss':>8}")
    print("-" * 40)
    for name, path in experiments:
        result = load_result(path)
        if result is None:
            print(f"{name:<12} {'(未找到结果)':>24}")
            continue
        print(
            f"{name:<12} "
            f"{result['test_acc']:>8.4f} "
            f"{result['test_f1']:>8.4f} "
            f"{result['test_loss']:>8.4f}"
        )


if __name__ == "__main__":
    main()
