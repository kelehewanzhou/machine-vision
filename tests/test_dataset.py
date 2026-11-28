import csv
import time
import tempfile
import argparse
from collections import Counter
from datetime import datetime
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import crop

def get_dataset(name):
    for path in [
        ROOT / f"test{name}",
        ROOT / "data" / f"test{name}"
    ]:
        if path.is_dir():
            return path

    raise FileNotFoundError(f"未找到测试集{name}")


def load_dataset(name):
    dataset = get_dataset(name)
    images = []

    groups = sorted(
        p for p in dataset.glob(f"{name}[0-9][0-9]")
        if p.is_dir()
    )

    for group in groups:
        files = sorted(
            p for p in group.iterdir()
            if p.is_file()
            and p.suffix.lower() in (".jpg", ".jpeg", ".png")
            and p.stem.isdigit()
        )

        print(group.name, "图片数量:", len(files))

        if name == "A":
            print("GT:", (group / "GT.jpg").is_file())

        for p in files:
            images.append((
                group.name,
                p.stem,
                p.relative_to(ROOT).as_posix()
            ))

    print("总组数:", len(groups))
    print("总图片数:", len(images))

    return images

def test_best_original(images):
    cfg = {
        "mode": "best_original",
        "runtime": {"run_id": "dataset_a_original"}
    }

    passed = 0
    failed = []

    for group_id, image_id, path in images:
        r = crop.generate_candidates(
            path, image_id, cfg, group_id=group_id
        )

        if r["status"] != "ok":
            failed.append((path, r["errors"]))
            continue

        candidates = r["data"]["candidates"]
        s = r["data"]["statistics"]

        valid = (
            len(candidates) == 1
            and candidates[0]["crop_id"] == 0
            and candidates[0]["is_original"]
            and candidates[0]["crop_path"] == path
            and s["proposed"] == 1
            and s["filtered"] == 0
            and s["deduplicated"] == 0
            and s["write_failed"] == 0
            and s["returned"] == 1
        )

        if valid:
            passed += 1
        else:
            failed.append((path, "候选结构异常"))

    print("\n===== best_original 测试结果 =====")
    print("测试总数:", len(images))
    print("通过数量:", passed)
    print("失败数量:", len(failed))
    print("通过率:", f"{passed / len(images):.2%}")

    for item in failed:
        print(item)

def test_best_overall(images, name):
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    rows = []
    status_counts = Counter()

    for i, (group_id, image_id, path) in enumerate(images, 1):

        with tempfile.TemporaryDirectory(
            prefix="dataset_a_",
            dir=output_dir
        ) as temp_dir:

            cfg = {
                "mode": "best_overall",
                "runtime": {
                    "run_id": Path(temp_dir).name
                }
            }

            start = time.perf_counter()

            r = crop.generate_candidates(
                path, image_id, cfg, group_id=group_id
            )

            elapsed = time.perf_counter() - start

            status_counts[r["status"]] += 1

            row = {
                "group_id": group_id,
                "image_id": image_id,
                "image_path": path,
                "status": r["status"],
                "elapsed_seconds": round(elapsed, 4),
                "errors": ";".join(
                    e["code"] for e in r["errors"]
                )
            }

            if r["data"] is not None:
                s = r["data"]["statistics"]
                candidates = r["data"]["candidates"]

                row.update(s)

                row["subject_method"] = (
                    candidates[0]["metadata"]["subject_method"]
                )

                row["conservation"] = (
                    s["proposed"]
                    == s["filtered"] + s["deduplicated"]
                    + s["write_failed"] + s["returned"]
                )

                row["files_exist"] = all(
                    (ROOT / c["crop_path"]).is_file()
                    for c in candidates
                )

            rows.append(row)

            print(
                f"[{i}/{len(images)}]",
                group_id,
                image_id,
                r["status"],
                "候选:", row.get("returned"),
                "耗时:", round(elapsed, 3),
                "秒"
            )

    fields = [
        "group_id", "image_id", "image_path",
        "status", "subject_method",
        "proposed", "filtered", "deduplicated",
        "write_failed", "returned",
        "conservation", "files_exist",
        "elapsed_seconds", "errors"
    ]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"dataset_{name.lower()}_overall_{stamp}.csv"

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("\n===== best_overall 测试结果 =====")
    print("测试总数:", len(images))
    print("状态统计:", dict(status_counts))

    for key in [
        "proposed", "filtered", "deduplicated",
        "write_failed", "returned"
    ]:
        print(key, sum(row.get(key, 0) for row in rows))

    print("主体定位方式:", dict(Counter(
        row.get("subject_method", "unavailable")
        for row in rows
    )))

    print("统计守恒异常:", sum(
        row.get("conservation") is False for row in rows
    ))

    print("文件存在性异常:", sum(
        row.get("files_exist") is False for row in rows
    ))

    print("总耗时:", round(sum(
        row["elapsed_seconds"] for row in rows
    ), 3), "秒")

    print("结果已保存:", csv_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["A", "B"])

    args = parser.parse_args()

    images = load_dataset(args.dataset)

    if not images:
        raise ValueError("测试集没有有效图片")

    test_best_original(images)
    test_best_overall(images, args.dataset)