# 候选取景框生成与裁剪模块

- 负责成员：成员 3
- 核心文件：`src/crop.py`
- 接口版本：v1.0（参见 [接口规范](interfaces.md)）
- 更新日期：2026-11-28
- 当前状态：模块开发、自测与批量实验已完成，待公共配置确认及系统联调。

## 1. 模块概述

本模块为智能选片系统提供候选取景区域。对输入原图进行解码、主体定位、多尺度及多长宽比候选框生成、约束过滤和去重，并将有效裁剪保存为 PNG，供后续五维评分、视觉语言模型比较及排序模块使用。

本模块只负责候选生成与裁剪，不负责计算综合评分、选择最佳照片或确定最终最佳取景框。

| 运行模式 | 模块行为 |
| --- | --- |
| `best_original`（公共接口默认） | 仅返回完整原图，不执行主体定位或生成裁剪；两项构图保留率为 `None`。 |
| `best_overall` | 返回原图及通过约束过滤、去重和文件写入检查的裁剪候选。 |

## 2. 文件与运行环境

```text
项目根目录/
├── src/
│   └── crop.py
├── models/
│   └── face_detection_yunet_2023mar.onnx
├── tests/
│   ├── test_crop.py
│   └── test_dataset.py
├── data/
│   └── test/
│       ├── test.jpg
│       ├── test_coffee.jpg
│       └── test_rocket.jpg
├── docs/
│   └── crop.md
└── outputs/                     # 运行时生成，不提交裁剪产物
```

开发与测试环境：Windows、Python 3.13.14、OpenCV 5.0.0。模块依赖 NumPy、Pillow 和支持 `FaceDetectorYN` 的 OpenCV；`unittest` 属于 Python 标准库，可视化示例额外使用 Matplotlib。建议使用团队锁定的依赖版本；上述环境为本模块已使用的环境，不代表已验证全部 Python/OpenCV 组合。

独立运行时可在支持上述 API 的 Python 环境中安装基础依赖：

```bash
python -m pip install numpy Pillow opencv-python
```

YuNet ONNX 权重默认放置于 `models/face_detection_yunet_2023mar.onnx`。模型可从 [OpenCV Zoo 的 YuNet 模型目录](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) 获取。请保留模型及测试图片的来源和许可说明；若不随仓库分发，需按约定路径自行准备。

## 3. 算法流程

```text
原图解码（EXIF 方向校正、RGB）
        │
        ├── best_original → 仅返回原图（crop_id=0）
        │
        └── best_overall
                ├── 谱残差显著性图计算
                ├── YuNet 人脸检测 → 有效人脸并集保护框
                │                    无有效人脸 → 显著性主体框
                └── 多比例、多尺度、九位置候选提案
                        ↓
                    精确重复框去除
                        ↓
                    主体覆盖率及显著性保留率过滤
                        ↓
                    IoU 贪心去重
                        ↓
                    PNG 原子写入及统一 Result 返回
```

### 3.1 候选生成

以指定宽高比下的最大内接矩形为基准，默认采用 1:1、4:5、16:9 三种比例，0.8、0.9、1.0 三种线性尺度，以及横纵方向各 0、0.5、1 三种相对位置。宽高和左上坐标向下取整，边长至少为 1 像素。遍历顺序为比例 → 尺度 → 横位置 → 纵位置。

默认初始提案数为 `1 + 3 × 3 × 9 = 82`（包含原图，尚未去重）。原图固定编号为 0；精确重复框保留最先生成项，之后的过滤和去重不重新编号，故返回的裁剪编号可能不连续。

### 3.2 主体定位

- 人脸优先：YuNet 检测到有效人脸时，取所有人脸的最小外接矩形，并按其宽高向左右各扩展 35%、上方扩展 50%、下方扩展 20%，裁到原图边界。检测模型输入最长边默认限制为 1280 像素，输出坐标映射回原图。
- 显著性补偿：无人脸时使用谱残差显著性图，通过阈值分割、形态学闭运算及最大连通区域提取主体框。显著性频域计算的缩放图最长边为 64 像素，随后恢复至原图尺寸。
- 无有效主体框：当前采用保守策略，仅保留原图，返回 `partial`。不将此情况视为整张图片不可用。

### 3.3 约束过滤与去重

主体覆盖率 = 候选框与主体框交集面积 / 主体框面积；显著性保留率 = 候选框内显著性之和 / 全图显著性之和（通过积分图计算）。默认分别要求不低于 0.90 和 0.70；未计算的指标为 `None`，不参与对应约束，不伪造为 0 或 1。

IoU 为两个框交集面积与并集面积之比。按生成顺序将候选框与已保留项比较，IoU 大于或等于 0.85 时去掉当前候选。完整原图始终保留。无主体策略排除的非原图候选计入 `filtered`。

### 3.4 裁剪保存

原图候选直接引用输入路径，不生成重复文件。其余候选使用校正后的全分辨率 RGB 图像切片，以 PNG 无损格式写入同目录临时文件后原子替换，压缩级别为 1。输出路径为：

```text
outputs/{run_id}/crops/{group_id}/{image_id}/{crop_id}.png
```

## 4. 公共接口

```python
def generate_candidates(
    image_path: str,
    image_id: str,
    config: dict,
    *,
    group_id: str
) -> dict:
    ...
```

| 参数 | 说明 |
| --- | --- |
| `image_path` | 项目根目录相对路径，使用 `/`，实际编码为 JPEG 或 PNG。 |
| `image_id` | 原图内部 ID。正式数据集由入口提供全局唯一 ID。 |
| `config` | 公共运行配置，至少提供 `runtime.run_id`；未指定 `mode` 时为 `best_original`。 |
| `group_id` | 显式传入场景 ID，不从文件名推断。 |

路径解析以项目根目录为基准；拒绝绝对路径、`..` 以及解析后超出项目根目录的路径。用于文件目录的 ID 仅允许字母、数字、下划线和连字符。`run_id` 由调用入口生成，建议每次独立运行使用不同值。

返回为 v1.0 `Result[CandidateBatch]`，顶层字段为 `schema_version`、`status`、`reason`、`errors`、`data`。成功或部分成功时，`data` 包含 `candidates` 和 `statistics`；完全失败时为 `None`。

### 4.1 Candidate 字段

| 字段 | 含义 |
| --- | --- |
| `group_id`, `image_id`, `crop_id` | 候选三元主键；原图 `crop_id=0`。 |
| `image_path`, `crop_path` | 原图及实际候选文件的项目相对路径。 |
| `box` | `[x1, y1, x2, y2]`，整数像素坐标，左闭右开。 |
| `image_width`, `image_height` | EXIF 校正后的原图尺寸。 |
| `is_original` | 是否为原图。 |
| `subject_coverage` | 主体覆盖率，[0,1] 或 `None`。 |
| `saliency_retention` | 显著性保留率，[0,1] 或 `None`。 |
| `metadata` | 可选诊断信息，见下。 |

`metadata` 包含 `scale`、`aspect`、`position`、`subject_box`、`subject_method`。其中 `position=[px, py]` 为相对生成位置，原图为 `None`；`subject_method` 为 `face`、`saliency` 或 `none`。所有框均对应校正方向后的原图坐标系，评分模块应直接读取 `crop_path`，不需要再次裁剪。

### 4.2 统计字段

| 字段 | 含义 |
| --- | --- |
| `proposed` | 包含原图的初始提案数。 |
| `filtered` | 被主体保护等策略排除的数量。 |
| `deduplicated` | 精确重复及 IoU 去重移除数量。 |
| `write_failed` | 裁剪保存失败数量。 |
| `returned` | 最终实际返回候选数（包含原图）。 |

统计满足 `proposed = filtered + deduplicated + write_failed + returned`。原图模式中 `proposed=returned=1`，其余为 0。

## 5. 配置参数

以下为当前裁剪模块支持的配置键及默认值，团队最终公共配置文件仍需确认。未列入的内部算法常量不作为公共配置项使用。

| 配置键 | 默认值 | 含义 |
| --- | --- | --- |
| `mode` | `best_original` | 运行模式。 |
| `runtime.run_id` | 由入口提供 | 独立运行目录 ID。 |
| `image.max_image_pixels` | `40000000` | 输入像素数上限。 |
| `crop.aspect_ratios` | `[[1,1],[4,5],[16,9]]` | 宽高比。 |
| `crop.scales` | `[0.8,0.9,1.0]` | 线性缩放尺度。 |
| `crop.positions` | `[0,0.5,1]` | 横纵方向相对位置。 |
| `crop.subject_coverage_min` | `0.90` | 主体覆盖率下限。 |
| `crop.saliency_retention_min` | `0.70` | 显著性保留率下限。 |
| `crop.iou_threshold` | `0.85` | IoU 去重阈值。 |
| `crop.face_model_path` | `models/face_detection_yunet_2023mar.onnx` | 人脸模型路径；设为 `None` 时主动关闭人脸检测。 |
| `crop.face_score_threshold` | `0.80` | 人脸检测置信度下限。 |
| `crop.face_max_side` | `1280` | 人脸模型输入最长边上限。 |

模块读取入口传入的配置字典，不负责独立加载团队配置文件。独立测试可使用模块默认值。

## 6. 状态与降级处理

| 状态 / 错误码 | 触发条件及处理 |
| --- | --- |
| `ok` | 当前请求正常完成。 |
| `error / INVALID_INPUT` | 非法 ID 或输入路径。 |
| `error / INVALID_CONFIG` | 非法模式、阈值、运行 ID 等配置。 |
| `error / IMAGE_NOT_FOUND` | 原图文件不存在。 |
| `error / UNSUPPORTED_FORMAT` | 实际编码不是 JPEG/PNG，包括 MPO。 |
| `error / INVALID_IMAGE_SIZE` | 图片尺寸非法或超过像素上限。 |
| `error / IMAGE_DECODE_FAILED` | 图片损坏或无法解码。 |
| `partial / MODEL_UNAVAILABLE` | YuNet 权重缺失或初始化失败，尝试显著性定位。 |
| `partial / INFERENCE_FAILED` | 人脸或显著性计算发生已处理的推理异常，或未定位到主体；保留可用结果。 |
| `partial / CROP_WRITE_FAILED` | 部分裁剪保存失败，排除对应候选并保留成功结果及原图。 |

纯色图等无有效主体时返回 `partial`，只保留原图，是当前模块采用的策略；若团队调整该策略，应同步配置、文档和测试。未知程序异常由上层流程隔离，不能视为正常成功。

## 7. 使用与测试

以下命令均从项目根目录执行。先准备回归测试图片及模型权重。

### 7.1 Python 调用示例

```python
from src.crop import generate_candidates
from uuid import uuid4

config = {
    "mode": "best_overall",
    "runtime": {"run_id": f"demo_{uuid4().hex}"}
}

result = generate_candidates(
    "data/test/test.jpg",
    "img01",
    config,
    group_id="demo"
)

print(result["status"])
if result["data"] is not None:
    print(result["data"]["statistics"])
    for candidate in result["data"]["candidates"]:
        print(candidate["crop_id"], candidate["crop_path"])
print(result["errors"])
```

### 7.2 独立命令行调试

```bash
python src/crop.py data/test/test.jpg img01 demo debug_crop_01
```

该调试入口使用 `best_overall`，输出 JSON；公共接口默认模式仍为 `best_original`。请为不同实验指定独立的运行 ID。

### 7.3 自动化测试及批量实验

```bash
python -m unittest discover -s tests -p test_crop.py -v
python tests/test_dataset.py A
python tests/test_dataset.py B
```

自动化测试覆盖原图模式、正常裁剪、无主体策略、参数及路径校验、模型缺失、推理与写入异常、EXIF 方向校正、透明 PNG、损坏图片、极端尺寸、像素一致性、编号稳定性及 IoU 约束等，共 20 项。

批量脚本兼容根目录 `testA/`、`testB/` 和 `data/testA/`、`data/testB/` 两种实际目录布局，分组分别命名为 `A01`～`A20` 和 `B01`～`B05`，图片使用纯数字文件名及 JPEG/PNG 扩展名，排除 `GT.jpg`。注意：该独立测试脚本使用组内文件编号配合 `group_id` 区分照片，不替代正式入口要求的数据集内全局唯一 `image_id`。

批量实验在临时运行目录中保存并检查候选文件，随后自动清理临时图片，只保留 `outputs/dataset_a_overall_*.csv` 或 `outputs/dataset_b_overall_*.csv` 等统计结果。CSV 记录状态、主体定位方法、候选统计、文件存在性、守恒检查及每张图处理耗时。

## 8. 实验结果

测试集 A 为教师提供的 20 组、99 张待评选图片（不含 GT）；测试集 B 为自行收集的 5 组、20 张生活照片，涵盖人像、风景、晚霞及两类餐食。B 中 14 张 MPO 图片在实验前提取主图像、应用 EXIF 方向校正并转换为无损 PNG，保留原始分辨率；原文件另行备份，6 张普通 JPEG 不变。

| 指标 | 测试集 A | 测试集 B（优化后） |
| --- | ---: | ---: |
| `best_original` 通过数量 | 99/99 | 20/20 |
| `best_overall` 正常运行数量 | 99/99 | 20/20 |
| 初始提案 | 8118 | 1640 |
| 约束过滤 | 5246 | 1029 |
| 重复框移除 | 2147 | 450 |
| 最终候选（含原图） | 725 | 161 |
| 候选压缩率 | 91.07% | 90.18% |
| 平均每图候选数 | 7.32 | 8.05 |
| 人脸 / 显著性定位张数 | 11 / 88 | 4 / 16 |
| 写入失败、守恒异常、文件存在性异常 | 均为 0 | 均为 0 |

候选压缩率按 `1 - returned/proposed` 计算。A 的统计取自 PNG 压缩优化前的批量实验，B 采用优化后的复测数据。自动化测试在优化后再次全部通过。这里的成功率是模块运行及接口检查通过率，不代表主体定位准确率或最终选片准确率。

### 高分辨率性能优化

使用 `cProfile` 对 B01/01.jpg 分析，主要开销集中在 PNG 编码及文件保存。将 PNG 压缩级别由默认 6 调整为 1，保持无损编码方式及裁剪分辨率不变。

| B 集 20 张图 | 优化前 | 优化后 |
| --- | ---: | ---: |
| 累计处理耗时 | 329.467 s | 152.248 s |
| 平均每张耗时 | 16.473 s | 7.612 s |
| 最终候选数量 | 161 | 161 |

累计耗时降低约 53.79%，加速约 2.16 倍。两轮 CSV 除耗时外的各项记录逐张一致；优化后 20 项自动化测试全部通过，包括裁剪像素一致性测试。耗时包含解码、检测、候选生成及 PNG 写入，为单轮实测，尚未进行多轮统计，也未量化输出文件体积增幅。

实验原始记录（本地保存，供报告及复核）：

- `dataset_a_overall_20261128_001958.csv`
- `dataset_b_overall_20261128_122130.csv`（优化前）
- `dataset_b_overall_20261128_123900.csv`（优化后）

## 9. 局限性与联调说明

- 人脸框保护不等同于全身保护，仍可能出现身体局部截断；侧脸、遮挡或低分辨率可能导致漏检。显著性定位也可能受到背景或多个主体干扰。测试集 A 的 A17/09、A17/06 是典型观察案例。
- 目前验证的是候选生成和筛选效率，最终裁剪的审美质量及整套系统选片结果由后续评分和排序实验验证。
- 降低 PNG 压缩级别可减少编码耗时，但文件体积可能增加，高分辨率照片仍有一定处理开销。
- 联调时应确认公共配置键名、全局原图 ID 映射、无主体处理策略和 `best_original` 下两项构图指标均为 `None` 的缺失维度处理；评分模块直接使用 `crop_path`，保留完整候选三元主键。
- 公共 `contracts.py` 校验与 `pipeline.select_scene`、GUI 正式后端接入由团队统一完成，本模块的独立测试不等同于系统联调通过。
- 提交时仅包含代码、测试脚本、本文档及允许分发的必要测试资源和模型（附来源及许可）。不提交个人照片、原始备份、运行输出、缓存或旧版调试文件；实验数据及图表按团队约定单独归档。
