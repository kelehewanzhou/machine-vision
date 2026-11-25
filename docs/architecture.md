# 系统架构与模块边界

本项目采用五个内部模块，CLI、GUI 和实验入口共享同一流程。统一接口以 [interfaces.md](interfaces.md) 为准。本文件描述待实现架构。

```text
main.py / app.py / experiments/evaluate.py
                    |
             pipeline.select_scene
                    |
    crop.generate_candidates → score.score_candidate
                    |
             rank.rank_scene（粗排）
                    |
    冻结 Top-K 相邻比较计划 → vlm_eval.compare_images
                    |
             rank.rank_scene（最终排序）
                    |
           pipeline 汇总状态并持久化
```

crop 负责图像解码方向、候选坐标、裁剪文件与统计；score 负责五维指标、固定校准、缺失维度权重重分配及综合分；vlm_eval 负责提示词、模型调用、有限重试与响应校验；rank 负责模式筛选、每图代表、稳定排序和偏好融合；pipeline 负责隔离故障、汇总状态和结果导出。公共类型与校验集中于 contracts。

rank_scene 是纯计算接口，不能调用 crop、score 或 VLM。这样可以在不安装模型的环境中使用固定评分记录验证排序，并确保 GUI 和实验脚本采用相同决策逻辑。第一轮排序用于确定代表候选和冻结比较计划，第二轮使用同一组评分及已取得的比较结果；整个过程不重复评分。

best_original 保持原图选择口径。best_overall 先为每张照片确定一个代表画面，再在原图之间比较，避免一张照片因裁剪候选多而占满 Top-K。两种模式分别评估，不把裁剪质量等同于完整原图质量。

单张图无法解码时跳过该图；单个指标失败时以 null 标明并重分配其余权重；VLM 失败时保留数值排序。只要有有效结果即可 partial 返回，完全无有效结果才 error。输出持久化失败时流程返回 error。所有部分成功和失败信息保留到最终输出及日志。

本阶段不选择具体模型后端或 GUI 库。下一阶段由负责成员落实模型、归一化算法及环境版本，在接口不变的前提下实现模块，补充真实运行和联调记录。
