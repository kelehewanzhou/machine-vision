# 智能选片系统接口规范

版本：`1.0`。本文件是后续模块实现与联调的统一契约。所有函数均为待实现接口；本文确定的是规范，不是现有代码的能力说明。与期中接口冲突的部分按第九节迁移。

## 一、通用规则

- 模块间传递可直接 JSON 序列化的 Python `dict` / `list`；禁止传递 ndarray、Tensor、Path、异常对象、NaN 或 Infinity。整数不接受布尔值代替。
- JSON、JSONL、CSV 和 Markdown 统一 UTF-8。JSON 使用 `null`，Python 对应 `None`；缺失数值用 `null`，不得用 0、空字符串或 -1 伪装。
- 下列结构的已声明字段均须返回，明确标为可选的扩展除外。输入缺必填字段属于 `INVALID_INPUT`。结果不得原地修改输入字典或列表。
- `schema_version` 固定为字符串 `"1.0"`。未识别的顶层字段拒绝；自定义信息放入可选的 `metadata` 对象，且不得改变标准字段含义。
- `group_id`、`image_id` 使用非空字符串，区分大小写；`image_id` 在一个数据集内唯一，`group_id` 在一个数据集内唯一。候选主键为三元组 `(group_id, image_id, crop_id)`；比较和排序必须使用完整主键，不能只凭文件名或图片 ID 区分裁剪。
- `crop_id` 为非负整数，在单张图内唯一；原图固定为 `0`，裁剪从 `1` 起。候选按确定顺序编号，相同图像内容和配置下编号稳定。不得使用 ID 拼接后的字符串反向解析主键。
- 配置由入口读取 `configs/default.json`，递归合并显式覆盖项并完整校验；未知键、类型错误或非法范围均拒绝。模块接收同一份合并后的配置，不能暗中修改。模型私有配置放在对应 `backend_options`。

### 路径与图像约定

所有接口和落盘记录中的路径为项目根目录相对路径，使用 `/` 分隔；解析基准固定为 `Intelligent_Photo_Selection/`，不随当前工作目录改变。GUI 导入外部图像时先复制到 `data/raw/`，不得静默重编码。拒绝绝对路径、`..` 及解析后逃出项目根目录的路径（包含符号链接）。原始数据只读；所有产物写入 `outputs/{run_id}/`。`run_id` 为每次运行唯一的安全目录名，由流程入口生成，只允许字母、数字、`_`、`-`。

支持 JPEG/PNG。解码时统一应用 EXIF 方向后转为 RGB；透明 PNG 与白色背景合成。坐标均对应**校正方向后的全分辨率原图**。`image_width`、`image_height` 为该坐标系中的整数尺寸。模型内部缩放不得改变对外坐标；检测框映射回原尺寸后再落盘。

`box=[x1,y1,x2,y2]` 为整数像素、左闭右开坐标，原点在左上角，x 向右、y 向下；约束为 `0 <= x1 < x2 <= image_width`、`0 <= y1 < y2 <= image_height`。原图框为 `[0,0,W,H]`。旧 `(x,y,w,h)` 转换为 `[x,y,x+w,y+h]`，禁止仅重命名字段。候选生成内部可先对浮点框采用左上取 floor、右下取 ceil 并裁到边界，再校验、去重；跨模块收到非法框直接拒绝，不能静默纠正。

原图候选 `crop_path=image_path`，不重复保存；裁剪候选保存校正后 RGB 像素为 PNG，路径为 `outputs/{run_id}/crops/{group_id}/{image_id}/{crop_id}.png`。用于目录的 group/image ID 限定为 `[A-Za-z0-9_-]+`，外部 ID 不符合时须在导入清单建立安全内部 ID 映射，禁止直接拼入路径。供教师提交的外部 ID 保存在数据映射中，不能丢失。

## 二、统一状态与错误对象

所有公共函数使用以下信封，字段齐全：

```python
Result = {
    "schema_version": "1.0",
    "status": "ok",        # ok | partial | error
    "reason": "处理完成",  # 面向使用者的简短说明，非异常堆栈
    "errors": [],          # list[ErrorDetail]，可以包含成功降级后的警告
    "data": {}             # 成功/部分成功为对应数据对象；完全失败为 None
}
ErrorDetail = {
    "code": "MODEL_UNAVAILABLE",
    "module": "score",    # crop | score | vlm_eval | rank | pipeline
    "message": "美学模型不可用，已采用其余维度",
    "retryable": False,
    "context": {"group_id": "scene_001", "image_id": "scene_001_A", "crop_id": 0}
}
```

`context` 是对象，按可定位范围填写 group/image/crop 主键、维度或比较主键；无法定位时为 `{}`。`ok` 表示请求范围内全部完成；`partial` 表示有有效业务结果但发生缺失或降级，`errors` 必须非空；`error` 表示无可用业务结果，`data=null` 且 `errors` 非空。主动关闭 VLM、原图模式主动不生成裁剪，不属于异常。不得仅靠解析 `reason` 判断状态。

### 异常处理边界

模块公共函数校验输入，并将预期的输入、图像、文件及模型故障转换为上述结果；正常业务失败不向调用方抛异常。模块内部可使用异常。未知程序错误应向上抛出，流程入口在每张图/每个候选/每次比较的隔离边界记录完整堆栈，并转换成 `INTERNAL_ERROR` 后继续可继续的项目；不能 `except: pass`。不捕获 `KeyboardInterrupt`、`SystemExit` 等进程控制异常。

| 错误码 | 典型触发条件 | 重试与处理 |
| --- | --- | --- |
| `INVALID_INPUT` | 类型不符、缺字段、非法框、同图自比较、混合场景或非法分数 | 不重试，修正输入 |
| `INVALID_CONFIG` | 权重非法、阈值非法、启用模型但配置缺失 | 不重试，场景启动前拒绝 |
| `DUPLICATE_ID` | 重复图片 ID、候选主键或重复比较对 | 不重试，不静默覆盖 |
| `IMAGE_NOT_FOUND` | 清单引用文件不存在 | 不重试，跳过该图 |
| `IMAGE_DECODE_FAILED` | 损坏或无法解码 | 不重试，跳过该图 |
| `UNSUPPORTED_FORMAT` | 非 JPEG/PNG | 不重试，跳过该图 |
| `INVALID_IMAGE_SIZE` | 宽高无效或超过 `max_image_pixels` | 不重试，跳过该图 |
| `CROP_WRITE_FAILED` | 部分裁剪无法写入 | 不重试，保留成功候选，返回 partial |
| `MODEL_UNAVAILABLE` | 权重缺失、模型初始化失败 | 不重试，评分降级或 VLM 回退 |
| `INFERENCE_FAILED` | 指标或模型推理失败 | 默认不重试；该维度设 null 或该比较失败 |
| `MODEL_TIMEOUT` | 超过单次 VLM 超时 | 可重试，耗尽后回退 |
| `RATE_LIMITED` | VLM 服务限流 | 可重试，耗尽后回退 |
| `SERVICE_UNAVAILABLE` | 网络中断或服务端临时故障 | 可重试，耗尽后回退 |
| `INVALID_MODEL_RESPONSE` | 非法 JSON、缺字段、枚举或置信度越界 | 不重试，拒绝该比较 |
| `NO_VALID_CANDIDATES` | 候选为空或全部无法评分 | 不重试，场景失败 |
| `INSUFFICIENT_IMAGES` | 请求不足两张图，或处理后仅剩一张有效图 | 输入不足两张为 error；处理后只剩一张为 partial |
| `OUTPUT_WRITE_FAILED` | 最终结果无法持久化 | 不重试，流程 error；不声称导出成功 |
| `INTERNAL_ERROR` | 未预期的程序错误 | 不自动重试，由流程边界隔离并记录堆栈 |

VLM 默认每次超时 30 秒，最多额外重试 2 次；等待依次 1、2 秒，仅对 `retryable=true` 重试。模型响应中的任意值不决定本地状态码。日志以 JSONL 写到 `outputs/{run_id}/logs/events.jsonl`，含 UTC ISO 8601 时间、run_id、module、level、主键、code 和 message；错误堆栈保存在本地日志，禁止写入 API 密钥或完整授权头。

## 三、统一数据字段

### 原图记录 ImageRecord

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `group_id` | str | 场景组 ID |
| `image_id` | str | 数据集内唯一的原图 ID，与路径解耦 |
| `image_path` | str | 项目相对原图路径 |

### 候选记录 Candidate

以下字段全部必填；可选的 `metadata` 仅用于额外诊断。

| 字段 | 类型 | 约束 |
| --- | --- | --- |
| `group_id`, `image_id`, `image_path` | str | 与原图记录完全一致 |
| `crop_id` | int | 原图 0，其余正整数 |
| `box` | list[int]，长度4 | 按第一节坐标规则 |
| `image_width`, `image_height` | int | 校正后原图尺寸，均大于0 |
| `crop_path` | str | 实际供评分/VLM 读取的候选像素文件 |
| `is_original` | bool | 当且仅当 crop_id=0 为 true，此时 box 必须为全图 |
| `subject_coverage` | float 或 null | [0,1]，未计算为 null |
| `saliency_retention` | float 或 null | [0,1]，未计算为 null |

候选列表只包含像素可读取且通过校验的候选；失败项写入信封 errors，不生成伪候选。全图裁剪重复框去掉，只保留 crop_id=0。

### 评分记录 ScoredCandidate

在 Candidate 全部字段上增加：

| 字段 | 类型 | 约束 |
| --- | --- | --- |
| `raw_scores` | dict[str, float或null] | 五个维度键均存在，模型/指标原始输出，不同量纲 |
| `sharpness_score` | float 或 null | 清晰度，标准分 [0,100] |
| `exposure_score` | float 或 null | 曝光质量，标准分 [0,100] |
| `color_score` | float 或 null | 色彩质量，标准分 [0,100] |
| `aesthetic_score` | float 或 null | 美学，标准分 [0,100] |
| `composition_score` | float 或 null | 构图，标准分 [0,100] |
| `effective_weights` | dict[str,float] | 五个维度键均存在，实际计算使用的权重 |
| `total_score` | float | 有限数，[0,100]，五维加权综合分 |
| `score_status` | str | ok 或 partial，与本次评分信封状态一致 |
| `normalization_id` | str | 指标归一化配置的版本/指纹 |
| `scoring_profile_id` | str | 算法、预处理、权重和模型版本共同标识 |

维度键固定为 `sharpness`、`exposure`、`color`、`aesthetic`、`composition`，不使用 `aesthetics` 等别名。标准分字段本身就是归一化后的五维分，不再添加一套同义的 `normalized_scores`。

原始指标转换到标准分的具体算法由评分模块实现并记录；归一化配置至少包含每维的算法名、方向、固定校准参数及版本。采用线性边界映射时：高值更好用 `100*clip((x-L)/(U-L),0,1)`，低值更好用其补值，要求 `U>L`；曝光/色彩等非单调质量应先通过已声明的质量函数再映射，不能直接当作越大越好。校准只在开发/验证数据确定并固定，禁止依据当前测试组重新拟合。

五维权重须为有限非负数，总和在 `1±1e-6` 内，且至少一个为正；非法时返回 `INVALID_CONFIG`，不静默归一化错误配置。某维失败时标准分和原始值均为 null，其有效权重为0；有效维度集合 V 内 `effective_weights[d]=w[d]/sum(w[v] for v in V)`，`total_score=sum(effective_weights[d]*score[d])`。可用正权重之和为0时评分 error，不产生 ScoredCandidate。部分缺失产生 partial，仍可排序，缺失不能当作零分。

不同缺失维度的候选允许比较，但最终场景须标 partial 并解释降级。计算和排序保持完整浮点精度，展示时再四舍五入到两位小数。同一场景评分须具有一致的 normalization_id 和 scoring_profile_id；不一致返回 INVALID_INPUT，不能混排。

## 四、函数接口

下面的类型名称是逻辑结构名称；后续 `src/contracts.py` 统一实现 TypedDict 与校验函数。`Result[T]` 表示第二节信封，data 类型为 T。

```python
def generate_candidates(
    image_path: str, image_id: str, config: dict, *, group_id: str
) -> Result[CandidateBatch]: ...

def score_candidate(candidate: Candidate, config: dict) -> Result[ScoredCandidate]: ...

def compare_images(
    image_a: Candidate, image_b: Candidate, prompt: str, config: dict
) -> ComparisonResult: ...

def rank_scene(
    group_id: str, scored_candidates: list[ScoredCandidate], config: dict,
    *, vlm_results: list[ComparisonResult] | None = None
) -> Result[SceneRanking]: ...

def select_scene(
    group_id: str, images: list[ImageRecord], config: dict, *, run_id: str
) -> Result[PipelineOutput]: ...
```

### 4.1 候选生成 generate_candidates

`group_id` 显式传入，避免从文件名或配置推断场景。返回 data：

```json
{
  "candidates": [],
  "statistics": {"proposed": 0, "filtered": 0, "deduplicated": 0, "write_failed": 0, "returned": 0}
}
```

成功时 candidates 至少含一项完整原图。statistics 的 proposed 包含原图及所有初始提案；后续计数分别为被筛除、去重移除、写出失败和返回数量，互斥且总和等于 proposed。不同生成算法可以采用不同提案数量，必须报告实际统计，不能硬写“10～15个”。原图不受裁剪过滤或去重排除。

`best_original` 仅返回原图，统计为 proposed=returned=1，其余0；`best_overall` 生成原图及裁剪。若裁剪部分失败但原图可用，返回 partial；原图不能读取则 error。默认候选参数为比例 1:1、4:5、16:9，线性尺度 0.8、0.9、1.0，位置比例0、0.5、1。每个比例r和尺度s，以最大内接r矩形 `(W0,H0)` 为基准，宽高取 `max(1,floor(s*W0))`、`max(1,floor(s*H0))`，左上取 `floor(p_x*(W-w))`、`floor(p_y*(H-h))`。确定遍历顺序为比例→尺度→横位置→纵位置，原图在首位；尺寸重复候选保留最先出现项，再按 IoU>=0.85 去重（与已保留项比较）。主体/显著性信息未计算时对应保留率为 null，不冒充达到阈值；未启用这些过滤是允许的，具体策略写入运行配置。

### 4.2 图像评分 score_candidate

读取 candidate.crop_path，不再从 image_path 二次裁剪。返回 data 为完整 ScoredCandidate，保留候选身份与所有坐标字段。评分模块拥有五维归一化和 total_score 的计算责任，排序模块不重复计算或覆盖。没有真实模型输出时不得填造美学分；模型不可用按缺失规则处理。

### 4.3 VLM 比较 compare_images

image_a 和 image_b 是同组、不同原图的代表候选；使用 crop_path 作为模型输入。prompt 为非空字符串，可包含用户偏好；系统提示必须要求 JSON 结构且约束只依据图像证据输出。模型及提示词版本记录在比较结果中。返回 data 为 Comparison：

```json
{
  "group_id": "scene_001",
  "image_a": {"group_id": "scene_001", "image_id": "scene_001_A", "crop_id": 0},
  "image_b": {"group_id": "scene_001", "image_id": "scene_001_B", "crop_id": 0},
  "preferred": "a",
  "confidence": 0.86,
  "reason": "A 的主体轮廓更清楚。",
  "model_id": "实际模型及版本",
  "prompt_id": "pairwise_v1",
  "attempts": 1
}
```

preferred 只允许 `a`、`b`、`tie`；confidence 为 [0,1] 有限数，reason 为非空字符串，attempts 为实际尝试次数（1～3）。image_a/image_b 主键由本地代码写入，不信任模型返回的身份。模型只需输出 preferred、confidence、reason 三个业务字段；缺字段、错误类型和越界均拒绝，不能随意 clamp 或把偏好字符串猜成 ID。

`ComparisonResult` 是完整 `Result[Comparison]` 信封，并额外含顶层 `request` 对象，形如 `{"group_id":"scene_001","image_a":{...},"image_b":{...}}`，主键对象格式同上；成功失败均保留 request，用于 data=null 时定位失败比较。该扩展仅用于 compare_images 的返回。失败比较不使用 preferred=tie 伪装成功。VLM 关闭时 pipeline 不调用本函数；直接调用且 enabled=false 返回 INVALID_CONFIG。

### 4.4 综合排序 rank_scene

只对已评分的同组候选排序，不读取图片、不调用模型、不保存文件；职责与 pipeline 分开。`vlm_results=None` 与 `[]` 都表示没有可采纳的比较；是否属于失败由传入的 ComparisonResult 状态判断。输出 SceneRanking：

| 字段 | 类型 | 定义 |
| --- | --- | --- |
| `group_id` | str | 当前组 |
| `mode` | str | best_original / best_overall |
| `selected_image_id` | str | 原图 ID，必须来自输入 |
| `selected_crop_id` | int | 代表候选 ID，原图模式固定0 |
| `selected_box` | list[int] | 对应原图坐标 |
| `selected_crop_path` | str | 所选候选像素路径 |
| `ranking` | list[RankedItem] | 按最终名次排列，每个原图仅一项 |
| `candidates` | list[ScoredCandidate] | 所有传入候选，便于审计；不含失败评分 |
| `comparisons` | list[ComparisonResult] | 所有传入比较，含失败结果 |
| `vlm_applied` | bool | 是否有比较实际改变最终次序 |
| `reason` | str | 结果理由，与信封 reason 保持一致 |

RankedItem 含 `rank`（从1连续整数）、`group_id`、`image_id`、`crop_id`、`box`、`crop_path`、`is_original`、`total_score`、`group_normalized_score`、`score_status`。首项必须与 selected_* 完全一致。

确定性规则：

1. 校验主键唯一、同组、字段合法、分数及 profile 一致；空列表 error。best_original 仅允许 crop_id=0，收到裁剪时返回 INVALID_INPUT，防止误用；best_overall 按原图分组。
2. best_overall 每图选 total_score 最大候选；同分优先原图，其后 crop_id 小者。原图可用且最佳裁剪比原图高分不足3分（严格 `<3`）时保留原图。恰好3分选较高分候选。原图评分失败但裁剪可用时允许代表该图，场景 partial。
3. 代表照片按 total_score 降序、image_id 的 Unicode 字典序升序、crop_id 升序稳定粗排。只在代表照片集合计算 `group_normalized_score=100*(S-min)/(max-min)`；全部同分或仅一张时统一50。该分仅展示，不覆盖 total_score，不参与阈值判断。
4. 默认 Top-K=3。冻结粗排前K项中的**相邻比较计划**：第1/2、第2/3项，且绝对 total_score 差严格 `<5` 才进入计划（恰好5不触发）。pipeline 据此调用 VLM；rank_scene 用同一规则重建计划。不以交换后的相邻关系新增比较。
5. 按冻结计划的原始顺序处理，每对最多一次。只采纳 status=ok 且 confidence 严格 `>0.7` 的 a/b 偏好；若偏好与当前两者位置冲突则交换这两项的当前位置。tie 或 confidence<=0.7 保留次序。交换只发生在原 Top-K 内，其他项保持粗排。完成后将 Top-K 写回完整列表，重算全部 rank。该规则明确保留简化顺序比较策略，不承诺满足全局偏好最优。
6. 比较 request 必须同组、不同图、属于冻结计划且对应代表候选；data 身份须与 request 一致（顺序也一致）。逆序重复对同样视为 DUPLICATE_ID。非计划比较或身份冲突返回 INVALID_INPUT；失败比较保留 errors 并返回 partial，不影响数值顺序。低置信度/tie 的有效响应不属于异常。
7. 只剩一张有效图时仍可推荐但 partial（INSUFFICIENT_IMAGES）；全部失败则 error。候选评分 partial 或原图缺失降级同样向场景汇总。理由只描述真实评分和已采纳的模型证据；不能在 VLM 改序后仍声称首选综合分最高。禁止因某维低于30而未声明地删除候选，可在理由中提示风险。

### 4.5 流程入口 select_scene

负责启动校验、分配配置中的 runtime.run_id、逐图生成、逐候选评分、第一次 rank_scene 得到代表和粗排、构造比较计划、调用 VLM、再 rank_scene，最终导出。rank_scene 不承担这类副作用。输入 images 至少两项且 ID 唯一，记录 group_id 均等于参数。pipeline 将 rank_scene 的“仅剩一张”应用于处理后结果，不能把不足两张的原始请求作为完整场景接受。

返回 PipelineOutput 包含 `run_id`、`scene`（SceneRanking）、`artifacts`（`result_path`、`candidates_path`、`comparisons_path`、`log_path` 四个路径）。产物依次为 `outputs/{run_id}/scenes/{group_id}/result.json`、`candidates.jsonl`、`comparisons.jsonl` 及公共日志路径。JSONL 按每次调用写完整结果信封（包括 error），保留失败主键 context/request；不要只落盘成功 data。

result.json 保存与最终 pipeline 返回一致的信封。先写候选、比较及日志，再在同目录临时文件写最终结果并原子替换；任一必要产物写出失败则返回 error（OUTPUT_WRITE_FAILED），不得返回成功 artifacts。运行失败但可写日志时仍记录错误。一个批次可包含多组，各组分别返回结果；单组失败不停止其他组。上游模块 partial/errors 必须汇总到 pipeline 信封，reason 说明跳过和降级，不能在最终结果“恢复”为 ok。

## 五、配置基线

[configs/default.json](../configs/default.json) 可直接作为配置文件读取，但它尚不能替代模块实现。基线关闭 VLM 和美学模型，避免虚假模型可用性；启用时必须提供实际模型 ID、权重/服务配置和版本。关闭的美学维度按缺失规则产生 partial，不填造分数。

| 配置项 | 默认值/范围 | 含义 |
| --- | --- | --- |
| mode | best_original | 只允许两种已定义模式 |
| score.weights | 0.25/0.20/0.15/0.20/0.20 | 初始权重，非已验证最优值 |
| score.normalization_id | pending | 占位；评分实现后必须改为真实校准标识，否则评分返回 INVALID_CONFIG |
| score.scoring_profile_id | pending | 占位；模块落实后替换并校验，不得实际评分时使用 pending |
| rank.top_k | 3，整数>=2 | 实际取 min(K,代表照片数) |
| rank.score_gap_threshold | 5.0，[0,100] | 严格小于才比较，单位为 total_score 的分 |
| rank.vlm_confidence_threshold | 0.7，[0,1] | 严格大于才采纳 |
| rank.prefer_original_threshold | 3.0，[0,100] | 裁剪优势不足阈值优先原图 |
| vlm.timeout_seconds | 30，正数 | 单次调用超时 |
| vlm.max_retries | 2，整数0～2 | 初次调用以外的重试次数 |
| image.max_image_pixels | 40000000，正整数 | 解码尺寸上限 |

crop.scales 在 `(0,1]`，crop.positions 在 `[0,1]`，aspect_ratios 每项为两个正整数；IoU 阈值在 `(0,1]`。runtime.run_id 在入口注入，模块不得自行新建另一 run_id。配置快照写入日志，屏蔽后端密钥；密钥用环境变量读取，不进入版本库。

## 六、最小联调样例

下面展示一个成功生成的候选数据对象，可作为 score_candidate 的输入；图片路径是样例，不附带真实图像。

```json
{
  "group_id": "scene_001",
  "image_id": "scene_001_A",
  "image_path": "data/raw/scene_001/scene_001_A.jpg",
  "crop_id": 0,
  "box": [0, 0, 1920, 1080],
  "image_width": 1920,
  "image_height": 1080,
  "crop_path": "data/raw/scene_001/scene_001_A.jpg",
  "is_original": true,
  "subject_coverage": null,
  "saliency_retention": null
}
```

评分算法与校准确定后，若五维标准分依次为80、90、70、85、75且无缺失，total_score 应为80.50。若美学失败，其值为null，其他有效权重依次为0.3125、0.25、0.1875、0、0.25，total_score=79.375，评分信封 status=partial。该示例只核验融合计算，不代表模型实测。

```python
# 对接伪代码：模块实现后使用；config 为经过校验的合并配置
generated = generate_candidates(
    record["image_path"], record["image_id"], config, group_id=record["group_id"]
)
scored = []
if generated["data"] is not None:
    for candidate in generated["data"]["candidates"]:
        outcome = score_candidate(candidate, config)
        if outcome["data"] is not None:
            scored.append(outcome["data"])
# 完整 pipeline 还必须保留每次 outcome/errors，收集多张原图并执行 VLM 计划。
```

## 七、目录与文件输出约定

代码布局见 [README](../README.md)。src 不写入源目录或 data；src/contracts.py 是类型、枚举、版本和校验的唯一代码来源。main.py、app.py 和 evaluate.py 均调用 pipeline，不复制一套选片算法。模型放 models/ 并登记名称、版本、权重来源、哈希、许可、预处理、设备及是否微调；具体采用的库和版本在实际验证后写 requirements.txt。

## 八、后续实现验收清单

- 候选：原图0号、EXIF方向、边界及空框、重复框、稳定编号、原图不落重复文件、裁剪写失败和统计守恒。
- 评分：保留全部主键、五维原始值与标准分、全维成功、单维失败权重重分配、全维失败、非有限值拒绝、profile 混用拒绝。
- VLM：a/b/tie、0和1置信度边界、0.7不采纳、非法JSON、缺字段、超时有限重试、失败 request 仍可定位、未启用时不调用。
- 排序：同图多裁剪只占一个照片名次、3分与5分严格边界、稳定同分顺序、组内同分50、VLM改序写回完整ranking、失败比较回退。
- 流程：原始请求不足两图拒绝、单图损坏不拖垮整组、处理后只剩一图 partial、全组失败 error、输出写失败不声称成功、A/B 分离及结果 ID 映射。

以上是后续代码验收要求。本阶段只完成文档及配置的一致性检查，不表示这些算法测试已经执行。

## 九、旧接口迁移

| 期中名称/行为 | v1.0 统一约定 | 迁移注意 |
| --- | --- | --- |
| bbox=(x,y,w,h) | box=[x1,y1,x2,y2] | 必须数值转换；确认旧数据实际含义，不凭字段名猜测 |
| evaluate_image | score_candidate | 输入完整 Candidate，输出信封及 ScoredCandidate |
| compare_images_vlm | compare_images | 输入候选对象；preferred=a/b/tie |
| preferred_image | preferred | 旧 image_a/image_b 标签映射 a/b；旧原图ID必须核实双侧候选身份后转换 |
| rank_and_select | rank_scene | 纯排序函数，流程调用迁入 select_scene |
| recommended_image | selected_image_id | 保证原始图片ID |
| recommended_crop.bbox | selected_box | 同时转换坐标 |
| final_score / ranking[].score | total_score | 不覆盖五维加权综合分 |
| scale=1.0 / position=original | is_original + crop_id=0 | 用全图框校验，不能仅凭scale猜测 |
| total_score_normalized 覆盖 total_score | group_normalized_score 独立保留 | 仅展示，阈值使用total_score |
| 空列表、print警告、任意异常 | 统一Result和错误码 | 保留可追溯信息并由pipeline隔离 |

`../期中/rank.py` 当前仍采用旧行为，不能直接作为新接口模块导入。后续迁移需修正其归一化覆盖综合分、VLM精排写回完整列表及主键映射等问题。本阶段不改写历史代码。公共变更先更新本文件、提高 schema_version、给出迁移表，再同步全部模块；模块私下兼容别名不得进入公共输出。
