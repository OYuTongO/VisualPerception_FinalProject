# ASL Gesture Recognition Learning Game — Development Plan

> 视觉感知课程项目 · 2025
> 使用此文档追踪开发进度。每完成一项在 `[ ]` 内打 `x` 变为 `[x]`。

---

## 项目技术栈

| 层级 | 技术 |
|------|------|
| 语言 | Python 3.10+ |
| 视觉 | OpenCV + MediaPipe |
| 模型 | scikit-learn (RandomForest / MLP) |
| 界面 | Pygame |
| 数据集 | [Kaggle ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) |

---

## 核心设计：吉祥物小人

界面中常驻一个 Pygame 绘制的卡通小人，作为主要视觉反馈角色，有三种状态：

| 状态 | 触发条件 | 表现 |
|------|---------|------|
| **IDLE**（等待） | 未检测到手 / 等待用户 | 小人站立，手臂自然下垂，轻微呼吸动画 |
| **CORRECT**（正确） | 识别字母与目标匹配 | 小人举双手欢呼，显示绿色光晕 |
| **WRONG**（错误） | 识别字母与目标不匹配 | 小人摇头摊手，显示红色抖动效果 |

小人用 Pygame 基本图形（圆形、线条）绘制，无需图片资源，三种状态通过不同姿态的坐标实现。

---

## 单词库设计

游戏以**单词**为出题单位（而非单个字母），用户逐字母拼出整个单词。

**单词库规则：**
- 所有单词只含字母 A-Z，**排除 J 和 Z**（动态手势）
- 按难度分级，学习模式用简单词，测试模式混合难度

**示例单词库：**
```python
WORD_LIST = {
    "easy":   ["HI", "BYE", "CAT", "DOG", "SUN", "HAT", "BAG", "CUP", "RUN", "FUN"],
    "medium": ["HELLO", "THANK", "WATER", "APPLE", "BOOKS", "MOUSE", "HAPPY", "DANCE"],
    "hard":   ["FRIEND", "SCHOOL", "FINGER", "PYTHON", "CAMERA", "VISUAL", "SIGNAL"],
}
```

---

## 项目结构（目标）

```
finalProject/
├── PLAN.md                  # 本文件（进度追踪）
├── README.md                # 项目说明
├── requirements.txt         # 依赖列表
│
├── data/                    # 数据相关（不上传至 GitHub）
│   ├── raw/                 # Kaggle 原始图片数据集（.gitignore）
│   └── landmarks.csv        # 提取后的关键点数据（可上传）
│
├── model/                   # 训练产物
│   └── asl_classifier.pkl   # 训练好的模型文件
│
├── src/
│   ├── extract_landmarks.py # 阶段二：从图片提取 MediaPipe 关键点
│   ├── train_model.py       # 阶段三：训练分类模型
│   ├── recognizer.py        # 阶段四：实时识别模块（摄像头 → 字母）
│   │
│   ├── game/
│   │   ├── __init__.py
│   │   ├── main.py          # 阶段五：游戏入口 & 主循环
│   │   ├── menu.py          # 主菜单界面
│   │   ├── mascot.py        # 阶段五：吉祥物小人绘制与状态管理
│   │   ├── word_list.py     # 单词库（含难度分级）
│   │   ├── learn_mode.py    # 阶段六：学习模式
│   │   ├── test_mode.py     # 阶段七：测试模式
│   │   └── freeplay_mode.py # 阶段八：实时识别展示模式
│   │
│   └── utils/
│       ├── draw.py          # Pygame 绘图工具函数
│       └── score.py         # 计分与排行逻辑
│
├── assets/
│   ├── fonts/               # 字体文件
│   ├── images/              # ASL 字母参考图示
│   └── sounds/              # 音效（可选）
│
└── scripts/
    └── download_dataset.py  # 辅助脚本：Kaggle 数据集下载
```

---

## 阶段划分与进度

### Phase 0 — 项目初始化
> 目标：建立可运行的开发环境，所有人都能跑起来

- [x] 创建 GitHub 仓库，推送初始结构
- [x] 编写 `requirements.txt`
- [x] 编写 `README.md`（含环境安装步骤）
- [x] 创建 `.gitignore`（排除 `data/raw/`、`__pycache__/`、`.pkl` 等）
- [x] 验证：`python -c "import cv2, mediapipe, pygame, sklearn; print('OK')"` 通过

**产出物：** 可克隆、可安装依赖的空项目骨架

---

### Phase 1 — 数据集获取
> 目标：下载 Kaggle ASL 数据集到本地 `data/raw/`

- [x] 安装 Kaggle CLI：`pip install kaggle`
- [x] 配置 `kaggle.json` API Token（参考 README）
- [x] 下载数据集：`kaggle datasets download grassknoted/asl-alphabet`
- [x] 解压至 `data/raw/asl_alphabet_train/`
- [x] 确认目录结构：26个字母子文件夹，每文件夹约3000张图

**实际路径：** `data/raw/asl_alphabet_train/asl_alphabet_train/<letter>/`（含多一层同名子目录）

**注：** `data/raw/` 不上传 GitHub，仅 `landmarks.csv` 上传

---

### Phase 2 — 关键点提取
> 目标：用 MediaPipe 从数据集图片提取手部关键点，保存为 CSV

**脚本：** `src/extract_landmarks.py`

- [x] 遍历 `data/raw/` 中每张图片
- [x] 用 MediaPipe Hands 提取 21 个关键点（共 42 个 float：x0,y0,x1,y1,...,x20,y20）
- [x] 归一化坐标（相对于手部边界框）
- [x] 写入 `data/landmarks.csv`，格式：`label, x0, y0, x1, y1, ..., x20, y20`
- [x] 处理无法检测到手的图片（跳过并记录数量）
- [x] 打印统计：各字母成功提取数量

**结果：** 57,339 条样本，24 个字母类别，无缺失值

**注：** MediaPipe 0.10.35 已移除 `mp.solutions` API，改用 Tasks API（需 `model/hand_landmarker.task`）

**关键点说明：** MediaPipe 输出坐标已归一化至 [0,1]（相对图片尺寸），需进一步相对手部边界框归一化以提升平移鲁棒性

---

### Phase 3 — 模型训练
> 目标：训练手势分类模型，准确率 ≥ 90%

**脚本：** `src/train_model.py`
**模型选择：** `RandomForestClassifier`（备选 `MLPClassifier`）

- [x] 读取 `data/landmarks.csv`
- [x] 划分训练集/测试集（80/20）
- [x] 训练 `RandomForestClassifier(n_estimators=100, random_state=42)`
- [x] 在测试集上评估（accuracy、confusion matrix）
- [x] 若准确率 < 90%，切换至 `MLPClassifier` 或调参
- [x] 保存模型至 `model/asl_classifier.pkl`
- [x] 打印各字母 precision/recall（查找弱项字母）

**结果：** 测试集准确率 **98.70%**，最弱字母 N(94.4%) 和 M(95.4%)

**产出物：** `model/asl_classifier.pkl`，测试集准确率报告

---

### Phase 4 — 实时识别模块
> 目标：摄像头输入 → 字母预测，封装为可复用的模块

**脚本：** `src/recognizer.py`

- [x] 封装 `Recognizer` 类，接口如下：
  ```python
  rec = Recognizer(model_path="model/asl_classifier.pkl")
  letter, confidence = rec.predict(frame)  # frame: numpy BGR image
  landmarks = rec.get_landmarks()          # 用于 Pygame 绘制骨架
  ```
- [x] 内部流程：BGR→RGB → MediaPipe → 归一化 → 模型推理
- [x] 无手检测时返回 `(None, 0.0)`
- [x] 置信度低于阈值（如 0.6）时返回 `(None, conf)`
- [x] 独立测试：运行后弹出 OpenCV 窗口显示识别结果（命令行测试用）

**结果：** 检测到手的字母置信度 99-100%，平滑缓冲 5 帧，`HAND_CONNECTIONS` 提供 23 条骨架连接供 Pygame 绘制

---

### Phase 5 — Pygame 主框架 + 吉祥物
> 目标：建立游戏主循环、场景管理、吉祥物系统

**脚本：** `src/game/main.py`、`src/game/mascot.py`、`src/game/word_list.py`

**主框架：**
- [ ] 初始化 Pygame，设定窗口尺寸（建议 1280×720）
- [ ] 实现场景管理器（SceneManager），支持场景切换：
  - `MENU` → `LEARN` / `TEST` / `FREEPLAY`
  - 各场景可返回 `MENU`
- [ ] 主循环：事件处理 → 更新 → 渲染（目标 30 FPS）
- [ ] 摄像头帧在子线程中采集，避免阻塞主循环
- [ ] 资源加载器：字体、参考图片预加载
- [ ] 按 `ESC` 返回上一场景，按 `Q` 退出游戏

**吉祥物小人（`mascot.py`）：**
- [ ] 用 Pygame 基本图形绘制小人（头/身/四肢均为圆形+线条）
- [ ] 实现三种静态姿态坐标：
  - `IDLE`：站立，手臂自然下垂
  - `CORRECT`：双手高举欢呼，绿色光晕
  - `WRONG`：双手摊开，头部轻微倾斜，红色效果
- [ ] 状态切换带过渡动画（线性插值，约 0.3 秒）
- [ ] `Mascot` 类接口：`mascot.set_state("IDLE" | "CORRECT" | "WRONG")`

**单词库（`word_list.py`）：**
- [ ] 定义三档难度单词列表（easy / medium / hard）
- [ ] 所有单词不含 J、Z
- [ ] 提供 `get_word(difficulty)` 随机取词函数

---

### Phase 6 — Diana 对话教学模式
> 目标：通过预设对话脚本引导用户逐字母比划，Diana 提供即时反馈和纠错

**核心设计理念：**
- 目标用户是**零基础非手语使用者**，字母表是最小可学习单元，学习成本远低于完整 ASL 词汇手势
- 手指拼写在真实 ASL 场景中有实际用途（专有名词、新词、与聋哑人建立基本沟通）
- 预设对话脚本解决了手势输入速度慢的问题，用户只需跟着提示比划

**脚本：** `src/game/learn_mode.py`、`src/game/diana.py`

**界面布局：**
- 左侧：摄像头画面（含手部骨架叠加）
- 右侧：Diana 角色（AI 预生成的多表情静态图）
- 右侧上方：Diana 的悬浮气泡聊天窗（显示预设台词）
- 右侧下方：用户输入气泡窗（显示灰色提示词 + 当前拼写进度）

**核心交互流程：**
```
Diana 发出对话提示（如 "Can you spell HELLO?"）
    ↓
用户输入窗显示灰色提示：H E L L O，指针指向第一个字母 H
    ↓
Recognizer 持续推理
    ├── 3秒内置信度 > 阈值 → 字母变绿，指针右移到下一字母
    └── 3秒内置信度全部低于阈值 → 触发纠错流程
                    ├── Diana 切换为对应表情，播放预设纠错台词（如 "Oops! Let me show you~"）
                    ├── 展示该字母的 ASL 标准参考手势图
                    └── 用户重试当前字母
    ↓
整词拼写完成 → Diana 切换为 CORRECT 表情，播放鼓励台词，进入下一轮对话
```

**Diana 角色设计（`diana.py`）：**
- [ ] 使用 AI 预生成 5-6 张不同表情的 Diana 静态图（IDLE / HAPPY / CORRECT / WRONG / THINKING / ENCOURAGING）
- [ ] `Diana` 类接口：`diana.set_state(state)` 切换表情图
- [ ] 气泡聊天窗显示预设台词，支持逐字打出动画效果
- [ ] 接入 TTS（edge-tts / pyttsx3），台词同步语音播报

**预设对话脚本（`src/game/dialogue_script.py`）：**
- [ ] 设计若干轮对话，每轮包含一个目标单词（不含 J、Z）
- [ ] 每轮结构：Diana 开场白 → 用户拼写 → Diana 反馈（正确/纠错）→ 下一轮
- [ ] 纠错台词库：多条随机选取，避免重复
- [ ] 鼓励台词库：拼写正确时随机播报

**拼写进度 UI（`src/utils/draw.py` 扩展）：**
- [ ] 灰色显示待输入字母，绿色显示已确认字母
- [ ] 当前目标字母高亮 + 下方指示箭头
- [ ] 识别失败时目标字母红色闪烁提示

**判断逻辑：**
- 正确：`Recognizer.predict()` 返回目标字母且置信度 ≥ 0.6（3 帧平滑确认）
- 错误：超过 3 秒无任何字母置信度 ≥ 0.6，触发纠错
- 重试不限次数，直到正确为止

---

### Phase 7 — 测试模式
> 目标：随机单词出题，计时计分，小人同步反馈

**脚本：** `src/game/test_mode.py`

- [ ] 随机从 easy+medium 词库取词出题
- [ ] 显示完整目标单词，用户需逐字母拼出
- [ ] 每个字母限时 10 秒，超时算错，自动跳下一字母
- [ ] 答对字母 +10 分，完整拼对单词额外 +20 分
- [ ] 吉祥物实时反映每个字母的判断结果（CORRECT / WRONG / IDLE）
- [ ] 共 5 个单词（可配置），结束后显示结果页：
  - 总分、正确字母数/总字母数、用时
  - 最难的字母（错误次数最多）
- [ ] 本地排行榜（`score.json`）：Top 5 历史最高分

---

### Phase 8 — 实时识别展示模式
> 目标：自由练习 / 演示场景，小人持续跟随状态

**脚本：** `src/game/freeplay_mode.py`

- [ ] 全屏摄像头画面作为背景
- [ ] 右下角显示吉祥物小人（小尺寸），跟随识别结果切换状态
- [ ] 左上角实时显示当前识别字母（大字号）+ 置信度进度条
- [ ] 手部骨架实时叠加在摄像头画面上
- [ ] 识别历史记录（最近 5 个字母横向滚动显示）

---

### Phase 9 — 集成测试与优化
> 目标：整体联调，修复 bug，优化体验

- [ ] 端到端测试：从启动到三个模式全部跑通
- [ ] 实测识别延迟 < 200ms
- [ ] 吉祥物状态切换流畅，无卡顿
- [ ] 检查 J、Z 是否会被误识别，必要时加提示
- [ ] 在不同光线条件下测试识别鲁棒性
- [ ] 内存 / CPU 占用检查（目标：普通笔记本可流畅运行）

---

### Phase 10 — 收尾
> 目标：文档、演示视频、提交

- [ ] 完善 `README.md`（运行方法、截图、已知限制）
- [ ] 录制 Demo 演示视频（≥ 2分钟，覆盖三个模式）
- [ ] 整理代码注释
- [ ] 确认 `.gitignore` 排除所有大文件
- [ ] 最终 push 并打 tag `v1.0`

---

## 当前阶段

> **正在进行：Phase 5 — Pygame 主框架 + Diana 角色系统**
> 上次更新：2026-05-21

---

## 项目定位说明

**目标用户：** 零基础的非手语使用者（想入门 ASL 的普通人）

**为什么选字母表而不是完整词汇手势：**
- 完整 ASL 词汇手势数千个，学习成本对初学者过高
- 字母表只需掌握 24 个静态手型，是最小可学习单元
- 字母拼写（Fingerspelling）在真实 ASL 场景有实际用途：专有名词、新词、与聋哑人建立基本沟通
- 类比：学中文先学拼音，而不是直接背单词

**为什么不用传统手势识别：**
- 目标是教真实 ASL，手势集合由语言学决定，不能自定义
- M/N 手型相似是客观事实，不是设计失误，这正是技术难度所在

---

## 注意事项

- J 和 Z 是**动态手势**，单词库和对话脚本中完全排除
- Diana 角色使用 AI 预生成静态表情图，无需实时生成，便于队友直接运行
- 模型文件 `asl_classifier.pkl` 体积较大，建议 README 中提供下载链接
- 摄像头画面在 Pygame 中显示需转换：`cv2 BGR → RGB → pygame.surfarray`
- 识别结果做**平滑处理**（连续 3 帧一致才确认），避免 Diana 状态频繁抖动
- 错误判定需配合**超时机制**（3秒内无字母超过阈值），避免用户刚举手就被判错
