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
- [ ] 验证：`python -c "import cv2, mediapipe, pygame, sklearn; print('OK')"` 通过

**产出物：** 可克隆、可安装依赖的空项目骨架

---

### Phase 1 — 数据集获取
> 目标：下载 Kaggle ASL 数据集到本地 `data/raw/`

- [ ] 安装 Kaggle CLI：`pip install kaggle`
- [ ] 配置 `kaggle.json` API Token（参考 README）
- [ ] 下载数据集：`kaggle datasets download grassknoted/asl-alphabet`
- [ ] 解压至 `data/raw/asl_alphabet_train/`
- [ ] 确认目录结构：26个字母子文件夹，每文件夹约3000张图

**注：** `data/raw/` 不上传 GitHub，仅 `landmarks.csv` 上传

---

### Phase 2 — 关键点提取
> 目标：用 MediaPipe 从数据集图片提取手部关键点，保存为 CSV

**脚本：** `src/extract_landmarks.py`

- [ ] 遍历 `data/raw/` 中每张图片
- [ ] 用 MediaPipe Hands 提取 21 个关键点（共 42 个 float：x0,y0,x1,y1,...,x20,y20）
- [ ] 归一化坐标（相对于手部边界框）
- [ ] 写入 `data/landmarks.csv`，格式：`label, x0, y0, x1, y1, ..., x20, y20`
- [ ] 处理无法检测到手的图片（跳过并记录数量）
- [ ] 打印统计：各字母成功提取数量

**关键点说明：** MediaPipe 输出坐标已归一化至 [0,1]（相对图片尺寸），需进一步相对手部边界框归一化以提升平移鲁棒性

---

### Phase 3 — 模型训练
> 目标：训练手势分类模型，准确率 ≥ 90%

**脚本：** `src/train_model.py`
**模型选择：** `RandomForestClassifier`（备选 `MLPClassifier`）

- [ ] 读取 `data/landmarks.csv`
- [ ] 划分训练集/测试集（80/20）
- [ ] 训练 `RandomForestClassifier(n_estimators=100, random_state=42)`
- [ ] 在测试集上评估（accuracy、confusion matrix）
- [ ] 若准确率 < 90%，切换至 `MLPClassifier` 或调参
- [ ] 保存模型至 `model/asl_classifier.pkl`
- [ ] 打印各字母 precision/recall（查找弱项字母）

**产出物：** `model/asl_classifier.pkl`，测试集准确率报告

---

### Phase 4 — 实时识别模块
> 目标：摄像头输入 → 字母预测，封装为可复用的模块

**脚本：** `src/recognizer.py`

- [ ] 封装 `Recognizer` 类，接口如下：
  ```python
  rec = Recognizer(model_path="model/asl_classifier.pkl")
  letter, confidence = rec.predict(frame)  # frame: numpy BGR image
  landmarks = rec.get_landmarks()          # 用于 Pygame 绘制骨架
  ```
- [ ] 内部流程：BGR→RGB → MediaPipe → 归一化 → 模型推理
- [ ] 无手检测时返回 `(None, 0.0)`
- [ ] 置信度低于阈值（如 0.6）时返回 `(None, conf)`
- [ ] 独立测试：运行后弹出 OpenCV 窗口显示识别结果（命令行测试用）

---

### Phase 5 — Pygame 主框架
> 目标：建立游戏主循环、场景管理、资源加载

**脚本：** `src/game/main.py`

- [ ] 初始化 Pygame，设定窗口尺寸（建议 1280×720）
- [ ] 实现场景管理器（SceneManager），支持场景切换：
  - `MENU` → `LEARN` / `TEST` / `FREEPLAY`
  - 各场景可返回 `MENU`
- [ ] 主循环：事件处理 → 更新 → 渲染（目标 30 FPS）
- [ ] 摄像头帧在子线程中采集，避免阻塞主循环
- [ ] 资源加载器：字体、参考图片预加载
- [ ] 按 `ESC` 返回上一场景，按 `Q` 退出游戏

---

### Phase 6 — 学习模式
> 目标：A-Z 顺序学习，实时反馈

**脚本：** `src/game/learn_mode.py`

- [ ] 左侧显示当前字母的参考图示（来自 `assets/images/`）
- [ ] 右侧显示摄像头画面（含手部骨架叠加）
- [ ] 屏幕中央显示目标字母（大字号）
- [ ] 识别结果匹配目标字母时：显示 ✅，自动停留 1.5 秒后进入下一个字母
- [ ] 识别结果不匹配时：显示 ❌ + 当前识别到的字母
- [ ] 进度条显示当前字母序号（如 "5 / 26"）
- [ ] 支持按 `→` / `←` 手动切换字母
- [ ] 完成 26 个字母后显示完成界面

---

### Phase 7 — 测试模式
> 目标：随机出题，计时计分

**脚本：** `src/game/test_mode.py`

- [ ] 每题随机显示一个字母（A-Z）
- [ ] 倒计时（每题 10 秒，可配置）
- [ ] 用户比出手势，识别匹配则得分（答对 +10 分）
- [ ] 显示：当前分数、题目编号（共10题）、剩余时间
- [ ] 10题结束后显示结果页：得分、正确率、最难的字母
- [ ] 本地排行榜（保存到 `score.json`）：显示历史最高分 Top 5

---

### Phase 8 — 实时识别展示模式
> 目标：自由练习 / 演示场景

**脚本：** `src/game/freeplay_mode.py`

- [ ] 全屏摄像头画面
- [ ] 实时显示当前识别字母（右上角，大字号）
- [ ] 显示置信度进度条
- [ ] 手部骨架实时叠加在摄像头画面上
- [ ] 识别历史记录（最近5个字母滚动显示）

---

### Phase 9 — 集成测试与优化
> 目标：整体联调，修复 bug，优化体验

- [ ] 端到端测试：从启动到三个模式全部跑通
- [ ] 实测识别延迟 < 200ms
- [ ] 检查 J、Z（动态手势）是否会被误识别，必要时加提示
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

> **正在进行：Phase 1 — 数据集获取**
> 上次更新：2025-05-07

---

## 注意事项

- J 和 Z 是**动态手势**（需要运动轨迹），本项目**暂不支持**，测试模式中排除这两个字母
- 模型文件 `asl_classifier.pkl` 体积较大，建议通过 Git LFS 管理或在 README 中提供下载链接
- 摄像头画面在 Pygame 中显示需转换：`cv2 BGR → RGB → pygame.surfarray`
- 识别结果建议做**平滑处理**（连续N帧一致才确认），避免误触发
