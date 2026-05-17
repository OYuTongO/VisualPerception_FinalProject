# ASL Project Marp Presentation - Source Code
# Instructions: 
# 1. Copy this entire block.
# 2. Paste into a new file.
# 3. Save as "presentation.md".
# 4. Open in VS Code with Marp extension.

---
marp: true
theme: default
paginate: true
backgroundColor: #0d1117
color: #c9d1d9
style: |
  section {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    font-size: 24px;
  }
  h1 {
    color: #58a6ff;
    font-size: 50px;
  }
  h2 {
    color: #58a6ff;
    border-bottom: 2px solid #30363d;
    padding-bottom: 10px;
  }
  h3 {
    color: #58a6ff;
  }
  code {
    background-color: #161b22;
    color: #ff7b72;
  }
  table {
    width: 100%;
    border-collapse: collapse;
  }
  th {
    background-color: #161b22;
    color: #58a6ff;
  }
  td {
    border-bottom: 1px solid #30363d;
  }
  .blue { color: #58a6ff; }
  .green { color: #3fb950; }
  .red { color: #f78166; }
  .card-container {
    display: flex;
    justify-content: space-between;
    gap: 20px;
  }
  .card {
    background: #161b22;
    padding: 15px;
    border-radius: 10px;
    flex: 1;
    border-left: 5px solid #58a6ff;
  }
  .stat-card {
    text-align: center;
    background: linear-gradient(145deg, #161b22, #0d1117);
    padding: 10px;
    border-radius: 8px;
    border: 1px solid #30363d;
  }
  .stat-value {
    font-size: 32px;
    font-weight: bold;
    display: block;
  }
  .grid-3 {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 15px;
  }
  .footer-tags {
    font-size: 18px;
    color: #8b949e;
    text-align: center;
    margin-top: 40px;
  }
---

<!-- _class: center -->

# ASL Gesture Recognition Learning Game
### 基于计算机视觉的美式手语字母实时识别与互动学习游戏

<div class="footer-tags">
`视觉感知课程` · `华东师范大学` · `2025` · `期中 Proposal`
</div>

<br>

<div style="text-align: center; font-size: 20px; color: #8b949e;">
`OpenCV` · `MediaPipe` · `RandomForest` · `Pygame` · `LanXin Eagle Pro`
</div>

---

## 为什么做这个项目？

<div class="card-container" style="margin-bottom: 20px;">
  <div class="stat-card"><span class="stat-value blue">7,200 万+</span>全球聋哑人口</div>
  <div class="stat-card"><span class="stat-value green">26 个</span>ASL 字母手势</div>
  <div class="stat-card"><span class="stat-value red">无实时反馈</span>传统 App 痛点</div>
</div>

<div class="card-container">
  <div class="card" style="border-color: #f78166;">
    <b class="red">现有问题</b>
    <ul>
      <li>学习门槛高，依赖线下教师</li>
      <li>现有 App 无交互闭环</li>
      <li>缺乏即时、可量化的质量反馈</li>
    </ul>
  </div>
  <div class="card" style="border-color: #3fb950;">
    <b class="green">我们的切入点</b>
    <ul>
      <li>实时识别，延迟 < 200ms</li>
      <li>游戏化单词拼写，构建闭环</li>
      <li>工业级相机 + 纯 Python 实现</li>
    </ul>
  </div>
</div>

---

## 技术选型：为什么是这套组合？

**端到端流程：**
`相机` $\rightarrow$ `OpenCV(子线程)` $\rightarrow$ `MediaPipe(21点)` $\rightarrow$ `42维特征` $\rightarrow$ `RandomForest` $\rightarrow$ `3帧平滑` $\rightarrow$ `Pygame`

| 层级 | 选用方案 | 备选方案 | 选用理由 |
| :--- | :--- | :--- | :--- |
| **相机** | **LanXin Eagle Pro** | 内置摄像头 | 高帧率、低畸变、强光适应 |
| **检测** | **MediaPipe (Tasks)** | YOLOv8-pose | 轻量化、CPU实时、精度高 |
| **特征** | **42-dim 归一化坐标** | 原始 CNN 特征 | 平移/尺度不变、速度极快 |
| **分类** | **RandomForest** | MLP / SVM | **98.7% 准确率**、推理 < 1ms |
| **界面** | **Pygame** | Tkinter / Qt | 帧级渲染控制、精确计时 |

---

## 视觉管线：从像素到字母

<div class="grid-3">
<div class="card">
<b class="blue">1. API 升级</b><br>
适配 MediaPipe 0.10.35+ <br>
使用 <code>Tasks API</code> 与 <code>.task</code> 模型<br>
确保项目长期可维护性
</div>
<div class="card">
<b class="blue">2. 二次归一化</b><br>
消除手部在画面中位置/距离影响<br>
<code>x_norm = (x - x_min) / (x_max - x_min)</code><br>
实现平移与尺度不变性
</div>
<div class="card">
<b class="blue">3. 状态平滑</b><br>
使用 <code>deque(maxlen=3)</code><br>
连续 3 帧一致才触发输出<br>
解决吉祥物状态抖动问题
</div>
</div>

---

## 数据管线 + 模型训练结果

<div class="card-container" style="margin-bottom: 15px;">
  <div class="stat-card"><span class="stat-value blue">57,339</span>有效样本</div>
  <div class="stat-card"><span class="stat-value blue">24</span>分类类别</div>
  <div class="stat-card"><span class="stat-value green">98.70%</span>测试准确率</div>
</div>

| 模型 | 测试集准确率 | 结论 |
| :--- | :--- | :--- |
| **RandomForest (n=100)** | **98.70%** | ✅ **采用方案** |
| MLPClassifier | ~97.00% | 备选方案 |

<b class="red">弱项分析：</b> 字母 **N** (94.4%) 与 **M** (95.4%) 易混淆（指纹折叠数差异）。
<br>$\rightarrow$ *策略：规避高频 MN 连续组合；后期进行针对性数据增强。*

---

## 模块化架构：从采集到游戏

**三层逻辑架构：**

1. **输入层 (Input)**: `LanXin Eagle Pro` $\rightarrow$ `OpenCV Sub-thread` (解耦 I/O)
2. **视觉核心层 (Vision Core)**: 
   - `MediaPipe` 检测 $\rightarrow$ `Normalization` $\rightarrow$ `RandomForest` 推理 $\rightarrow$ `3-Frame Buffer`
3. **游戏层 (Game Layer)**: 
   - `SceneManager` 驱动 `Learn/Test/Freeplay` 模式
   - `Mascot` 情感化交互 & `Scoring` 系统

- **解耦设计**：`Recognizer` 仅通过 `predict()` 接口与游戏层通信。
- **性能优化**：摄像头采集放在独立子线程，确保 Pygame 渲染维持在 30 FPS。

---

## 游戏设计：三种模式构建学习闭环

<div class="grid-3">
<div class="card" style="border-color: #3fb950;">
<b class="green">📖 学习模式</b><br>
<small>零基础引导</small>
<hr>
- 目标单词高亮显示<br>
- 提供 ASL 参考图示<br>
- 吉祥物实时反馈
</div>
<div class="card" style="border-color: #f78166;">
<b class="red">⏱ 测试模式</b><br>
<small>成果检验</small>
<hr>
- 10s 限时挑战<br>
- 计分机制 (+10/+20)<br>
- 生成弱点统计报告
</div>
<div class="card" style="border-color: #58a6ff;">
<b class="blue">🕹 自由模式</b><br>
<small>展示/练习</small>
<hr>
- 全屏骨架叠加<br>
- 置信度实时进度条<br>
- 最近 5 词历史滚动
</div>
</div>

---

## 细节决定体验：吉祥物 × 工业相机

<div class="card-container">
<div style="flex: 1;">
<b class="blue">亮点一：吉祥物 (Mascot)</b>
<ul style="font-size: 18px;">
<li><b>IDLE:</b> 呼吸动画 (sin 抖动)</li>
<li><b>CORRECT:</b> 欢呼 + 绿色光晕</li>
<li><b>WRONG:</b> 摇头 + 红色抖动</li>
<li>通过 <b>Lerp (线性插值)</b> 实现平滑姿态过渡</li>
</ul>
</div>
<div style="flex: 1;">
<b class="blue">亮点二：工业相机升级</b>
<table style="font-size: 16px;">
<tr><th>指标</th><th>内置</th><th>Eagle Pro</th></tr>
<tr><td>帧率</td><td>30</td><td><b>≥ 60</b></td></tr>
<tr><td>光照</td><td>弱光噪点</td><td><b>低噪/大光圈</b></td></tr>
<tr><td>畸变</td><td>较大</td><td><b>低畸变</b></td></tr>
</table>
</div>
</div>

---

## 当前进度：CV 核心已完成，游戏层构建中

| Phase | 阶段名称 | 状态 | 关键产出物 |
| :--- | :--- | :--- | :--- |
| 0-2 | 数据与特征提取 | ✅ | 57,339 条关键点数据 |
| 3-4 | 模型训练与识别模块 | ✅ | **98.70% 准确率模型** |
| **5** | **Pygame 框架与吉祥物** | 🔨 | **进行中 (SceneManager)** |
| 6-8 | 三大模式开发 | 📋 | 待实现 |
| 9-10 | 集成测试与交付 | 📋 | 待实现 |

<div class="card" style="margin-top: 10px; border-color: #58a6ff;">
<b>已完成里程碑：</b> CV 管线全通，模型精度远超 90% 基线，Recognizer 接口稳定。
</div>

---

## 难点复盘 × 后续深化方向

<div class="card-container">
<div class="card" style="flex: 0.8; border-color: #f78166;">
<b class="red">已解决难点</b>
- MediaPipe API 迁移 (Tasks API)
- 关键点坐标二次归一化
- 交互状态机 (3帧平滑)
</div>
<div class="card" style="flex: 1.2; border-color: #58a6ff;">
<b class="blue">后续深化 (Proposal)</b>
1. **动态手势 (J/Z)**: 基于高帧率轨迹建模 (LSTM)
2. **手势质量评分**: 计算与标准模板的余弦相似度
3. **自适应学习**: 基于错误率的热力图加权出题
4. **增量学习**: 引入用户校准环节，适配个体差异
</div>
</div>

<br>

<div style="text-align: center; font-size: 18px;">
<b>目标：</b> 打造一个延迟 < 200ms、具备情感化反馈的完整手语学习系统。
</div>