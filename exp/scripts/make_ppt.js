#!/usr/bin/env node
/**
 * 生成课程作业汇报 PPT（题目三）。
 *
 * 设计取舍：
 * - 面向课程评审：只讲作业内容（问题、方法、实验、结果、讨论），
 *   不含汇报组织安排（谁讲哪页、几分钟、交接语）这类与内容无关的信息。
 * - 用 pptxgenjs 的 structured deck（theme + layout + section），配色改一处生效，
 *   文字位置由 layout 的 placeholder 统一，避免逐页手调导致标题漂移。
 * - 数据全部来自 exp/runs/ 的 json 与 交付/报告/实验报告.md，不在这里另算。
 * - 图表用 PowerPoint 原生 chart（不用截图），便于现场改数据。
 *
 * 用法：cd exp && node scripts/make_ppt.js
 */
const path = require('path');
const fs = require('fs');
const os = require('os');
const PptxGenJS = require('pptxgenjs');

const SKILL = path.join(os.homedir(), '.agents/skills/pptx/scripts/apply_theme.js');
const { applyTheme } = require(SKILL);

// 输出到 exp/slides/：与 figures/ 一致，exp/ 是唯一源头，
// 交付/ppt/ 由 make_delivery.py 复制生成（交付目录整体是生成物）。
const OUT_DIR = path.resolve(__dirname, '../slides');
const OUT = path.join(OUT_DIR, '题目三_记忆打分_汇报.pptx');

// ---------- 配色：深墨蓝 + 琥珀 ----------
// 主题是「从冗余记忆里挑出重要的」，所以底色用档案感的深墨蓝，
// 唯一的高饱和色留给「被选中的片段」（琥珀），负结果用砖红。
const INK = '12203A';
const INK2 = '2A3A5C';
const PAPER = 'FFFFFF';
const MIST = 'F2F4F7';
const AMBER = 'C8862A';
const TEAL = '4E9A8F';
const BRICK = 'B4504A';
const SLATE = '7A8CA8';
const SAND = 'E8DCC8';

const HEAD = 'Microsoft YaHei';
const BODY = 'Microsoft YaHei';
const MONO = 'Courier New';

const THEME = {
  name: 'Ink and Amber',
  headFontFace: HEAD,
  bodyFontFace: BODY,
  colors: {
    dk1: INK, lt1: PAPER, dk2: INK2, lt2: MIST,
    accent1: AMBER, accent2: TEAL, accent3: BRICK,
    accent4: SLATE, accent5: SAND, accent6: INK2,
    hlink: AMBER, folHlink: SLATE,
  },
};

const GROUP = '第 29 组';
const MEMBERS = ['许兆丞', '文萧尧', '张瑞哲', '马家源'];

// 四个技术部分（按内容切，不按人切）
const PARTS = [
  { no: 1, title: '问题与方案', slides: [2, 3, 4, 5] },
  { no: 2, title: '数据与标签', slides: [6, 7, 8, 9, 10] },
  { no: 3, title: '模型与评测', slides: [11, 12, 13, 14, 15, 16, 17] },
  { no: 4, title: '实验结果', slides: [18, 19, 20, 21, 22, 23, 24, 25, 26, 27] },
  { no: 5, title: '校验与讨论', slides: [28, 29, 30, 31, 32, 33] },
];

const pres = new PptxGenJS();
pres.layout = 'LAYOUT_WIDE'; // 13.333 x 7.5，必须在加页之前设
pres.title = '小型神经网络驱动的 Agent 记忆重要性打分与长上下文优化';
pres.subject = '高级机器学习 专业选修 · 题目 3';
pres.author = '同济大学 计算机科学与技术';
pres.company = 'Tongji University';

const C = pres.SchemeColor;
const S = pres.ShapeType;

// ---------- layouts ----------
pres.defineSlideMaster({
  title: 'COVER',
  background: { color: INK },
  objects: [],
});

pres.defineSlideMaster({
  title: 'DARK',
  background: { color: INK },
  objects: [
    { text: { text: '高级机器学习 · 题目 3 · 记忆重要性打分与长上下文优化', options: { x: 0.75, y: 6.85, w: 8.5, h: 0.3, fontSize: 10, color: '9AA9C0', fontFace: BODY } } },
  ],
  slideNumber: { x: 12.2, y: 6.85, w: 0.7, h: 0.3, fontSize: 10, color: '9AA9C0', fontFace: BODY, align: 'right' },
});

pres.defineSlideMaster({
  title: 'CONTENT',
  background: { color: PAPER },
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: 0.75, y: 0.4, w: 11.8, h: 0.92, fontSize: 30, bold: true, color: INK, fontFace: HEAD, valign: 'middle', align: 'left', margin: 0 } } },
    { placeholder: { options: { name: 'body', type: 'body', x: 0.75, y: 1.45, w: 11.8, h: 4.9, fontSize: 15, color: INK2, fontFace: BODY, margin: 0 } } },
    { text: { text: '高级机器学习 · 题目 3 · 记忆重要性打分与长上下文优化', options: { x: 0.75, y: 6.88, w: 8.5, h: 0.28, fontSize: 9.5, color: SLATE, fontFace: BODY } } },
  ],
  slideNumber: { x: 12.2, y: 6.88, w: 0.7, h: 0.28, fontSize: 9.5, color: SLATE, fontFace: BODY, align: 'right' },
});

// ---------- 小工具 ----------
const txt = (s, text, o) => s.addText(text, Object.assign({ isTextBox: true, fontFace: BODY, color: INK2, margin: 0, valign: 'top' }, o));

/** 大数字 + 小标签 */
function statCard(s, { x, y, w, h, value, unit, label, accent, bg, fg }) {
  s.addShape(S.roundRect, { x, y, w, h, fill: { color: bg || MIST }, line: { color: bg || MIST }, rectRadius: 0.06 });
  txt(s, value, { x: x + 0.22, y: y + 0.16, w: w - 0.44, h: 0.62, fontSize: 28, bold: true, color: accent || AMBER, fontFace: HEAD });
  if (unit) txt(s, unit, { x: x + 0.22, y: y + 0.72, w: w - 0.44, h: 0.26, fontSize: 10.5, color: fg || SLATE });
  txt(s, label, { x: x + 0.22, y: y + (unit ? 0.98 : 0.78), w: w - 0.44, h: h - (unit ? 1.03 : 0.83), fontSize: 10.5, color: fg || INK2, lineSpacingMultiple: 1.22 });
}

/** 小节标题 */
function kicker(s, text, x, y, w, color) {
  txt(s, text, { x, y, w, h: 0.3, fontSize: 11, bold: true, color: color || AMBER, charSpacing: 1.2 });
}

/** 条目：圆点 + 标题 + 说明 */
function bullet(s, { x, y, w, head, desc, hw, dw, dot, headSize, descSize }) {
  s.addShape(S.ellipse, { x, y: y + 0.06, w: 0.14, h: 0.14, fill: { color: dot || AMBER }, line: { color: dot || AMBER } });
  txt(s, head, { x: x + 0.28, y, w: hw || w - 0.28, h: 0.32, fontSize: headSize || 12.5, bold: true, color: INK, lineSpacingMultiple: 1.1 });
  if (desc) txt(s, desc, { x: x + 0.28, y: y + 0.33, w: dw || w - 0.28, h: 0.34, fontSize: descSize || 10.5, color: INK2, lineSpacingMultiple: 1.3 });
}

/** 页码位置统一：内容页的标题占位 */
const title = (s, t) => s.addText(t, { placeholder: 'title' });

/** 页底结论条 */
function takeaway(s, text, { y = 6.28, h = 0.3, color = INK2, size = 11.5, bold = false } = {}) {
  txt(s, text, { x: 0.78, y, w: 11.8, h, fontSize: size, color, bold, lineSpacingMultiple: 1.25 });
}

// ============================================================
// 1. 封面
// ============================================================
{
  const s = pres.addSlide({ masterName: 'COVER' });

  // 右侧装饰：把「打分 → 过滤」画成一组横条，最长的三条是琥珀色
  const bars = [
    [1.15, 0], [0.72, 0], [1.85, 1], [0.55, 0], [1.32, 0], [2.15, 1],
    [0.88, 0], [1.55, 1], [0.62, 0], [1.05, 0], [1.72, 0], [0.95, 0],
  ];
  bars.forEach(([w, keep], i) => {
    s.addShape(S.roundRect, {
      x: 10.35, y: 1.75 + i * 0.36, w, h: 0.17, rectRadius: 0.03,
      fill: { color: keep ? AMBER : SLATE, transparency: keep ? 0 : 72 },
      line: { color: keep ? AMBER : SLATE, transparency: keep ? 0 : 100 },
    });
  });
  txt(s, '重要性得分', { x: 10.35, y: 1.42, w: 2.4, h: 0.26, fontSize: 10, color: '9AA9C0' });
  txt(s, '保留', { x: 12.0, y: 6.2, w: 0.9, h: 0.26, fontSize: 10, color: AMBER, align: 'right' });

  txt(s, '高级机器学习 · 专业选修 · 小组作业', { x: 0.9, y: 1.05, w: 8.5, h: 0.32, fontSize: 12.5, color: AMBER, bold: true, charSpacing: 1.5 });
  txt(s, '小型神经网络驱动的\nAgent 记忆重要性打分\n与长上下文优化', {
    x: 0.9, y: 1.6, w: 9.0, h: 2.5, fontSize: 38, bold: true, color: PAPER, fontFace: HEAD, lineSpacingMultiple: 1.16,
  });
  txt(s, '题目 3 ｜ MultiWOZ 2.2 ｜ 浅层全连接打分网络 + 预算内记忆过滤', { x: 0.9, y: 4.35, w: 9.0, h: 0.4, fontSize: 14, color: SAND });

  s.addShape(S.line, { x: 0.9, y: 5.15, w: 3.2, h: 0, line: { color: '3D4E6B', width: 1 } });
  txt(s, `${GROUP}　｜　${MEMBERS.join('　')}`, { x: 0.9, y: 5.32, w: 9.0, h: 0.34, fontSize: 13, color: 'B9C4D6' });
  txt(s, '同济大学 计算机科学与技术', { x: 0.9, y: 5.7, w: 9.0, h: 0.32, fontSize: 11.5, color: '9AA9C0' });
  txt(s, '2026 年 11 月', { x: 0.9, y: 6.06, w: 5, h: 0.32, fontSize: 11.5, color: '8A99B0' });

  s.addNotes(
    '封面。本汇报围绕题目三的技术内容展开：问题定义、方法设计、实验设置、结果与讨论。'
  );
}

pres.addSection({ title: '第一部分 · 问题与方案' });

// ============================================================
// 2. 内容概览
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第一部分 · 问题与方案' });
  title(s, '内容概览');

  const items = [
    ['01', '问题与方案', '长上下文的记忆冗余问题\n四模块架构与设计取舍'],
    ['02', '数据与标签', 'MultiWOZ 2.2 与数据对齐修正\n弱监督标签构造及其已知偏差'],
    ['03', '模型与评测', '打分网络结构与训练配置\n两层评测体系与指标定义'],
    ['04', '实验结果', 'L1 选择质量、预算扫描、消融\nL2 任务完成率与配对检验'],
    ['05', '校验与讨论', '标签三重独立校验\n负结果、局限与改进方向'],
  ];
  // 5 张卡必须落在 13.333 画布内：5*cw + 4*gap ≤ 12.55（左边距 0.78）
  const cw = 2.29, gap = 0.22, y0 = 1.75;
  items.forEach(([no, name, d], i) => {
    const x = 0.78 + i * (cw + gap);
    s.addShape(S.roundRect, { x, y: y0, w: cw, h: 3.05, fill: { color: MIST }, line: { color: MIST }, rectRadius: 0.06 });
    s.addShape(S.rect, { x, y: y0, w: cw, h: 0.08, fill: { color: [AMBER, TEAL, SLATE, INK, BRICK][i] }, line: { color: [AMBER, TEAL, SLATE, INK, BRICK][i] } });
    txt(s, no, { x: x + 0.2, y: y0 + 0.24, w: cw - 0.4, h: 0.4, fontSize: 20, bold: true, color: [AMBER, TEAL, SLATE, INK, BRICK][i], fontFace: HEAD });
    txt(s, name, { x: x + 0.2, y: y0 + 0.72, w: cw - 0.4, h: 0.32, fontSize: 13, bold: true, color: INK });
    txt(s, d, { x: x + 0.2, y: y0 + 1.2, w: cw - 0.4, h: 1.5, fontSize: 9.5, color: INK2, lineSpacingMultiple: 1.42 });
  });

  s.addShape(S.roundRect, { x: 0.78, y: 5.12, w: 11.8, h: 1.0, fill: { color: INK }, line: { color: INK }, rectRadius: 0.05 });
  txt(s, '核心结论', { x: 1.02, y: 5.26, w: 1.3, h: 0.28, fontSize: 10.5, color: AMBER, bold: true });
  txt(s, '浅层全连接打分网络在固定预算下显著优于位置与随机基线；但真正的瓶颈不是网络容量，而是打分器看不到对话边界，无法区分「本对话的历史」与「其他对话的干扰」。', {
    x: 2.35, y: 5.24, w: 10.0, h: 0.72, fontSize: 11.5, color: 'D7DEEA', lineSpacingMultiple: 1.35,
  });

  s.addNotes('汇报结构：五个部分依次为问题与方案、数据与标签、模型与评测、实验结果、校验与讨论。');
}

// ============================================================
// 3. 问题定义
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第一部分 · 问题与方案' });
  title(s, '问题：长上下文的记忆冗余');

  txt(s, '对话智能体需要历史记忆来保持一致性，但把全部历史塞进上下文会带来三个问题：', {
    x: 0.78, y: 1.45, w: 11.8, h: 0.32, fontSize: 12.5, color: INK2,
  });

  const probs = [
    ['成本线性增长', '上下文长度与 token 开销成正比。MultiWOZ 单段对话平均 7.7 轮（约 280 token），构造到 512 轮时库规模达 16,874 token。', BRICK],
    ['有效信息被稀释', '真正决定答案的往往只有几轮，其余是重复确认与寒暄。全部放入时，模型需要在噪声中自行定位。', AMBER],
    ['干扰引入错误', '本实验的直接观察：全量上下文（7,277 token）的槽位准确率 0.3222，反而低于筛选后的 1,489 token（0.3778）。', INK],
  ];
  probs.forEach(([h, d, col], i) => {
    const y = 1.95 + i * 1.32;
    s.addShape(S.roundRect, { x: 0.78, y, w: 0.06, h: 1.06, fill: { color: col }, line: { color: col } });
    txt(s, h, { x: 1.05, y, w: 3.0, h: 0.32, fontSize: 13.5, bold: true, color: INK });
    txt(s, d, { x: 1.05, y: y + 0.36, w: 11.3, h: 0.68, fontSize: 11.5, color: INK2, lineSpacingMultiple: 1.35 });
  });

  txt(s, '因此需要回答两个问题：', { x: 0.78, y: 5.8, w: 11.8, h: 0.3, fontSize: 11.5, bold: true, color: INK });
  txt(s, '① 如何判断一段历史「重要」——需要可规模化的监督信号，而不是人工标注；\n② 如何在小模型上把这个判断做准——网络该多深、需要什么特征。', {
    x: 0.78, y: 6.12, w: 11.8, h: 0.6, fontSize: 11.5, color: AMBER, lineSpacingMultiple: 1.38,
  });

  s.addNotes('问题的三个层面：成本、稀释、干扰。第三个是本次实验最重要的发现，也是后面 ORACLE 分析的出发点。');
}

// ============================================================
// 4. 题目要求与完成度
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第一部分 · 问题与方案' });
  title(s, '题目要求与完成情况');

  kicker(s, '必做要求', 0.78, 1.45, 6.2);
  const must = [
    ['具备记忆模块的对话智能体 + 向量检索记忆库', '四模块：memory / scorer / agent / eval'],
    ['构造记忆样本数据集，训练全连接打分网络', '65,213 轮弱监督样本，384→256→256→1'],
    ['三组基线：全量上下文 / 仅检索 / 检索+打分', 'B0 / B1 / B2，另加 6 组零成本基线'],
    ['消融：层数 / 学习率 / L2 正则 → 三指标', '12 个配置 × 3 seeds'],
  ];
  must.forEach(([h, d], i) => {
    const y = 1.82 + i * 1.12;
    s.addShape(S.ellipse, { x: 0.8, y: y + 0.04, w: 0.2, h: 0.2, fill: { color: AMBER }, line: { color: AMBER } });
    txt(s, h, { x: 1.14, y, w: 5.9, h: 0.5, fontSize: 12.5, bold: true, color: INK, lineSpacingMultiple: 1.15 });
    txt(s, d, { x: 1.14, y: y + 0.5, w: 5.9, h: 0.32, fontSize: 11, color: SLATE });
  });

  // 右侧：选做与交付物
  s.addShape(S.roundRect, { x: 7.45, y: 1.45, w: 5.1, h: 2.55, fill: { color: MIST }, line: { color: MIST }, rectRadius: 0.06 });
  txt(s, '选做拓展', { x: 7.7, y: 1.62, w: 4.6, h: 0.28, fontSize: 11.5, bold: true, color: INK });
  txt(s, '① 注入错误记忆，观察失效场景', { x: 7.7, y: 1.96, w: 4.6, h: 0.28, fontSize: 11.5, color: INK2 });
  txt(s, '部分完成 — 冲突干扰分析（第 26 页）', { x: 7.95, y: 2.22, w: 4.4, h: 0.26, fontSize: 10, color: BRICK });
  txt(s, '② 在打分基础上增加记忆摘要压缩', { x: 7.7, y: 2.56, w: 4.6, h: 0.28, fontSize: 11.5, color: INK2 });
  txt(s, '未做 — 工作量优先分配给标签校验', { x: 7.95, y: 2.82, w: 4.4, h: 0.26, fontSize: 10, color: BRICK });
  txt(s, '两项均如实标注完成度，不作模糊表述。', { x: 7.7, y: 3.28, w: 4.6, h: 0.3, fontSize: 10, color: SLATE });

  s.addShape(S.roundRect, { x: 7.45, y: 4.18, w: 5.1, h: 1.95, fill: { color: INK }, line: { color: INK }, rectRadius: 0.06 });
  txt(s, '交付物', { x: 7.7, y: 4.34, w: 4.6, h: 0.28, fontSize: 11.5, bold: true, color: AMBER });
  txt(s, '报告与实验报告（合并为一份，44K 字）\n四模块代码 + 48 项单元自检\n训练日志 · 消融原始结果 · 7 张图', {
    x: 7.7, y: 4.68, w: 4.6, h: 1.3, fontSize: 11, color: 'D7DEEA', lineSpacingMultiple: 1.45,
  });

  s.addNotes('四条必做逐条对应到实现，选做两项如实标注完成度：第一项部分完成（冲突干扰分析），第二项未做。');
}

// ============================================================
// 5. 四模块架构
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第一部分 · 问题与方案' });
  title(s, '四模块架构：依赖单向');

  const mods = [
    ['memory', '切片 · 编码 · 长历史构造\n检索 · 预算内选择', 'select() 是所有基线共用的截断逻辑', AMBER],
    ['scorer', '384→256→256→1 打分网络\n6 组零成本基线', '训练时必须同时报位置基线 AUC', TEAL],
    ['agent', 'prompt 构造 · LLM 客户端\n响应缓存 · 主循环', '缓存是必需项，不是性能优化', SLATE],
    ['eval', 'CR / NDCG / 任务完成率\n/ token 成本', 'L1 与 L2 两层独立调度', INK],
  ];
  const bw = 2.62, gap = 0.45, y0 = 1.75;
  mods.forEach(([name, d, key, col], i) => {
    const x = 0.78 + i * (bw + gap);
    s.addShape(S.roundRect, { x, y: y0, w: bw, h: 2.35, fill: { color: MIST }, line: { color: MIST }, rectRadius: 0.06 });
    s.addShape(S.rect, { x, y: y0, w: bw, h: 0.07, fill: { color: col }, line: { color: col } });
    txt(s, name, { x: x + 0.22, y: y0 + 0.2, w: bw - 0.44, h: 0.32, fontSize: 16, bold: true, color: INK, fontFace: MONO });
    txt(s, d, { x: x + 0.22, y: y0 + 0.58, w: bw - 0.44, h: 0.85, fontSize: 10.5, color: INK2, lineSpacingMultiple: 1.3 });
    txt(s, key, { x: x + 0.22, y: y0 + 1.5, w: bw - 0.44, h: 0.72, fontSize: 9.5, color: AMBER, lineSpacingMultiple: 1.25 });
    if (i < 3) {
      s.addShape(S.rightArrow, { x: x + bw + 0.09, y: y0 + 1.05, w: 0.27, h: 0.24, fill: { color: SLATE }, line: { color: SLATE } });
    }
  });

  kicker(s, '两个设计取舍', 0.78, 4.32, 5.6);
  const design = [
    ['依赖单向 ⇒ L1 可以不加载 LLM', 'eval 只导入前三模块即可跑 L1，省 3.2 GB 内存，把一次评测从 17 秒压到毫秒级。', TEAL],
    ['select() 单独成函数 ⇒ 公平性有代码级保证', 'B0/B1/B2/R0/R1/R2/R3 共用同一截断逻辑，唯一变量是 scores 从哪来。', AMBER],
  ];
  design.forEach(([h, d, col], i) => {
    const y = 4.66 + i * 0.86;
    s.addShape(S.roundRect, { x: 0.78, y, w: 0.06, h: 0.7, fill: { color: col }, line: { color: col } });
    txt(s, h, { x: 1.05, y, w: 11.4, h: 0.3, fontSize: 12, bold: true, color: INK });
    txt(s, d, { x: 1.05, y: y + 0.32, w: 11.4, h: 0.36, fontSize: 10.5, color: INK2 });
  });

  s.addNotes('四模块依赖单向是评测设计的前提：L1 不加载 LLM，因此可以全量跑、随时重跑。select() 共用保证七组基线可比。');
}

pres.addSection({ title: '第二部分 · 数据与标签' });

// ============================================================
// 6. 数据集
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第二部分 · 数据与标签' });
  title(s, '数据集：MultiWOZ 2.2');

  statCard(s, { x: 0.78, y: 1.5, w: 2.72, h: 1.5, value: '1,000', label: 'test 对话数' });
  statCard(s, { x: 3.7, y: 1.5, w: 2.72, h: 1.5, value: '8,372', label: 'test 轮次总数', accent: TEAL });
  statCard(s, { x: 6.62, y: 1.5, w: 2.72, h: 1.5, value: '7.7', label: '平均对话轮数', accent: SLATE });
  statCard(s, { x: 9.54, y: 1.5, w: 3.04, h: 1.5, value: '35.95', label: '单轮平均 token（中位 36，p90 59）', accent: INK });

  kicker(s, '为什么选这个数据集', 0.78, 3.25, 5.6);
  const why = [
    '提供逐轮 belief state 标注（states_22），使弱监督标签成为可能 —— 这是本方法不需要人工标注的前提。',
    '多领域（hotel / restaurant / train / taxi 等），槽位定义清晰，便于构造定向填空任务做 L2 评测。',
    '单段对话短（7.7 轮），天然需要人为构造长历史，便于控制「长度」与「干扰密度」两个变量。',
  ];
  why.forEach((t, i) => {
    txt(s, '· ' + t, { x: 0.78, y: 3.58 + i * 0.62, w: 11.8, h: 0.56, fontSize: 11.5, color: INK2, lineSpacingMultiple: 1.32 });
  });

  s.addShape(S.roundRect, { x: 0.78, y: 5.62, w: 11.8, h: 0.86, fill: { color: 'FBF3E6' }, line: { color: 'FBF3E6' }, rectRadius: 0.05 });
  txt(s, '划分方式', { x: 1.0, y: 5.74, w: 1.3, h: 0.28, fontSize: 10.5, bold: true, color: BRICK });
  txt(s, '按 dialogue_id 划分而非按轮次随机划分：train 8,437 段 / dev 1,000 段 / test 1,000 段。同一对话的轮次不会跨集合，避免标签泄漏。', {
    x: 2.32, y: 5.72, w: 10.0, h: 0.66, fontSize: 10.5, color: INK2, lineSpacingMultiple: 1.32,
  });

  s.addNotes('数据集三个特点：有 belief state 标注、多领域、对话短。第三个特点决定了必须自己构造长历史。');
}

// ============================================================
// 7. 数据对齐修正
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第二部分 · 数据与标签' });
  title(s, '数据对齐：一处必须修正的错位');

  txt(s, 'MultiWOZ 原始数据在一轮内先存 system 再存 user，即 sys_utt[t] 回复的是 usr_utt[t-1]。若直接按同下标配对，会把回复与其问题错开一轮。', {
    x: 0.78, y: 1.45, w: 11.8, h: 0.5, fontSize: 11.5, color: INK2, lineSpacingMultiple: 1.32,
  });

  // 探针表
  const head = ['探针', '现象', '判定'].map((t) => ({
    text: t, options: { bold: true, color: PAPER, fill: { color: INK }, fontSize: 10.5, align: 'center' },
  }));
  const rows = [
    ['用户问 postcode（309 次）', '邮编落在 sys[t+1] 266 次（86.1%），sys[t] 仅 7 次', '回复滞后一轮'],
    ['用户提 booking（844 次）', '确认落在 sys[t+1] 381 次，sys[t] 35 次', '回复滞后一轮'],
    ['sys 以问号结尾（4,530 次）', '短答落在 usr[t+1] 1,015 次，usr[t] 514 次', '提问后置一轮'],
    ['turn 0 的 sys_utt', '1,000/1,000 段对话全部为 none', '决定性证据'],
  ].map((r, i) => r.map((c, j) => ({
    text: c,
    options: {
      fontSize: 10, align: j === 2 ? 'center' : 'left', color: j === 2 ? BRICK : INK2,
      fill: { color: i === 3 ? 'FBF3E6' : (i % 2 ? MIST : PAPER) }, bold: i === 3,
    },
  })));

  s.addTable([head, ...rows], {
    x: 0.78, y: 2.08, w: 11.8,
    colW: [3.1, 6.3, 2.4],
    border: { type: 'solid', color: 'E3E7EC', pt: 1 },
    rowH: 0.44, valign: 'middle', fontFace: BODY, margin: 0.06,
  });

  kicker(s, '修正结果', 0.78, 4.38, 4.0, TEAL);
  s.addShape(S.roundRect, { x: 0.78, y: 4.72, w: 5.75, h: 1.0, fill: { color: INK }, line: { color: INK }, rectRadius: 0.05 });
  txt(s, 'turn t = (usr_utt[t], sys_utt[t+1], states_22[t])', { x: 1.0, y: 4.86, w: 5.35, h: 0.32, fontSize: 11.5, color: AMBER, fontFace: MONO });
  txt(s, '按此定义全量重跑，此前的错位结果全部废弃。', { x: 1.0, y: 5.22, w: 5.35, h: 0.36, fontSize: 10, color: 'D7DEEA' });

  txt(s, '状态标注本身对齐 user（delta 值出现在 usr[t] 占 71.8%，usr[t-1] 仅 2.5%），所以错位只影响对话文本配对，不影响标签来源。', {
    x: 6.78, y: 4.76, w: 5.8, h: 0.9, fontSize: 10.5, color: INK2, lineSpacingMultiple: 1.35,
  });

  takeaway(s, '结论：四条独立探针一致指向回复滞后一轮。这是数据层面的系统性问题，不修正会让所有下游结论建立在错位的输入上。', { y: 5.98, h: 0.5, color: SLATE, size: 10.5 });

  s.addNotes('四条探针中最有说服力的是第四条：turn 0 的 system 在 1000 段对话里全为 none，因为首轮没有「上一轮的回复」。');
}

// ============================================================
// 8. 弱监督标签
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第二部分 · 数据与标签' });
  title(s, '弱监督标签：belief state 差分');

  s.addShape(S.roundRect, { x: 0.78, y: 1.5, w: 6.9, h: 2.35, fill: { color: INK }, line: { color: INK }, rectRadius: 0.06 });
  const code = [
    'cur    = norm_state(states_22[t])',
    'delta  = {k: v for k, v in cur.items()',
    '          if prev.get(k) != v}',
    'n_valid = sum(1 for k, v in delta.items()',
    '              if final.get(k) == v)',
    'label  = int(n_valid > 0)',
  ];
  code.forEach((line, i) => {
    txt(s, line, { x: 1.02, y: 1.7 + i * 0.32, w: 6.5, h: 0.3, fontSize: 11, color: i === 5 ? AMBER : 'D7DEEA', fontFace: MONO });
  });

  txt(s, '第 t 轮「重要」⟺ 该轮引入了状态变化，且至少一个 (slot, value) 到对话结束仍是原值。', {
    x: 0.78, y: 3.98, w: 6.9, h: 0.34, fontSize: 11, color: INK2, lineSpacingMultiple: 1.3,
  });
  txt(s, 'final 取对话中最后一个非空 state，而不是最后一轮 —— 末尾可能有未标注轮。', {
    x: 0.78, y: 4.34, w: 6.9, h: 0.34, fontSize: 10.5, color: SLATE,
  });

  statCard(s, { x: 8.0, y: 1.5, w: 2.2, h: 1.55, value: '65,213', label: 'train 标签轮数' });
  statCard(s, { x: 10.38, y: 1.5, w: 2.2, h: 1.55, value: '55.6%', label: '正例率（预期 25–40%）' });

  s.addShape(S.roundRect, { x: 8.0, y: 3.24, w: 4.58, h: 1.5, fill: { color: 'FBF3E6' }, line: { color: 'FBF3E6' }, rectRadius: 0.06 });
  txt(s, '位置偏置：必须先处理', { x: 8.22, y: 3.4, w: 4.2, h: 0.28, fontSize: 11, bold: true, color: BRICK });
  txt(s, '正例率 turn 0 = 85.0% → turn 10 = 16.0%（首因效应）。\n仅用「对话内相对位置」这一条零成本规则，AUC 就有 0.8172。', {
    x: 8.22, y: 3.7, w: 4.2, h: 0.9, fontSize: 10, color: INK2, lineSpacingMultiple: 1.32,
  });

  kicker(s, '为什么不用人工标注', 0.78, 4.9, 6.0);
  const why = [
    ['65,213 条', '按每条 8 秒计需 145 小时，且跨标注员一致性无法保证'],
    ['规则可复核', '任何人拿 states_22 重跑这三行代码，得到逐位相同的标签'],
    ['代价是偏差', '标的是「引入」而非「必要」，49.2% 轮次在集合覆盖下可删（第 28–30 页量化）'],
  ];
  why.forEach(([k, v], i) => {
    const y = 5.24 + i * 0.46;
    txt(s, k, { x: 0.78, y, w: 1.65, h: 0.3, fontSize: 10.5, bold: true, color: AMBER });
    txt(s, v, { x: 2.5, y, w: 6.1, h: 0.3, fontSize: 10, color: INK2 });
  });

  s.addNotes('三行代码定义标签。两个数字需要强调：65,213 轮、正例率 55.6%。位置偏置必须主动说明，否则 0.9332 的 AUC 会被误认为全部来自网络。');
}

// ============================================================
// 9. 标签的已知偏差
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第二部分 · 数据与标签' });
  title(s, '标签的已知偏差：标的是「引入」不是「必要」');

  s.addShape(S.roundRect, { x: 0.78, y: 1.5, w: 5.75, h: 1.85, fill: { color: MIST }, line: { color: MIST }, rectRadius: 0.06 });
  txt(s, '标签的定义', { x: 1.0, y: 1.66, w: 5.3, h: 0.28, fontSize: 11, bold: true, color: INK });
  txt(s, '只要这一轮引入了新的状态值、且该值到对话结束仍有效，就标为重要。\n它回答的是「这一轮有没有带来信息」，而不是「缺了这一轮任务会不会失败」。', {
    x: 1.0, y: 1.98, w: 5.3, h: 1.2, fontSize: 10.5, color: INK2, lineSpacingMultiple: 1.38,
  });

  s.addShape(S.roundRect, { x: 6.83, y: 1.5, w: 5.75, h: 1.85, fill: { color: 'FBF3E6' }, line: { color: 'FBF3E6' }, rectRadius: 0.06 });
  txt(s, '偏差有多大', { x: 7.05, y: 1.66, w: 5.3, h: 0.28, fontSize: 11, bold: true, color: BRICK });
  txt(s, '集合覆盖分析：test 上 49.24% 的轮次可删（正例中有 33.34% 不在最小覆盖集）。\n即大量被标为重要的轮次其实是冗余的 —— 同一槽位被多轮重复确认时，只需保留最后一轮。', {
    x: 7.05, y: 1.98, w: 5.3, h: 1.2, fontSize: 10.5, color: INK2, lineSpacingMultiple: 1.38,
  });

  kicker(s, '这个偏差怎么处理', 0.78, 3.58, 6.0);
  const handle = [
    ['不改标签，先验证影响', '按「必要」重新定义标签（正例率 55.6% → 32.3%）重训打分器，配对比较 CR 0.7778 vs 0.7556，p=0.500 不显著。83.3% 场景结果完全相同（第 32 页）。', TEAL],
    ['在报告中量化并披露', '三个独立口径校验（反事实删除 / JGA / 模拟抽检），全部报告限定条件，不把「引入」当作「必要」使用。', AMBER],
    ['把偏差写进适用范围', '结论只在「过滤冗余」这一目标下成立；若目标是「最小充分子集」，需要重新定义标签。', SLATE],
  ];
  handle.forEach(([h, d, col], i) => {
    const y = 3.92 + i * 0.78;
    s.addShape(S.roundRect, { x: 0.78, y, w: 0.06, h: 0.64, fill: { color: col }, line: { color: col } });
    txt(s, h, { x: 1.05, y, w: 4.0, h: 0.3, fontSize: 11.5, bold: true, color: INK });
    txt(s, d, { x: 1.05, y: y + 0.31, w: 11.4, h: 0.34, fontSize: 10, color: INK2, lineSpacingMultiple: 1.25 });
  });

  s.addNotes('这一页是主动披露局限。关键点：偏差真实存在（49.24% 可删），但换标签重训后指标无显著差异，所以保留现标签并在结论中限定适用范围。');
}

// ============================================================
// 10. 长历史构造
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第二部分 · 数据与标签' });
  title(s, '长历史构造：五个控制点');

  txt(s, 'MultiWOZ 单段对话平均 7.7 轮（约 280 token），无法直接制造上下文溢出，因此需要人为构造。构造方式直接决定实验是否可信。', {
    x: 0.78, y: 1.45, w: 11.8, h: 0.34, fontSize: 11, color: INK2,
  });

  const steps = [
    ['① 主对话', '取一段真实对话作为 primary，保留其全部轮次，不删不改。'],
    ['② 干扰轮', '从同领域的其他对话抽取 decoy 补足到目标长度 L，保证语义相关而非随机噪声。'],
    ['③ 等距铺开', '主对话轮次等距插入，而不是全挤在开头 —— 否则位置基线会凭空获胜。'],
    ['④ 匹配轮次分布', 'decoy 的 turn_idx 只从 hist_idx 池中抽取，与主对话分布一致，否则绝对轮次基线会假性领先。'],
    ['⑤ needle 位置', 'needle 落在 0–90% 分位随机，避免固定在某一区间。'],
  ];
  steps.forEach(([h, d], i) => {
    const y = 1.95 + i * 0.82;
    s.addShape(S.roundRect, { x: 0.78, y, w: 7.6, h: 0.72, fill: { color: i % 2 ? MIST : PAPER }, line: { color: i % 2 ? MIST : 'E8ECF1' }, rectRadius: 0.04 });
    txt(s, h, { x: 1.0, y: y + 0.06, w: 1.85, h: 0.3, fontSize: 11, bold: true, color: AMBER });
    txt(s, d, { x: 2.9, y: y + 0.05, w: 5.3, h: 0.6, fontSize: 10, color: INK2, lineSpacingMultiple: 1.25 });
  });

  // 右侧：实际达到的规模（诚实标注）
  s.addShape(S.roundRect, { x: 8.65, y: 1.95, w: 3.93, h: 2.55, fill: { color: 'FBF0EF' }, line: { color: 'FBF0EF' }, rectRadius: 0.06 });
  txt(s, '实际达到的规模', { x: 8.87, y: 2.1, w: 3.5, h: 0.28, fontSize: 11, bold: true, color: BRICK });
  txt(s, 'L=128：90 场景全部达到\nL=256：90 场景全部达到\nL=512：仅 23 个真到 512 轮\n　　　 其余落在 354–486', {
    x: 8.87, y: 2.42, w: 3.5, h: 1.2, fontSize: 10.5, color: INK2, lineSpacingMultiple: 1.45,
  });
  txt(s, '候选池不足导致。报告与图中均注明，避免「512 轮也扛得住」的误读。', {
    x: 8.87, y: 3.66, w: 3.5, h: 0.7, fontSize: 9.5, color: BRICK, lineSpacingMultiple: 1.3,
  });

  s.addShape(S.roundRect, { x: 8.65, y: 4.66, w: 3.93, h: 1.3, fill: { color: MIST }, line: { color: MIST }, rectRadius: 0.06 });
  txt(s, '库规模（token）', { x: 8.87, y: 4.8, w: 3.5, h: 0.26, fontSize: 10, color: SLATE });
  txt(s, 'L=128　4,497\nL=256　8,910\nL=512　16,874', {
    x: 8.87, y: 5.06, w: 3.5, h: 0.82, fontSize: 10.5, color: INK, fontFace: MONO, lineSpacingMultiple: 1.4,
  });

  takeaway(s, '这五个控制点决定了后续所有对比是否成立：任何一处放松，位置或长度基线都会虚高。', { y: 6.18, h: 0.34, color: SLATE, size: 10.5 });

  s.addNotes('构造方式决定实验可信度。第③④两点最关键：不控制的话位置基线和绝对轮次基线会假性领先。');
}

pres.addSection({ title: '第三部分 · 模型与评测' });

// ============================================================
// 11. 记忆片段与选择函数
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第三部分 · 模型与评测' });
  title(s, '记忆片段与预算内选择');

  kicker(s, '片段定义', 0.78, 1.45, 4.0);
  txt(s, '一轮对话 (user, system) 为一条记忆。token 数用 Qwen2.5 tokenizer 精确计数，不按空格估算。', {
    x: 0.78, y: 1.75, w: 5.6, h: 0.5, fontSize: 10.5, color: INK2, lineSpacingMultiple: 1.3,
  });
  const stats = [['35.95', '均值'], ['36', '中位'], ['59', 'p90'], ['152', '最大']];
  stats.forEach(([v, l], i) => {
    const x = 0.78 + i * 1.42;
    s.addShape(S.roundShape || S.roundRect, { x, y: 2.3, w: 1.3, h: 0.86, fill: { color: MIST }, line: { color: MIST }, rectRadius: 0.05 });
    txt(s, v, { x, y: 2.38, w: 1.3, h: 0.36, fontSize: 16, bold: true, color: AMBER, align: 'center', fontFace: HEAD });
    txt(s, l, { x, y: 2.76, w: 1.3, h: 0.26, fontSize: 9.5, color: SLATE, align: 'center' });
  });

  kicker(s, '选择函数', 0.78, 3.38, 4.0, TEAL);
  s.addShape(S.roundRect, { x: 0.78, y: 3.7, w: 5.6, h: 0.62, fill: { color: INK }, line: { color: INK }, rectRadius: 0.05 });
  txt(s, 'select(scores, tokens, budget, reserve_recent=2)', { x: 1.0, y: 3.84, w: 5.2, h: 0.34, fontSize: 11, color: AMBER, fontFace: MONO });
  txt(s, '按分数降序贪心填入，直到下一个片段会超出预算为止；reserve_recent=2 为最近两轮保留位置。\n所有选择器共用此函数，唯一变量是 scores 的来源。', {
    x: 0.78, y: 4.44, w: 5.6, h: 0.86, fontSize: 10.5, color: INK2, lineSpacingMultiple: 1.35,
  });

  // 右侧：分数来源对照
  s.addShape(S.roundRect, { x: 6.78, y: 1.45, w: 5.8, h: 4.35, fill: { color: MIST }, line: { color: MIST }, rectRadius: 0.06 });
  txt(s, '同一函数，不同分数来源', { x: 7.0, y: 1.6, w: 5.4, h: 0.3, fontSize: 11.5, bold: true, color: INK });
  const src = [
    ['B2', '打分网络输出（全库稠密打分）', AMBER],
    ['B1', 'query 与片段的余弦相似度', TEAL],
    ['B0', '时间序，超预算后从头截断', SLATE],
    ['B2R', '先检索 top-50 再重排（已弃用）', 'C9D0DA'],
    ['R0', '−相对位置（越早越重要）', SLATE],
    ['R1', 'recency（越近越重要）', SLATE],
    ['R2', '随机', SLATE],
    ['R3', '规则「前 3 轮」', SLATE],
    ['LEN', '片段长度', SLATE],
  ];
  src.forEach(([k, v, col], i) => {
    const y = 2.0 + i * 0.41;
    txt(s, k, { x: 7.0, y, w: 0.75, h: 0.28, fontSize: 10.5, bold: true, color: col, fontFace: MONO });
    txt(s, v, { x: 7.8, y, w: 4.55, h: 0.28, fontSize: 10, color: INK2 });
  });

  takeaway(s, '把选择逻辑抽成单一函数，是为了让「打分器是否有效」这个问题不被实现差异污染。', { y: 6.06, h: 0.34, color: SLATE, size: 10.5 });

  s.addNotes('片段定义与选择函数。关键设计：所有基线共用同一截断逻辑，因此指标差异只能归因于分数来源。');
}

// ============================================================
// 12. 十组选择器
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第三部分 · 模型与评测' });
  title(s, '十组选择器：三组主基线 + 七组对照');

  const groups = [
    ['主基线（题目要求）', [
      ['B0', '原始全量上下文', '不筛选，超预算后按时间序从头截断'],
      ['B1', '仅向量检索', '按 query 余弦相似度排序'],
      ['B2', '检索 + 打分（本题方法）', '全库稠密打分，与 query 无关的绝对判断'],
    ], AMBER],
    ['零成本对照（自加）', [
      ['R0 / R0A', '位置启发式', '越早越重要 / 按绝对轮次'],
      ['R1 / R2 / R3', 'recency / 随机 / 前 3 轮', '三个不依赖模型的朴素策略'],
      ['LEN', '片段长度', '长片段优先，用于检验是否只是复现长度信号'],
      ['B2R', '两阶段检索后重排', '已弃用，见下页原因'],
    ], TEAL],
  ];
  groups.forEach(([gname, rows, col], gi) => {
    // 第一组 3 行、第二组 4 行：第二组要多一行，行距必须更紧，
    // 否则最后一行 B2R 会被底部说明条（y=6.14）压住。
    const y0 = gi === 0 ? 1.5 : 3.68;
    const step = gi === 0 ? 0.62 : 0.56;
    kicker(s, gname, 0.78, y0, 6.0, col);
    rows.forEach(([k, name, d], i) => {
      const y = y0 + 0.34 + i * step;
      txt(s, k, { x: 0.78, y, w: 1.35, h: 0.3, fontSize: 11, bold: true, color: col, fontFace: MONO });
      txt(s, name, { x: 2.2, y, w: 3.2, h: 0.3, fontSize: 11, bold: true, color: INK });
      txt(s, d, { x: 5.5, y, w: 7.0, h: 0.3, fontSize: 10, color: INK2 });
    });
  });

  s.addShape(S.roundRect, { x: 0.78, y: 6.14, w: 11.8, h: 0.56, fill: { color: 'FBF3E6' }, line: { color: 'FBF3E6' }, rectRadius: 0.05 });
  txt(s, '为什么要七组对照', { x: 1.0, y: 6.22, w: 1.8, h: 0.26, fontSize: 10, bold: true, color: BRICK });
  txt(s, '三组主基线只能回答「比不筛选好多少」；七组零成本对照才能回答「这个提升是不是随便一个启发式就能拿到」。位置基线 AUC 0.8172 就是后者的产物。', {
    x: 2.85, y: 6.2, w: 9.5, h: 0.44, fontSize: 9.5, color: INK2, lineSpacingMultiple: 1.24,
  });

  s.addNotes('三组主基线是题目要求，七组对照是自加。加对照的原因：没有它们无法判断提升是否来自方法本身。');
}

// ============================================================
// 13. 为什么全库稠密打分
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第三部分 · 模型与评测' });
  title(s, '为什么不用「先检索再重排」');

  txt(s, '两阶段方案（先取余弦 top-50，再用打分器重排）看起来更省算力，但实测在长历史上会失效。', {
    x: 0.78, y: 1.45, w: 11.8, h: 0.34, fontSize: 11, color: INK2,
  });

  // 关键数字：needle 进入 top-50 的比例随 L 下降
  const data = [['L=128', '0.678'], ['L=256', '0.456'], ['L=512', '0.344']];
  data.forEach(([l, v], i) => {
    const x = 0.78 + i * 3.15;
    s.addShape(S.roundRect, { x, y: 1.95, w: 2.95, h: 1.35, fill: { color: MIST }, line: { color: MIST }, rectRadius: 0.06 });
    txt(s, l, { x: x + 0.22, y: 2.08, w: 2.5, h: 0.28, fontSize: 10.5, color: SLATE });
    txt(s, v, { x: x + 0.22, y: 2.34, w: 2.5, h: 0.5, fontSize: 24, bold: true, color: i === 2 ? BRICK : AMBER, fontFace: HEAD });
    txt(s, 'needle 进入余弦 top-50 的比例', { x: x + 0.22, y: 2.86, w: 2.5, h: 0.32, fontSize: 9.5, color: INK2 });
  });

  txt(s, '随历史变长单调下降：L=512 时只有 34.4% 的 needle 能进入候选池，两阶段方案的上限已被锁死。', {
    x: 0.78, y: 3.42, w: 11.8, h: 0.32, fontSize: 11, color: BRICK, bold: true,
  });

  // 对比：全库打分的成本
  s.addShape(S.roundRect, { x: 0.78, y: 4.0, w: 5.75, h: 1.85, fill: { color: 'FBF0EF' }, line: { color: 'FBF0EF' }, rectRadius: 0.06 });
  txt(s, '两阶段（B2R）', { x: 1.0, y: 4.14, w: 5.3, h: 0.3, fontSize: 11.5, bold: true, color: BRICK });
  txt(s, '候选池受限，长历史上限被锁死。\n实测跨长度 CR 0.5000，低于纯检索的 0.4778 以上水平，已弃用。', {
    x: 1.0, y: 4.46, w: 5.3, h: 1.2, fontSize: 10.5, color: INK2, lineSpacingMultiple: 1.38,
  });

  s.addShape(S.roundRect, { x: 6.83, y: 4.0, w: 5.75, h: 1.85, fill: { color: MIST }, line: { color: MIST }, rectRadius: 0.06 });
  txt(s, '全库稠密打分（B2）', { x: 7.05, y: 4.14, w: 5.3, h: 0.3, fontSize: 11.5, bold: true, color: TEAL });
  txt(s, '对 8,372 段全部打分只需 0.014 秒（2 核 CPU，384 维句向量）。\n成本可忽略，因此没有理由牺牲召回。', {
    x: 7.05, y: 4.46, w: 5.3, h: 1.2, fontSize: 10.5, color: INK2, lineSpacingMultiple: 1.38,
  });

  takeaway(s, '这个取舍与题目目标一致：题目要的是「控制 token 开销」，而选择阶段的算力不是瓶颈 —— 真正昂贵的是 LLM。', { y: 6.06, h: 0.34, color: SLATE, size: 10.5 });

  s.addNotes('两阶段方案的失效是实测结果，不是理论推断：needle 进入 top-50 的比例随 L 从 0.678 降到 0.344。全库打分只需 0.014 秒，没有理由省这个。');
}

// ============================================================
// 14. 打分网络结构
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第三部分 · 模型与评测' });
  title(s, '打分网络：384 → 256 → 256 → 1');

  const boxes = [['384', '句向量\nall-MiniLM-L6-v2'], ['256', 'Linear + GELU\n+ Dropout 0.1'], ['256', 'Linear + GELU\n+ Dropout 0.1'], ['1', 'logit']];
  const bx = 0.78, bwd = 2.5, bgap = 0.55, by = 1.65;
  boxes.forEach(([n, d], i) => {
    const x = bx + i * (bwd + bgap);
    s.addShape(S.roundRect, { x, y: by, w: bwd, h: 1.3, fill: { color: i === 3 ? INK : MIST }, line: { color: i === 3 ? INK : 'DCE2EA' }, rectRadius: 0.06 });
    txt(s, n, { x, y: by + 0.14, w: bwd, h: 0.44, fontSize: 22, bold: true, color: i === 3 ? AMBER : INK, align: 'center', fontFace: HEAD });
    txt(s, d, { x: x + 0.1, y: by + 0.62, w: bwd - 0.2, h: 0.6, fontSize: 9.5, color: i === 3 ? 'B9C4D6' : SLATE, align: 'center', lineSpacingMultiple: 1.25 });
    if (i < 3) s.addShape(S.rightArrow, { x: x + bwd + 0.12, y: by + 0.52, w: 0.3, h: 0.24, fill: { color: SLATE }, line: { color: SLATE } });
  });

  statCard(s, { x: 0.78, y: 3.2, w: 2.72, h: 1.5, value: '0.9332', label: 'test AUC' });
  statCard(s, { x: 3.7, y: 3.2, w: 2.72, h: 1.5, value: '0.8172', label: '最强零成本基线（相对位置）', accent: SLATE });
  statCard(s, { x: 6.62, y: 3.2, w: 2.72, h: 1.5, value: '0.1160', label: '净优势（AUC 差值）', accent: TEAL });
  statCard(s, { x: 9.54, y: 3.2, w: 3.04, h: 1.5, value: '165,377', label: '参数量（2 层；1 层 99,585）', accent: AMBER });

  kicker(s, '为什么浅层就够', 0.78, 4.95, 5.0, BRICK);
  txt(s, '判据是「这一轮有没有引入状态变化」，属于低阶特征。384 维句向量中该信息已近似线性可分，加深只增加参数而不增加信息量。第 22 页消融验证：1 层 0.9352 反而最高。', {
    x: 0.78, y: 5.28, w: 11.8, h: 0.6, fontSize: 11, color: INK2, lineSpacingMultiple: 1.35,
  });

  takeaway(s, '前置 LayerNorm：句向量各维尺度差异大，不归一化时训练不稳定（前 2 个 epoch 出现梯度爆炸）。', { y: 6.02, h: 0.34, color: SLATE, size: 10.5 });

  s.addNotes('网络结构极简：两个隐藏层，16.5 万参数。重点是 AUC 0.9332 相对位置基线 0.8172 的净优势 0.1160 —— 这个差值才是方法的贡献。');
}

// ============================================================
// 15. 训练配置与 pos_weight
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第三部分 · 模型与评测' });
  title(s, '训练配置：一个反直觉的坑');

  const head = ['超参', '取值'].map((t) => ({
    text: t, options: { bold: true, color: PAPER, fill: { color: INK }, fontSize: 11, align: 'center' },
  }));
  const rows = [
    ['学习率', '3e-4'],
    ['weight_decay', '1e-4'],
    ['batch size', '256'],
    ['最大 epoch', '20'],
    ['早停', 'patience 3（监控 val AUC）'],
    ['pos_weight', 'auto = 0.7993'],
    ['随机种子', '3 seeds，报均值与标准差'],
  ].map((r, i) => r.map((c, j) => ({
    text: c,
    options: {
      fontSize: 10.5, align: j === 0 ? 'left' : 'center', color: INK2,
      fill: { color: i === 5 ? 'FBF3E6' : (i % 2 ? MIST : PAPER) }, bold: i === 5 && j === 1,
    },
  })));

  s.addTable([head, ...rows], {
    x: 0.78, y: 1.5, w: 5.4,
    colW: [1.85, 3.55],
    border: { type: 'solid', color: 'E3E7EC', pt: 1 },
    // 表头 + 7 行 = 8 行，0.38 才落在 4.54，给下面的观察项留出 0.2
    rowH: 0.38, valign: 'middle', fontFace: BODY, margin: 0.05,
  });

  // 右侧：pos_weight 的坑
  s.addShape(S.roundRect, { x: 6.6, y: 1.5, w: 5.98, h: 2.2, fill: { color: 'FBF3E6' }, line: { color: 'FBF3E6' }, rectRadius: 0.06 });
  txt(s, 'pos_weight 的方向会反', { x: 6.84, y: 1.64, w: 5.5, h: 0.3, fontSize: 12, bold: true, color: BRICK });
  txt(s, '本任务正例率 55.6%，偏正而非偏负。若按惯例设「负/正」= 0.80，方向恰好相反 —— 会把模型推向全判 1。', {
    x: 6.84, y: 1.98, w: 5.5, h: 0.66, fontSize: 10.5, color: INK2, lineSpacingMultiple: 1.35,
  });
  txt(s, '正确做法：用 auto，按实际正负比计算。实测负 25,159 / 正 31,477 ⇒ 0.7993。', {
    x: 6.84, y: 2.66, w: 5.5, h: 0.5, fontSize: 10.5, color: INK2, lineSpacingMultiple: 1.35,
  });
  txt(s, '注意：0.80 与 0.7993 数值几乎相同，但含义相反。这是本实验最容易出错的一处。', {
    x: 6.84, y: 3.2, w: 5.5, h: 0.44, fontSize: 10, color: BRICK, lineSpacingMultiple: 1.3,
  });

  kicker(s, '训练过程观察', 0.78, 4.72, 5.0, TEAL);
  const obs = [
    'val AUC 在第 5 个 epoch 达到峰值后不再提升，早停随即触发 —— 这也是 L2 正则消融失效的直接原因（第 22 页）。',
    'train 与 val 的 AUC 差距始终小于 0.01，说明 16.5 万参数相对 6.5 万训练样本并不过拟合。',
    '3 个种子间的 test AUC 标准差为 0.0007，与消融中任意超参造成的差异同量级。',
  ];
  obs.forEach((t, i) => {
    txt(s, '· ' + t, { x: 0.78, y: 5.02 + i * 0.5, w: 11.8, h: 0.46, fontSize: 10.5, color: INK2, lineSpacingMultiple: 1.25 });
  });

  s.addNotes('训练配置很常规，唯一的坑是 pos_weight。因为正例率过半，按惯例算「负/正」会得到 0.80，数值巧合地接近正确值 0.7993，但方向相反。');
}

// ============================================================
// 16. 两层评测
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第三部分 · 模型与评测' });
  title(s, '两层评测：为什么必须拆开');

  s.addShape(S.roundRect, { x: 0.78, y: 1.5, w: 5.75, h: 3.3, fill: { color: MIST }, line: { color: MIST }, rectRadius: 0.06 });
  txt(s, 'L1 · 选择质量', { x: 1.02, y: 1.66, w: 5.3, h: 0.36, fontSize: 15, bold: true, color: INK, fontFace: HEAD });
  txt(s, '不需要 LLM', { x: 1.02, y: 2.02, w: 5.3, h: 0.26, fontSize: 10, color: TEAL, bold: true });
  txt(s, '指标：CR（needle 是否被选入）\nRecall@budget、NDCG@k', { x: 1.02, y: 2.34, w: 5.3, h: 0.66, fontSize: 11.5, color: INK2, lineSpacingMultiple: 1.35 });
  txt(s, '成本：全库 8,372 段打分 0.014 秒\n90 场景 × 3 长度 = 153.6 秒', { x: 1.02, y: 3.08, w: 5.3, h: 0.66, fontSize: 11.5, color: INK2, lineSpacingMultiple: 1.35 });
  txt(s, '可全量跑、随时重跑', { x: 1.02, y: 3.84, w: 5.3, h: 0.3, fontSize: 10.5, color: AMBER, bold: true });
  txt(s, '用途：判断方法成不成立', { x: 1.02, y: 4.16, w: 5.3, h: 0.3, fontSize: 10.5, color: SLATE });

  s.addShape(S.roundRect, { x: 6.83, y: 1.5, w: 5.75, h: 3.3, fill: { color: INK }, line: { color: INK }, rectRadius: 0.06 });
  txt(s, 'L2 · 任务完成率', { x: 7.07, y: 1.66, w: 5.3, h: 0.36, fontSize: 15, bold: true, color: PAPER, fontFace: HEAD });
  txt(s, '需要 LLM backbone', { x: 7.07, y: 2.02, w: 5.3, h: 0.26, fontSize: 10, color: AMBER, bold: true });
  txt(s, '指标：slot_acc、场景成功率、AR\ntoken 消耗、CR', { x: 7.07, y: 2.34, w: 5.3, h: 0.66, fontSize: 11.5, color: 'D7DEEA', lineSpacingMultiple: 1.35 });
  txt(s, '成本：本地 0.5B 约 17 秒/样本\n90 场景 API 约 28 分钟', { x: 7.07, y: 3.08, w: 5.3, h: 0.66, fontSize: 11.5, color: 'D7DEEA', lineSpacingMultiple: 1.35 });
  txt(s, '只能跑子集，且需挑对 backbone', { x: 7.07, y: 3.84, w: 5.3, h: 0.3, fontSize: 10.5, color: AMBER, bold: true });
  txt(s, '用途：回答题目要求的三个指标', { x: 7.07, y: 4.16, w: 5.3, h: 0.3, fontSize: 10.5, color: '9AA9C0' });

  txt(s, '策略：先用 L1 全量扫（秒级）。若打分器赢不过位置基线就停下来改标签，不浪费半小时跑 L2 —— 把「发现方法不行」的成本从小时级压到秒级。', {
    x: 0.78, y: 5.0, w: 11.8, h: 0.5, fontSize: 11.5, color: INK2, lineSpacingMultiple: 1.3,
  });
  txt(s, 'L2 的 backbone 换过一次：本地 0.5B 的绝对 JGA 只有 0.589（train-day 槽位 0.036），改用 DeepSeek-V4.1-Flash 后为 0.9174。绝对数值以 API 为准，0.5B 仅用于打通管线。', {
    x: 0.78, y: 5.56, w: 11.8, h: 0.62, fontSize: 10.5, color: SLATE, lineSpacingMultiple: 1.32,
  });

  s.addNotes('两层评测是本次实验设计上最重要的决定。LLM 是唯一昂贵环节，其余全部免费，因此可以先全量验证方法是否成立。');
}

// ============================================================
// 17. 填空 vs 问答
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第三部分 · 模型与评测' });
  title(s, '任务形式：为什么用定向填空而不是问答');

  s.addShape(S.roundRect, { x: 0.78, y: 1.5, w: 5.75, h: 1.75, fill: { color: 'FBF0EF' }, line: { color: 'FBF0EF' }, rectRadius: 0.06 });
  txt(s, '问答式', { x: 1.02, y: 1.64, w: 5.3, h: 0.3, fontSize: 11.5, bold: true, color: BRICK });
  txt(s, '「Answer with the exact value: for how many people is the restaurant booked?」', {
    x: 1.02, y: 1.96, w: 5.3, h: 0.5, fontSize: 10, color: INK2, lineSpacingMultiple: 1.3, fontFace: MONO,
  });
  txt(s, '25 个数字槽位样本中只答对 5 个（0.20）', { x: 1.02, y: 2.52, w: 5.3, h: 0.3, fontSize: 11, bold: true, color: BRICK });
  txt(s, '原因：问句自带数字，弱模型倾向回显问句本身', { x: 1.02, y: 2.84, w: 5.3, h: 0.3, fontSize: 10, color: INK2 });

  s.addShape(S.roundRect, { x: 6.83, y: 1.5, w: 5.75, h: 1.75, fill: { color: MIST }, line: { color: MIST }, rectRadius: 0.06 });
  txt(s, '填空式', { x: 7.07, y: 1.64, w: 5.3, h: 0.3, fontSize: 11.5, bold: true, color: TEAL });
  txt(s, '「The restaurant table is booked for ___ people.」', {
    x: 7.07, y: 1.96, w: 5.3, h: 0.5, fontSize: 10, color: INK2, lineSpacingMultiple: 1.3, fontFace: MONO,
  });
  txt(s, '同 25 个样本中答对 13 个（0.52）', { x: 7.07, y: 2.52, w: 5.3, h: 0.3, fontSize: 11, bold: true, color: TEAL });
  txt(s, '提升 2.6 倍，且答案可字面匹配', { x: 7.07, y: 2.84, w: 5.3, h: 0.3, fontSize: 10, color: INK2 });

  // 下方：两种任务形式的逐项对照
  kicker(s, '两种形式的逐项对照', 0.78, 3.52, 6.0);
  const head = ['对比项', '问答式', '填空式'].map((t) => ({
    text: t, options: { bold: true, color: PAPER, fill: { color: INK }, fontSize: 10, align: t === '对比项' ? 'left' : 'center' },
  }));
  const rows = [
    ['答案是否出现在问句中', '是（问句含目标数字）', '否（目标位留空）'],
    ['模型可用的捷径', '直接回显问句里的数字', '只能从上下文提取'],
    ['答案匹配方式', '需归一化后模糊匹配', '字面匹配即可'],
    ['测得的绝对准确率', '0.20', '0.52'],
    ['能否区分选择质量', '否（捷径掩盖差异）', '是（差异可归因于上下文）'],
  ].map((r, i) => r.map((c, j) => ({
    text: c,
    options: {
      fontSize: 9.5, align: j === 0 ? 'left' : 'center',
      color: j === 2 && i === 4 ? TEAL : INK2,
      fill: { color: i % 2 ? MIST : PAPER }, bold: i === 4,
    },
  })));
  s.addTable([head, ...rows], {
    x: 0.78, y: 3.86, w: 7.55,
    colW: [3.0, 2.35, 2.2],
    border: { type: 'solid', color: 'E3E7EC', pt: 1 },
    rowH: 0.38, valign: 'middle', fontFace: BODY, margin: 0.05,
  });

  s.addShape(S.roundRect, { x: 8.55, y: 3.86, w: 4.03, h: 2.06, fill: { color: 'FBF3E6' }, line: { color: 'FBF3E6' }, rectRadius: 0.06 });
  txt(s, '为什么这个差别重要', { x: 8.77, y: 3.98, w: 3.6, h: 0.28, fontSize: 10.5, bold: true, color: BRICK });
  txt(s, '若用问答式，0.20 与 0.52 的差距会被误读成「方法不行」；实际是任务形式本身限制了上限。\n\n选择质量的差异只有在填空式下才可观测，因为答案必须从上下文提取。', {
    x: 8.77, y: 4.3, w: 3.6, h: 1.5, fontSize: 9.5, color: INK2, lineSpacingMultiple: 1.34,
  });

  // 表格 3.86 + 6×0.38 = 6.14，结论条要再往下让 0.16
  takeaway(s, '同一批 25 个样本、同一个 backbone，仅改变任务形式，绝对准确率相差 2.6 倍 —— 因此所有绝对数值都必须注明任务形式。', { y: 6.3, h: 0.42, color: SLATE, size: 10 });
}

// ============================================================
// 17b. 模板设计
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第三部分 · 模型与评测' });
  title(s, '填空模板的两条设计约束');

  const cons = [
    ['模板一律不含数字', '30 个模板全部改写为不含目标数字的形式。若模板自带数字，模型会倾向复述模板内容而非从上下文提取。', AMBER],
    ['排除无法字面匹配的值', 'yes / no / dontcare 这类符号值不参与评测 —— 它们与上下文中的表述形式不唯一，字面匹配会系统性低估准确率。', TEAL],
  ];
  cons.forEach(([h, d, col], i) => {
    const y = 1.55 + i * 1.55;
    s.addShape(S.roundRect, { x: 0.78, y, w: 11.8, h: 1.35, fill: { color: MIST }, line: { color: MIST }, rectRadius: 0.06 });
    s.addShape(S.rect, { x: 0.78, y, w: 0.07, h: 1.35, fill: { color: col }, line: { color: col } });
    txt(s, h, { x: 1.1, y: y + 0.16, w: 5.0, h: 0.32, fontSize: 12.5, bold: true, color: INK });
    txt(s, d, { x: 1.1, y: y + 0.54, w: 11.2, h: 0.68, fontSize: 11, color: INK2, lineSpacingMultiple: 1.35 });
  });

  kicker(s, '一个示例', 0.78, 4.78, 4.0, SLATE);
  s.addShape(S.roundRect, { x: 0.78, y: 5.1, w: 11.8, h: 1.0, fill: { color: INK }, line: { color: INK }, rectRadius: 0.05 });
  txt(s, '上下文', { x: 1.0, y: 5.22, w: 1.0, h: 0.26, fontSize: 9.5, color: '8A99B0' });
  txt(s, '…[usr] I need a taxi to the train station.  [sys] Where shall I pick you up?  [usr] At the Hilton. …', {
    x: 1.9, y: 5.22, w: 10.4, h: 0.28, fontSize: 9.5, color: 'D7DEEA', fontFace: MONO,
  });
  txt(s, '问题', { x: 1.0, y: 5.58, w: 1.0, h: 0.26, fontSize: 9.5, color: '8A99B0' });
  txt(s, 'The taxi is departing from ___.　　答案应为 Hilton，但若上下文混入其他对话的「station」则会答错', {
    x: 1.9, y: 5.58, w: 10.4, h: 0.3, fontSize: 9.5, color: AMBER,
  });

  takeaway(s, '任务形式的选择本身会显著影响测得的绝对准确率（0.20 vs 0.52），因此报告里所有绝对数值都注明任务形式。', { y: 6.3, h: 0.32, color: SLATE, size: 10.5 });

  s.addNotes('任务形式影响绝对数值。问答式 0.20、填空式 0.52，差 2.6 倍，所以所有绝对数值都必须注明任务形式，否则不可比。');
}

pres.addSection({ title: '第四部分 · 实验结果' });

// ============================================================
// 18. L1 主表
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第四部分 · 实验结果' });
  title(s, 'L1：固定预算下谁把关键信息选进来了');

  s.addChart(
    pres.ChartType.bar,
    [
      { name: 'B2 打分网络', labels: ['L=128', 'L=256', 'L=512'], values: [1.0, 0.7667, 0.5333] },
      { name: 'B1 仅向量检索', labels: ['L=128', 'L=256', 'L=512'], values: [0.7, 0.4, 0.3333] },
      { name: 'LEN 长度基线', labels: ['L=128', 'L=256', 'L=512'], values: [0.8333, 0.5333, 0.3] },
      { name: 'R0 位置基线', labels: ['L=128', 'L=256', 'L=512'], values: [0.4333, 0.2, 0.2] },
    ],
    {
      x: 0.62, y: 1.45, w: 7.6, h: 4.4,
      barDir: 'col', barGapWidthPct: 55,
      chartColors: [AMBER, TEAL, SLATE, 'C9D0DA'],
      showLegend: true, legendPos: 'b', legendFontSize: 10, legendFontFace: '+mn-lt',
      showValue: true, dataLabelFormatCode: '0.00', dataLabelPosition: 'outEnd', dataLabelFontSize: 9, dataLabelFontFace: '+mn-lt', dataLabelColor: INK2,
      valAxisMinVal: 0, valAxisMaxVal: 1.15, valAxisMajorUnit: 0.25,
      catAxisLabelColor: INK2, catAxisLabelFontSize: 11, catAxisLabelFontFace: '+mn-lt',
      valAxisLabelColor: SLATE, valAxisLabelFontSize: 9.5, valAxisLabelFontFace: '+mn-lt',
      valGridLine: { color: 'E3E7EC', size: 1 }, catGridLine: { style: 'none' },
      showTitle: false,
    }
  );
  txt(s, 'CR：needle 是否被选入上下文　｜　test 90 场景 × 3 长度　｜　B=2048', { x: 0.78, y: 5.86, w: 7.4, h: 0.28, fontSize: 9.5, color: SLATE });

  s.addShape(S.roundRect, { x: 8.45, y: 1.5, w: 4.13, h: 1.95, fill: { color: MIST }, line: { color: MIST }, rectRadius: 0.06 });
  txt(s, '跨长度 CR', { x: 8.67, y: 1.64, w: 3.7, h: 0.26, fontSize: 10, color: SLATE });
  txt(s, '0.7667', { x: 8.67, y: 1.88, w: 3.7, h: 0.56, fontSize: 30, bold: true, color: AMBER, fontFace: HEAD });
  txt(s, '比纯检索高 28.9 个百分点\n比位置基线高 48.9 个百分点\n比最强零成本基线（LEN）高 21.1', { x: 8.67, y: 2.5, w: 3.7, h: 0.85, fontSize: 10, color: INK2, lineSpacingMultiple: 1.3 });

  txt(s, '三条观察', { x: 8.45, y: 3.6, w: 4.13, h: 0.3, fontSize: 11, bold: true, color: INK });
  const obs = [
    '「优势随历史变长而扩大」不成立：B2/B1 比值 1.43× → 1.92× → 1.60×，非单调',
    'B0 与 R1 完全相同 —— 无记忆模块时，全量上下文等价于时间序截断',
    'LEN 意外地强（0.5555），但按长度四分位分层后 AUC 仍 0.86–0.95，内容信号独立有效',
  ];
  obs.forEach((t, i) => {
    txt(s, '· ' + t, { x: 8.45, y: 3.9 + i * 0.62, w: 4.13, h: 0.6, fontSize: 9.5, color: INK2, lineSpacingMultiple: 1.24 });
  });

  takeaway(s, 'B2 在三个长度上全部最优；但领先幅度在 L≥256 后不再增长，L=512 的 CR 0.5333 是最明显的短板。', { y: 6.22, h: 0.3, color: BRICK, size: 10.5 });

  s.addNotes('L1 主表。跨长度 CR 0.7667，比纯检索高 28.9 个百分点。三条观察都要说，尤其第一条负结果。');
}

// ============================================================
// 19. L1 完整表
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第四部分 · 实验结果' });
  title(s, 'L1：十个选择器的完整结果');

  const head = ['选择器', 'L=128', 'L=256', 'L=512', '跨长度 CR', 'Recall', 'NDCG'].map((t) => ({
    text: t, options: { bold: true, color: PAPER, fill: { color: INK }, fontSize: 9.5, align: t === '选择器' ? 'left' : 'center' },
  }));
  const rows = [
    ['B2 打分网络', '1.0000', '0.7667', '0.5333', '0.7667', '0.6051', '0.2114'],
    ['LEN 长度', '0.8333', '0.5333', '0.3000', '0.5555', '0.3620', '0.1271'],
    ['B2R 两阶段', '0.6333', '0.5000', '0.3667', '0.5000', '0.3857', '0.1849'],
    ['B1 仅检索', '0.7000', '0.4000', '0.3333', '0.4778', '0.3649', '0.1541'],
    ['R0 位置', '0.4333', '0.2000', '0.2000', '0.2778', '0.3158', '0.0573'],
    ['R2 随机', '0.5000', '0.1667', '0.0667', '0.2445', '0.2731', '0.0546'],
    ['R3 前 3 轮', '0.4667', '0.1000', '0.0333', '0.2000', '0.2177', '0.1274'],
    ['R1 recency', '0.4000', '0.1000', '0.0000', '0.1667', '0.1947', '0.0409'],
    ['B0 全量截断', '0.4000', '0.1000', '0.0000', '0.1667', '0.1947', '0.0409'],
    ['R0A 绝对轮次', '0.4000', '0.0667', '0.0000', '0.1556', '0.3400', '0.1574'],
  ].map((r, i) => r.map((c, j) => ({
    text: c,
    options: {
      fontSize: 9.5, align: j === 0 ? 'left' : 'center', color: i === 0 ? INK : INK2,
      fill: { color: i === 0 ? 'FBF3E6' : (i % 2 ? MIST : PAPER) }, bold: i === 0,
    },
  })));

  s.addTable([head, ...rows], {
    x: 0.78, y: 1.5, w: 11.8,
    colW: [2.3, 1.55, 1.55, 1.55, 1.75, 1.6, 1.5],
    border: { type: 'solid', color: 'E3E7EC', pt: 1 },
    // 表头 + 10 行 = 11 行，0.34 才落在 5.24
    rowH: 0.34, valign: 'middle', fontFace: BODY, margin: 0.04,
  });

  txt(s, '注：B0 与 R1 数值完全相同（0.4000 / 0.1000 / 0.0000），这是必然而非缺陷 —— 无记忆模块时「全量上下文」在超预算后就是按时间序从头截断。', {
    x: 0.78, y: 5.34, w: 11.8, h: 0.44, fontSize: 9.5, color: SLATE, lineSpacingMultiple: 1.28,
  });

  s.addShape(S.roundRect, { x: 0.78, y: 5.95, w: 11.8, h: 0.6, fill: { color: 'FBF3E6' }, line: { color: 'FBF3E6' }, rectRadius: 0.05 });
  txt(s, 'LEN 基线的自查', { x: 1.0, y: 6.05, w: 1.8, h: 0.28, fontSize: 10.5, bold: true, color: BRICK });
  txt(s, 'corr(score, n_tokens) = +0.6331，确实与长度相关；但按长度四分位分层后 AUC 为 0.9464 / 0.9084 / 0.8675 / 0.8556，说明内容信号独立有效，不是长度信号的复现。', {
    x: 2.85, y: 6.03, w: 9.5, h: 0.44, fontSize: 9.5, color: INK2, lineSpacingMultiple: 1.26,
  });

  s.addNotes('完整表给出十个选择器。两个需要解释的现象：B0 与 R1 数值完全相同是必然；LEN 很强但分层后 AUC 仍高，说明内容信号独立。');
}

// ============================================================
// 20. 预算扫描
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第四部分 · 实验结果' });
  title(s, '预算扫描：压缩越狠优势越大，但有例外');

  s.addChart(
    pres.ChartType.bar,
    [
      { name: 'B2 打分网络', labels: ['B=256', 'B=512', 'B=1024', 'B=2048'], values: [0.05, 0.225, 0.4, 0.625] },
      { name: 'B1 仅向量检索', labels: ['B=256', 'B=512', 'B=1024', 'B=2048'], values: [0.075, 0.1, 0.275, 0.275] },
      { name: 'R0 位置基线', labels: ['B=256', 'B=512', 'B=1024', 'B=2048'], values: [0.025, 0.05, 0.075, 0.1] },
    ],
    {
      x: 0.62, y: 1.45, w: 7.4, h: 4.2,
      barDir: 'col', barGapWidthPct: 40,
      chartColors: [AMBER, TEAL, 'C9D0DA'],
      showLegend: true, legendPos: 'b', legendFontSize: 10, legendFontFace: '+mn-lt',
      showValue: true, dataLabelFormatCode: '0.00', dataLabelPosition: 'outEnd', dataLabelFontSize: 9, dataLabelFontFace: '+mn-lt', dataLabelColor: INK2,
      valAxisMinVal: 0, valAxisMaxVal: 0.75, valAxisMajorUnit: 0.15,
      catAxisLabelColor: INK2, catAxisLabelFontSize: 10.5, catAxisLabelFontFace: '+mn-lt',
      valAxisLabelColor: SLATE, valAxisLabelFontSize: 9.5, valAxisLabelFontFace: '+mn-lt',
      valGridLine: { color: 'E3E7EC', size: 1 }, catGridLine: { style: 'none' },
      showTitle: false,
    }
  );
  txt(s, '横轴：token 预算　｜　纵轴：L=512 时的 CR（库 16,874 token）', { x: 0.78, y: 5.72, w: 7.2, h: 0.3, fontSize: 9.5, color: SLATE });

  txt(s, '结论需分开陈述', { x: 8.3, y: 1.5, w: 4.28, h: 0.3, fontSize: 11, bold: true, color: INK });

  s.addShape(S.roundRect, { x: 8.3, y: 1.85, w: 4.28, h: 1.45, fill: { color: MIST }, line: { color: MIST }, rectRadius: 0.06 });
  txt(s, 'B=1024 / 2048 的 6 个格子 B2 全部第一', { x: 8.52, y: 1.98, w: 3.9, h: 0.32, fontSize: 10.5, bold: true, color: TEAL });
  txt(s, '且压缩越激进优势越大。L=512、B=1024 时 0.400 vs 0.275 vs 0.075。', { x: 8.52, y: 2.32, w: 3.9, h: 0.9, fontSize: 9.5, color: INK2, lineSpacingMultiple: 1.3 });

  s.addShape(S.roundRect, { x: 8.3, y: 3.42, w: 4.28, h: 1.45, fill: { color: 'FBF0EF' }, line: { color: 'FBF0EF' }, rectRadius: 0.06 });
  txt(s, 'B=256 的 3 个格子里 2 个 B2 输给 B1', { x: 8.52, y: 3.55, w: 3.9, h: 0.32, fontSize: 10.5, bold: true, color: BRICK });
  txt(s, '只够放 8–9 个片段时，稠密打分器排序精度不如余弦。打分器学「片段整体重要性」，余弦直接对齐 query。', { x: 8.52, y: 3.89, w: 3.9, h: 0.9, fontSize: 9.5, color: INK2, lineSpacingMultiple: 1.3 });

  s.addShape(S.roundRect, { x: 8.3, y: 4.99, w: 4.28, h: 1.25, fill: { color: INK }, line: { color: INK }, rectRadius: 0.06 });
  txt(s, '推荐配置', { x: 8.52, y: 5.12, w: 3.9, h: 0.26, fontSize: 9.5, color: AMBER, bold: true });
  txt(s, 'L=512 时 B=2048（保留 12.1%），CR 0.625，相对 B1 的 0.275 是 2.3× 提升。', { x: 8.52, y: 5.4, w: 3.9, h: 0.72, fontSize: 9.5, color: 'D7DEEA', lineSpacingMultiple: 1.3 });

  takeaway(s, 'L=512、B≤512 时所有方法都接近 0（保留 1.5%–3.0%），此时任何选择器都救不回来 —— 这是「关键信息在极低预算下无法保障」的负结果。', { y: 6.24, h: 0.44, color: SLATE, size: 9.5 });

  s.addNotes('预算扫描。正面：B≥1024 时 B2 全部第一且压缩越狠优势越大。负面：B=256 时 2/3 格子输给余弦，这是真实边界。');
}

// ============================================================
// 21. 消融
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第四部分 · 实验结果' });
  title(s, '消融：三个超参都没有显著影响');

  const head = ['因素', '取值', 'test AUC', '标准差', '参数量'].map((t) => ({
    text: t, options: { bold: true, color: PAPER, fill: { color: INK }, fontSize: 10.5, align: 'center' },
  }));
  const rows = [
    ['网络层数', '1', '0.9352', '0.0007', '99,585'],
    ['', '2', '0.9340', '0.0007', '165,377'],
    ['', '3', '0.9341', '0.0003', '231,169'],
    ['', '4', '0.9339', '0.0001', '296,961'],
    ['学习率', '1e-4 … 3e-3', '0.9336 – 0.9349', '≤ 0.0008', '165,377'],
    ['L2 正则', '0 / 1e-5 / 1e-4 / 1e-3', '0.9340（四值全同）', '0.0007', '165,377'],
  ].map((r, i) => r.map((c, j) => ({
    text: c,
    options: {
      fontSize: 10.5, align: j === 0 ? 'left' : 'center', color: INK2,
      fill: { color: i === 0 ? 'FBF3E6' : (i % 2 ? MIST : PAPER) }, bold: i === 0 && j !== 0,
    },
  })));

  s.addTable([head, ...rows], {
    x: 0.78, y: 1.5, w: 7.3,
    colW: [1.35, 1.95, 1.85, 1.05, 1.1],
    border: { type: 'solid', color: 'E3E7EC', pt: 1 },
    rowH: 0.4, valign: 'middle', fontFace: BODY, margin: 0.05,
  });
  txt(s, '位置基线 0.8172 作为参照（未列入表中，各配置与它的差距均在 +0.116 以上）', { x: 0.78, y: 4.68, w: 7.3, h: 0.28, fontSize: 9.5, color: SLATE });

  statCard(s, { x: 8.35, y: 1.5, w: 2.1, h: 1.45, value: '0.0013', label: 'AUC 最大跨度' });
  statCard(s, { x: 10.6, y: 1.5, w: 1.98, h: 1.45, value: '0.0007', label: '种子间标准差', accent: SLATE });

  txt(s, '同量级 ⇒ 无显著差异', { x: 8.35, y: 3.08, w: 4.23, h: 0.3, fontSize: 11, bold: true, color: TEAL });

  const notes = [
    ['层数', '1 层反而最高（0.9352）。题目写「浅层」不是妥协，是正解 —— 加深只增参数。'],
    ['学习率', '1e-4 略优，但 3e-3 到 1e-4 全落在 0.9336–0.9349 窄带内，被种子噪声淹没。'],
    ['L2 正则', '四个取值小数点后四位完全相同，标准差也相同 —— 早停在 epoch 5 先触发，正则没机会起作用。'],
  ];
  notes.forEach(([k, v], i) => {
    txt(s, k, { x: 8.35, y: 3.46 + i * 0.92, w: 4.23, h: 0.26, fontSize: 10.5, bold: true, color: AMBER });
    txt(s, v, { x: 8.35, y: 3.72 + i * 0.92, w: 4.23, h: 0.66, fontSize: 9.5, color: INK2, lineSpacingMultiple: 1.24 });
  });

  takeaway(s, 'L2 正则那条是本消融最有信息量的一点：结论「无影响」在当前设置下是平凡成立的，不能外推为「L2 正则对本任务无用」。', { y: 6.1, h: 0.42, color: SLATE, size: 9.5 });

  s.addNotes('三个超参都无显著影响：最大跨度 0.0013 与种子标准差 0.0007 同量级。层数那条最有意思：1 层最高。L2 正则那条要说明是早停先触发。');
}

// ============================================================
// 22. L2 主表
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第四部分 · 实验结果' });
  title(s, 'L2：任务完成率（题目要求的三个指标）');

  s.addChart(
    pres.ChartType.bar,
    [{
      name: 'slot_acc',
      labels: ['B2 打分', 'B1 检索', 'R1 recency', 'B0 截断', 'R2 随机', 'R0 位置', 'ORACLE'],
      values: [0.3778, 0.3556, 0.3278, 0.3167, 0.3056, 0.2111, 0.8444],
    }],
    {
      x: 0.62, y: 1.45, w: 7.3, h: 4.2,
      barDir: 'col', barGapWidthPct: 45,
      chartColors: [AMBER, TEAL, SLATE, 'C9D0DA', 'C9D0DA', 'C9D0DA', INK],
      showLegend: false,
      showValue: true, dataLabelFormatCode: '0.00', dataLabelPosition: 'outEnd', dataLabelFontSize: 9.5, dataLabelFontFace: '+mn-lt', dataLabelColor: INK2,
      valAxisMinVal: 0, valAxisMaxVal: 1.0, valAxisMajorUnit: 0.2,
      catAxisLabelColor: INK2, catAxisLabelFontSize: 9.5, catAxisLabelFontFace: '+mn-lt',
      valAxisLabelColor: SLATE, valAxisLabelFontSize: 9.5, valAxisLabelFontFace: '+mn-lt',
      valGridLine: { color: 'E3E7EC', size: 1 }, catGridLine: { style: 'none' },
      showTitle: false,
    }
  );
  txt(s, '90 场景 / 180 槽位　｜　B=2048　｜　DeepSeek-V4.1-Flash　｜　定向填空', { x: 0.78, y: 5.68, w: 7.1, h: 0.28, fontSize: 9.5, color: SLATE });

  s.addShape(S.roundRect, { x: 8.2, y: 1.45, w: 4.38, h: 1.8, fill: { color: MIST }, line: { color: MIST }, rectRadius: 0.06 });
  txt(s, '压缩 4.9 倍，指标还涨 5.6 个百分点', { x: 8.42, y: 1.58, w: 4.0, h: 0.46, fontSize: 11, bold: true, color: TEAL, lineSpacingMultiple: 1.2 });
  txt(s, 'B2：1,489 token → 0.3778\nB0FULL：7,277 token → 0.3222', { x: 8.42, y: 2.08, w: 4.0, h: 0.58, fontSize: 10, color: INK2, lineSpacingMultiple: 1.4 });
  txt(s, '对题目「控制 token 开销的前提下保障关键信息」的直接回答', { x: 8.42, y: 2.7, w: 4.0, h: 0.42, fontSize: 9.5, color: SLATE, lineSpacingMultiple: 1.24 });

  txt(s, '配对比较（bootstrap 5,000 次）', { x: 8.2, y: 3.42, w: 4.38, h: 0.28, fontSize: 10.5, bold: true, color: INK });
  const pairs = [
    ['B2 − B1', '+0.0222', 'p=0.6964　不显著', BRICK],
    ['B2 − B0', '+0.0611', 'p=0.1948　不显著', SLATE],
    ['B2 − R0', '+0.1667', 'p<0.0001　显著', TEAL],
    ['B2 − R1', '+0.0500', 'p=0.3092　不显著', SLATE],
    ['B2 − R2', '+0.0722', 'p=0.0984　边缘', SLATE],
  ];
  pairs.forEach(([a, b, c, col], i) => {
    txt(s, a, { x: 8.2, y: 3.74 + i * 0.4, w: 1.2, h: 0.28, fontSize: 9.5, color: INK2 });
    txt(s, b, { x: 9.4, y: 3.74 + i * 0.4, w: 1.0, h: 0.28, fontSize: 9.5, bold: true, color: INK });
    txt(s, c, { x: 10.45, y: 3.74 + i * 0.4, w: 2.1, h: 0.28, fontSize: 9.5, color: col });
  });

  takeaway(s, 'B2 唯一显著赢的是位置基线。对纯检索只高 2.2 个百分点且不显著 —— 180 条样本的统计功效不足以分辨这个量级。CR 高不等于答对率高。', { y: 5.95, h: 0.62, color: BRICK, size: 10 });

  s.addNotes('L2 主表是题目要求的三指标。正面结论是压缩 4.9 倍指标反升；必须如实说明 B2 对 B1 不显著。');
}

// ============================================================
// 23. L2 分长度
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第四部分 · 实验结果' });
  title(s, 'L2：分长度结果与一个反超');

  const head = ['L', '选择器', 'slot_acc', '场景成功率', 'ctx token', 'CR'].map((t) => ({
    text: t, options: { bold: true, color: PAPER, fill: { color: INK }, fontSize: 9.5, align: t === '选择器' ? 'left' : 'center' },
  }));
  const rows = [
    ['128', 'B2', '0.5000', '0.2333', '1,496', '1.0000'],
    ['128', 'B1', '0.4500', '0.1667', '1,470', '0.7000'],
    ['128', 'B0', '0.4000', '0.2000', '1,475', '0.4000'],
    ['128', 'ORACLE', '0.8500', '0.7333', '216', '1.0000'],
    ['128', 'B0FULL', '0.3833', '0.1000', '3,242', '1.0000'],
    ['256', 'B2', '0.3333', '0.1333', '1,497', '0.7667'],
    ['256', 'B1', '0.3167', '0.0667', '1,456', '0.4000'],
    ['256', 'B0', '0.3000', '0.1000', '1,487', '0.1000'],
    ['256', 'ORACLE', '0.8500', '0.7333', '216', '1.0000'],
    ['256', 'B0FULL', '0.3167', '0.0000', '6,473', '1.0000'],
    ['512', 'B1', '0.3478', '0.0870', '1,421', '0.3913'],
    ['512', 'B2', '0.3043', '0.1304', '1,464', '0.4783'],
    ['512', 'B0', '0.2826', '0.0870', '1,475', '0.0000'],
    ['512', 'ORACLE', '0.8478', '0.7391', '246', '1.0000'],
    ['512', 'B0FULL', '0.3043', '0.0435', '12,406', '1.0000'],
  ].map((r, i) => r.map((c, j) => {
    const isB2L512 = r[1] === 'B2' && r[0] === '512';
    const isB1L512 = r[1] === 'B1' && r[0] === '512';
    return {
      text: c,
      options: {
        fontSize: 8.8, align: j === 1 ? 'left' : 'center',
        color: (isB1L512 && j === 2) ? BRICK : (r[1] === 'ORACLE' ? INK : INK2),
        fill: { color: r[1] === 'ORACLE' ? 'FBF3E6' : (isB2L512 ? 'FDF6EA' : (i % 2 ? MIST : PAPER)) },
        bold: (r[1] === 'B2' || r[1] === 'ORACLE' || isB1L512) && j <= 2,
      },
    };
  }));

  s.addTable([head, ...rows], {
    x: 0.78, y: 1.42, w: 7.55,
    colW: [0.7, 1.55, 1.3, 1.6, 1.4, 1.0],
    border: { type: 'solid', color: 'E3E7EC', pt: 1 },
    rowH: 0.31, valign: 'middle', fontFace: BODY, margin: 0.04,
  });

  s.addShape(S.roundRect, { x: 8.55, y: 1.42, w: 4.03, h: 1.95, fill: { color: 'FBF0EF' }, line: { color: 'FBF0EF' }, rectRadius: 0.06 });
  txt(s, 'L=512 时 B1 反超 B2', { x: 8.77, y: 1.56, w: 3.6, h: 0.3, fontSize: 11, bold: true, color: BRICK });
  txt(s, 'B1 的 slot_acc 0.3478 高于 B2 的 0.3043。\n但 B2 的场景成功率更高（0.1304 vs 0.0870）—— 单槽位更易错，多槽位场景反而更稳。', {
    x: 8.77, y: 1.9, w: 3.6, h: 1.2, fontSize: 9.5, color: INK2, lineSpacingMultiple: 1.32,
  });

  s.addShape(S.roundRect, { x: 8.55, y: 3.5, w: 4.03, h: 1.5, fill: { color: MIST }, line: { color: MIST }, rectRadius: 0.06 });
  txt(s, '原因', { x: 8.77, y: 3.62, w: 3.6, h: 0.28, fontSize: 10, bold: true, color: INK });
  txt(s, '长历史下 B2 的 CR 只有 0.4783，与 B1 的 0.3913 差距收窄，选择优势不足以抵消干扰轮带来的噪声。', {
    x: 8.77, y: 3.92, w: 3.6, h: 1.0, fontSize: 9.5, color: INK2, lineSpacingMultiple: 1.3,
  });

  s.addShape(S.roundRect, { x: 8.55, y: 5.12, w: 4.03, h: 1.15, fill: { color: INK }, line: { color: INK }, rectRadius: 0.06 });
  txt(s, 'ORACLE 的关键性质', { x: 8.77, y: 5.22, w: 3.6, h: 0.26, fontSize: 9.5, color: AMBER, bold: true });
  txt(s, '三个长度上都是 0.85 / 0.72，token 几乎不变（216 / 216 / 246）—— 任务难度不来自「历史长」，而来自「历史里有别的东西」。', {
    x: 8.77, y: 5.5, w: 3.6, h: 0.7, fontSize: 9, color: 'D7DEEA', lineSpacingMultiple: 1.28,
  });

  takeaway(s, 'B0FULL 在三个长度上的 token 分别是 3,242 / 6,473 / 12,406，是 ORACLE 的 15–50 倍，而准确率始终更低。', { y: 6.42, h: 0.3, color: SLATE, size: 9.5 });

  s.addNotes('分长度结果。L=512 时 B1 反超 B2，这是必须主动说明的负结果。ORACLE 的 token 不随长度变化是关键观察。');
}

// ============================================================
// 24. ORACLE 关键发现
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第四部分 · 实验结果' });
  title(s, '关键发现：瓶颈不在网络，在对话归属');

  const cards = [
    ['216', 'token', 'ORACLE：只给本对话真实历史', '0.8444', INK, PAPER],
    ['1,489', 'token', 'B2 打分网络 + 预算 2048', '0.3778', AMBER, PAPER],
    ['7,277', 'token', 'B0FULL：全量上下文（题目字面义）', '0.3222', SLATE, PAPER],
  ];
  cards.forEach(([v, u, label, acc, bg, fg], i) => {
    const x = 0.78 + i * 3.42;
    s.addShape(S.roundRect, { x, y: 1.45, w: 3.2, h: 2.0, fill: { color: bg }, line: { color: bg }, rectRadius: 0.06 });
    txt(s, v, { x: x + 0.24, y: 1.58, w: 2.7, h: 0.58, fontSize: 27, bold: true, color: i === 0 ? AMBER : INK, fontFace: HEAD });
    txt(s, u + '  →  slot_acc ' + acc, { x: x + 0.24, y: 2.18, w: 2.75, h: 0.3, fontSize: 10.5, bold: true, color: i === 0 ? PAPER : INK });
    txt(s, label, { x: x + 0.24, y: 2.52, w: 2.75, h: 0.8, fontSize: 9.5, color: i === 0 ? 'B9C4D6' : INK, lineSpacingMultiple: 1.3 });
  });

  kicker(s, '46.6 个百分点是什么', 0.78, 3.62, 5.6);
  txt(s, '不是「方法还不够好」，而是「任务本身有多难」的上界。\n\n若能把跨对话的干扰轮完美剔除，同样 216 token 就能到 84.4%。中间那 46.6 个百分点，就是「区分对话归属」这件事的价值。', {
    x: 0.78, y: 3.94, w: 5.6, h: 1.75, fontSize: 10.5, color: INK2, lineSpacingMultiple: 1.42,
  });

  kicker(s, 'token 消耗的实际含义', 6.8, 3.62, 5.78);
  const tk = [
    ['ORACLE', '216', '3.0%', '0.8444'],
    ['NEEDLE', '39', '0.5%', '0.3444'],
    ['B2', '1,489', '20.5%', '0.3778'],
    ['B1', '1,458', '20.0%', '0.3556'],
    ['B0FULL', '7,277', '100%', '0.3222'],
  ];
  // 表头逐列定位：用空格排版在比例字体下对不齐，四列必须各自设 x 与对齐方式
  txt(s, '条件', { x: 6.8, y: 3.9, w: 1.35, h: 0.26, fontSize: 8.5, color: SLATE });
  txt(s, 'token', { x: 8.2, y: 3.9, w: 1.0, h: 0.26, fontSize: 8.5, color: SLATE, align: 'right' });
  txt(s, '占 B0FULL', { x: 9.4, y: 3.9, w: 1.0, h: 0.26, fontSize: 8.5, color: SLATE, align: 'right' });
  txt(s, 'slot_acc', { x: 10.6, y: 3.9, w: 1.0, h: 0.26, fontSize: 8.5, color: SLATE, align: 'right' });
  tk.forEach(([n, t, p, a], i) => {
    const y = 4.16 + i * 0.38;
    txt(s, n, { x: 6.8, y, w: 1.35, h: 0.28, fontSize: 9.5, color: i === 2 ? AMBER : INK2, bold: i === 2 });
    txt(s, t, { x: 8.2, y, w: 1.0, h: 0.28, fontSize: 9.5, color: INK, align: 'right' });
    txt(s, p, { x: 9.4, y, w: 1.0, h: 0.28, fontSize: 9.5, color: SLATE, align: 'right' });
    txt(s, a, { x: 10.6, y, w: 1.0, h: 0.28, fontSize: 9.5, color: i === 2 ? AMBER : INK2, align: 'right', bold: i === 2 });
  });
  takeaway(s, 'ORACLE 是作弊基线（真实系统没有 dialogue_id 信息，若有了问题就不存在）。它的价值是给出上界，而非可达到的性能。', { y: 6.28, h: 0.32, color: SLATE, size: 9.5 });

  s.addNotes('这是全场最重要的一页。三张卡对比：ORACLE 216 token 达 0.8444，B2 1,489 token 0.3778，B0FULL 7,277 token 0.3222。46.6 个百分点是任务难度上界，不是方法缺陷。');
}

// ============================================================
// 25. 证据：不是没选到
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第四部分 · 实验结果' });
  title(s, '证据：不是「没选到」，是「被带偏」');

  kicker(s, '（1）先排除「选不到」这个解释', 0.78, 1.42, 6.0, TEAL);
  const head = ['场景', 'bank', '主对话段', 'B2 选中', '含主对话', 'needle'].map((t) => ({
    text: t, options: { bold: true, color: PAPER, fill: { color: INK }, fontSize: 9, align: 'center' },
  }));
  const rows = [
    ['MUL0003', '128', '6 段 / 296 tok', '48 段 / 2,010 tok', '5/6', '选中'],
    ['MUL0004', '128', '5 段 / 231 tok', '46 段 / 2,022 tok', '5/5', '选中'],
    ['MUL0011', '128', '11 段 / 411 tok', '45 段 / 2,041 tok', '8/11', '选中'],
    ['MUL0014', '128', '6 段 / 302 tok', '49 段 / 2,047 tok', '6/6', '选中'],
  ].map((r, i) => r.map((c, j) => ({
    text: c,
    options: { fontSize: 9, align: j === 0 ? 'left' : 'center', color: INK2, fill: { color: i % 2 ? MIST : PAPER } },
  })));

  s.addTable([head, ...rows], {
    x: 0.78, y: 1.74, w: 7.3,
    colW: [1.3, 0.85, 1.6, 1.85, 1.05, 0.65],
    border: { type: 'solid', color: 'E3E7EC', pt: 1 },
    rowH: 0.36, valign: 'middle', fontFace: BODY, margin: 0.04,
  });
  txt(s, '30 个 L=128 场景汇总：主对话段只被漏选 36 段，needle 全部选中。选得并不差。', {
    x: 0.78, y: 3.62, w: 7.3, h: 0.3, fontSize: 10, color: TEAL, bold: true,
  });

  kicker(s, '（2）真正的原因：同槽位冲突', 0.78, 4.1, 6.0, BRICK);
  txt(s, '30 个 L=128 场景中有 33 个（场景, 槽位）组合的选中上下文包含「同槽位、不同值」的冲突干扰轮：', {
    x: 0.78, y: 4.42, w: 7.3, h: 0.3, fontSize: 10, color: INK2,
  });
  const conf = [
    ['hotel-area', 'gold = centre', '另一段对话说 north', '模型答 north'],
    ['hotel-stars', 'gold = 0', '另一段对话说 4', '模型答 4'],
  ];
  conf.forEach(([slot, gold, decoy, ans], i) => {
    const y = 4.76 + i * 0.44;
    txt(s, slot, { x: 0.78, y, w: 1.5, h: 0.28, fontSize: 9.5, color: INK, bold: true, fontFace: MONO });
    txt(s, gold, { x: 2.35, y, w: 1.4, h: 0.28, fontSize: 9.5, color: TEAL });
    txt(s, decoy, { x: 3.85, y, w: 2.2, h: 0.28, fontSize: 9.5, color: SLATE });
    txt(s, ans, { x: 6.15, y, w: 1.9, h: 0.28, fontSize: 9.5, color: BRICK, bold: true });
  });

  // 右侧：解释
  s.addShape(S.roundRect, { x: 8.45, y: 1.42, w: 4.13, h: 2.3, fill: { color: MIST }, line: { color: MIST }, rectRadius: 0.06 });
  txt(s, '打分器的固有盲区', { x: 8.67, y: 1.56, w: 3.7, h: 0.3, fontSize: 11, bold: true, color: INK });
  txt(s, '它只能判断「这一轮在它自己的对话里重不重要」，判断不了「这一轮对当前 query 是不是噪声」。\n\n输入只有单轮文本的句向量，看不到对话边界。', {
    x: 8.67, y: 1.9, w: 3.7, h: 1.7, fontSize: 9.5, color: INK2, lineSpacingMultiple: 1.38,
  });

  s.addShape(S.roundRect, { x: 8.45, y: 3.86, w: 4.13, h: 1.4, fill: { color: 'FBF3E6' }, line: { color: 'FBF3E6' }, rectRadius: 0.06 });
  txt(s, 'B0FULL 是最直接的证据', { x: 8.67, y: 3.98, w: 3.7, h: 0.28, fontSize: 10, bold: true, color: BRICK });
  txt(s, '给全量上下文（7,277 token）反而比 B2 更差（0.3222 vs 0.3778），比 ORACLE 低 52 个百分点，token 却是 34 倍。', {
    x: 8.67, y: 4.28, w: 3.7, h: 0.9, fontSize: 9.5, color: INK2, lineSpacingMultiple: 1.3,
  });

  takeaway(s, '上下文越长、干扰越多，强模型也会被带偏。这就是记忆过滤存在的理由：不是为了省 token，而是因为不滤会答错。', { y: 5.5, h: 0.34, color: INK, size: 10.5, bold: true });
  takeaway(s, '改进方向：给打分器加入对话归属特征（dialogue_id 或对话级聚类），预期比调层数、调学习率有用得多。', { y: 5.9, h: 0.34, color: AMBER, size: 10.5 });

  s.addNotes('证据链两步：先排除选不到（只漏 36 段、needle 全中），再指出真实原因（33 个组含同槽位冲突轮）。B0FULL 是最直接证据。');
}

// ============================================================
// 26. 负结果
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第四部分 · 实验结果' });
  title(s, '四处没有跑赢的地方');

  const negs = [
    ['B2 相对 B1 在 L2 上不显著', '+0.0222，95% CI [−0.0667, +0.1111]，p=0.6964。180 条样本的功效不足以分辨这个量级。L1 上差 28.9 个百分点，但 CR 高不等于答对率高。'],
    ['B=256 时打分器不如余弦', '只够放 8–9 个片段时，3 个格子里有 2 个 B2 输给 B1。打分器学「片段整体重要性」，余弦直接对齐 query，极低预算下后者更不容易选错。'],
    ['L=512 时优势收窄并出现反超', 'CR 只有 0.5333（vs B1 0.3913），L2 上 B1 的 slot_acc 0.3478 反超 B2 的 0.3043。长历史仍是短板。'],
    ['「删噪声有益」在强模型上消失', '0.5B 上是 −0.075 且显著；换 DeepSeek-V4.1-Flash 后是 −0.0082，p=0.49 不显著。弱模型上的结论不能外推。'],
  ];
  negs.forEach(([h, d], i) => {
    const y = 1.5 + i * 1.2;
    s.addShape(S.roundRect, { x: 0.78, y: y + 0.04, w: 0.32, h: 0.32, fill: { color: 'FBF0EF' }, line: { color: 'FBF0EF' }, rectRadius: 0.04 });
    txt(s, String(i + 1), { x: 0.78, y: y + 0.08, w: 0.32, h: 0.26, fontSize: 11, bold: true, color: BRICK, align: 'center' });
    txt(s, h, { x: 1.26, y, w: 11.3, h: 0.34, fontSize: 12.5, bold: true, color: INK });
    txt(s, d, { x: 1.26, y: y + 0.36, w: 11.3, h: 0.66, fontSize: 10, color: INK2, lineSpacingMultiple: 1.3 });
  });

  takeaway(s, '这些结果限定了方法的适用范围：在预算充足（B≥1024）且历史中等（L≤256）时优势明确；预算极低或历史很长时优势消失。', { y: 6.3, h: 0.32, color: SLATE, size: 10 });

  s.addNotes('四处负结果。这一页专门放负结果，因为它们限定了方法的适用范围，也是后续改进的入口。');
}

pres.addSection({ title: '第五部分 · 校验与讨论' });

// ============================================================
// 27. 标签校验：三个口径总览
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第五部分 · 校验与讨论' });
  title(s, '标签可信吗：三个独立口径');

  txt(s, '如果标签本身是错的，前面所有结论都不成立。因此用三种互相独立的方式交叉验证。', {
    x: 0.78, y: 1.42, w: 11.8, h: 0.32, fontSize: 11, color: INK2,
  });

  const cols = [
    ['口径一', '反事实删除（覆盖率）', '10.56×', '同分母配对下，删重要轮的信息丢失率是删不重要轮的倍数', 'test 10.56× · dev 13.23× · train 10.71×', AMBER],
    ['口径二', 'JGA（LLM 答题）', '−0.6755', '删重要轮后答对率从 0.9174 掉到 0.2477（p<0.001）', '607 段 / 1,332 槽位 / DeepSeek-V4.1-Flash', TEAL],
    ['口径三', '模拟抽检（Cohen κ）', '0.43–0.50', '标签与两个规则判据达到中等一致', 'κ(J2,J3)=0.707，说明判据本身稳定', SLATE],
  ];
  cols.forEach(([k, name, big, desc, foot, col], i) => {
    const x = 0.78 + i * 4.05;
    s.addShape(S.roundRect, { x, y: 1.85, w: 3.85, h: 3.75, fill: { color: MIST }, line: { color: MIST }, rectRadius: 0.06 });
    txt(s, k, { x: x + 0.24, y: 2.0, w: 3.4, h: 0.26, fontSize: 9.5, color: col, bold: true, charSpacing: 1.2 });
    txt(s, name, { x: x + 0.24, y: 2.26, w: 3.4, h: 0.34, fontSize: 11.5, bold: true, color: INK });
    txt(s, big, { x: x + 0.24, y: 2.68, w: 3.4, h: 0.66, fontSize: 26, bold: true, color: col, fontFace: HEAD });
    txt(s, desc, { x: x + 0.24, y: 3.4, w: 3.4, h: 1.0, fontSize: 10, color: INK2, lineSpacingMultiple: 1.35 });
    txt(s, foot, { x: x + 0.24, y: 4.5, w: 3.4, h: 0.9, fontSize: 9, color: SLATE, lineSpacingMultiple: 1.3 });
  });

  s.addShape(S.roundRect, { x: 0.78, y: 5.75, w: 11.8, h: 0.72, fill: { color: INK }, line: { color: INK }, rectRadius: 0.05 });
  txt(s, '两个限定', { x: 1.0, y: 5.86, w: 1.1, h: 0.26, fontSize: 9.5, color: AMBER, bold: true });
  txt(s, '① κ 是模拟值而非人工校验（无时间标注，已导出 60 条 CSV 待填）　② 标签标的是「引入」而非「必要」，49.2% 轮次在集合覆盖下可删', {
    x: 2.15, y: 5.84, w: 10.2, h: 0.5, fontSize: 9.5, color: 'D7DEEA', lineSpacingMultiple: 1.26,
  });

  s.addNotes('三个口径方向一致才敢用这个标签。两个限定必须同时说明：κ 是模拟值、标签是「引入」不是「必要」。');
}

// ============================================================
// 28. 反事实删除
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第五部分 · 校验与讨论' });
  title(s, '口径一：反事实删除（覆盖率）');

  txt(s, '做法：分别删除「被标为重要」与「被标为不重要」的轮次，观察下游信息丢失率的差异。若标签有效，前者应显著更高。', {
    x: 0.78, y: 1.42, w: 11.8, h: 0.34, fontSize: 10.5, color: INK2,
  });

  statCard(s, { x: 0.78, y: 1.88, w: 2.72, h: 1.5, value: '10.56×', label: 'test 同分母配对倍数', accent: AMBER });
  statCard(s, { x: 3.7, y: 1.88, w: 2.72, h: 1.5, value: '13.23×', label: 'dev', accent: TEAL });
  statCard(s, { x: 6.62, y: 1.88, w: 2.72, h: 1.5, value: '10.71×', label: 'train', accent: SLATE });
  statCard(s, { x: 9.54, y: 1.88, w: 3.04, h: 1.5, value: '0.49', label: '可删比例（集合覆盖）', accent: INK });

  kicker(s, '为什么强调「同分母配对」', 0.78, 3.62, 6.0);
  txt(s, '不同条件下的可删总量不同，直接比绝对丢失率会被分母差异污染。因此只在「两个条件下都能删」的轮次子集内比较，使分子分母同时受控。', {
    x: 0.78, y: 3.94, w: 5.6, h: 0.8, fontSize: 10, color: INK2, lineSpacingMultiple: 1.34,
  });

  kicker(s, '三层位置控制', 6.8, 3.62, 5.78, TEAL);
  txt(s, '9.74× / 10.56× 是两个不同分层下的倍数，分别控制「对话内位置」与「绝对轮次」两个混淆变量。控制后倍数仍稳定在 10 倍量级，说明结论不来自位置。', {
    x: 6.8, y: 3.94, w: 5.78, h: 0.8, fontSize: 10, color: INK2, lineSpacingMultiple: 1.34,
  });

  s.addShape(S.roundRect, { x: 0.78, y: 4.95, w: 11.8, h: 1.15, fill: { color: 'FBF3E6' }, line: { color: 'FBF3E6' }, rectRadius: 0.05 });
  txt(s, '这个口径能证明什么', { x: 1.0, y: 5.08, w: 2.0, h: 0.28, fontSize: 10.5, bold: true, color: BRICK });
  txt(s, '能证明：被标为重要的轮次确实携带更多下游必需的信息，删除它们损失更大。\n不能证明：这些轮次全都是「必需」的 —— 覆盖率 49.2% 正说明其中有大量冗余，这是标签定义（引入 vs 必要）的直接后果。', {
    x: 3.05, y: 5.06, w: 9.3, h: 0.9, fontSize: 9.5, color: INK2, lineSpacingMultiple: 1.32,
  });

  takeaway(s, '三个 split 上倍数一致（10.56 / 13.23 / 10.71），说明这不是某一划分的偶然。', { y: 6.24, h: 0.32, color: SLATE, size: 10 });

  s.addNotes('反事实删除。关键在同分母配对：不同条件可删总量不同，直接比绝对丢失率会被分母污染。三个 split 倍数一致是稳健性证据。');
}

// ============================================================
// 29. JGA
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第五部分 · 校验与讨论' });
  title(s, '口径二：JGA（LLM 逐槽位答题）');

  txt(s, '做法：对每段对话的每个槽位，让 LLM 仅依据给定上下文作答，统计答对率。分别测试「保留全部」「删重要轮」「删不重要轮」「随机删」四种条件。', {
    x: 0.78, y: 1.42, w: 11.8, h: 0.34, fontSize: 10.5, color: INK2,
  });

  const head = ['条件', '槽位级准确率', '可答子集', '对话级 JGA'].map((t) => ({
    text: t, options: { bold: true, color: PAPER, fill: { color: INK }, fontSize: 10, align: t === '条件' ? 'left' : 'center' },
  }));
  const rows = [
    ['full（不删）', '0.9174', '1.0000', '0.8270'],
    ['drop_neg（删不重要）', '0.9820', '0.9820', '0.8353'],
    ['drop_rand（随机删）', '0.7332', '0.7332', '0.5338'],
    ['drop_pos（删重要）', '0.2676', '0.2676', '0.1598'],
  ].map((r, i) => r.map((c, j) => ({
    text: c,
    options: {
      fontSize: 10, align: j === 0 ? 'left' : 'center',
      color: i === 3 ? BRICK : INK2,
      fill: { color: i === 3 ? 'FBF0EF' : (i % 2 ? MIST : PAPER) }, bold: i === 3,
    },
  })));

  s.addTable([head, ...rows], {
    x: 0.78, y: 1.86, w: 7.2,
    colW: [2.6, 1.65, 1.4, 1.55],
    border: { type: 'solid', color: 'E3E7EC', pt: 1 },
    // 表头 + 4 行 = 5 行，0.38 才落在 3.76
    rowH: 0.38, valign: 'middle', fontFace: BODY, margin: 0.05,
  });
  txt(s, '607 段对话 / 1,332 个槽位 / 5,328 次生成 / 1,234 秒', { x: 0.78, y: 3.84, w: 7.2, h: 0.28, fontSize: 9, color: SLATE });

  s.addShape(S.roundRect, { x: 8.35, y: 1.86, w: 4.23, h: 1.6, fill: { color: MIST }, line: { color: MIST }, rectRadius: 0.06 });
  txt(s, '删重要轮 → 0.2477', { x: 8.57, y: 1.98, w: 3.8, h: 0.3, fontSize: 11, bold: true, color: BRICK });
  txt(s, '配对差值 −0.6755，95% CI [−0.7117, −0.6359]，p<0.001。这是三个口径里最强的一条。', {
    x: 8.57, y: 2.32, w: 3.8, h: 0.9, fontSize: 9.5, color: INK2, lineSpacingMultiple: 1.32,
  });

  s.addShape(S.roundRect, { x: 8.35, y: 3.6, w: 4.23, h: 1.6, fill: { color: 'FBF3E6' }, line: { color: 'FBF3E6' }, rectRadius: 0.06 });
  txt(s, '一个被推翻的旧结论', { x: 8.57, y: 3.72, w: 3.8, h: 0.3, fontSize: 10.5, bold: true, color: BRICK });
  txt(s, '0.5B 时「删噪声有益」显著（full − drop_neg = −0.075）；换强模型后为 −0.0082，p=0.49 不显著。弱模型受益于减少干扰，强模型不受影响。', {
    x: 8.57, y: 4.06, w: 3.8, h: 1.0, fontSize: 9.5, color: INK2, lineSpacingMultiple: 1.3,
  });

  takeaway(s, 'drop_neg 的 0.9820 高于 full 的 0.9174 —— 删掉不重要轮不仅不降，反而略升，说明这些轮次确实是干扰。', { y: 5.42, h: 0.34, color: TEAL, size: 10 });
  takeaway(s, '注意：drop_pos 在可答子集上的 0.2676 说明仍有约四分之一槽位能在缺失关键轮的情况下答对 —— 上下文存在冗余，这也是覆盖率 49.2% 的另一面。', { y: 5.84, h: 0.5, color: SLATE, size: 9.5 });

  s.addNotes('JGA 是三个口径里最强的一条：删重要轮后答对率从 0.9174 掉到 0.2477。drop_neg 反而略升，说明不重要轮确实是干扰。');
}

// ============================================================
// 30. 模拟抽检
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第五部分 · 校验与讨论' });
  title(s, '口径三：模拟抽检（Cohen κ）');

  txt(s, '做法：设计两个与主标签独立但不同的规则判据 J2、J3，在 60 条样本上计算它们与主标签的一致性。这是一致性检验，不是人工校验。', {
    x: 0.78, y: 1.42, w: 11.8, h: 0.34, fontSize: 10.5, color: INK2,
  });

  statCard(s, { x: 0.78, y: 1.88, w: 2.9, h: 1.6, value: '0.430', label: 'κ(标签, J2)', accent: AMBER });
  statCard(s, { x: 3.9, y: 1.88, w: 2.9, h: 1.6, value: '0.500', label: 'κ(标签, J3)', accent: TEAL });
  statCard(s, { x: 7.02, y: 1.88, w: 2.9, h: 1.6, value: '0.707', label: 'κ(J2, J3)', accent: INK });
  s.addShape(S.roundRect, { x: 10.14, y: 1.88, w: 2.44, h: 1.6, fill: { color: MIST }, line: { color: MIST }, rectRadius: 0.06 });
  txt(s, '中等一致', { x: 10.36, y: 2.02, w: 2.0, h: 0.34, fontSize: 14, bold: true, color: INK, fontFace: HEAD });
  txt(s, '按 Landis & Koch 标准，0.41–0.60 属 moderate', { x: 10.36, y: 2.4, w: 2.0, h: 0.9, fontSize: 9, color: INK2, lineSpacingMultiple: 1.3 });

  kicker(s, '怎么读这三个数', 0.78, 3.7, 6.0);
  const read = [
    ['κ(标签, J2)=0.430 与 κ(标签, J3)=0.500', '两个独立判据与主标签都是中等一致 —— 方向一致但不完全重合，符合「弱监督标签」的预期。'],
    ['κ(J2, J3)=0.707', '两个规则判据之间高度一致，说明判据本身稳定、可复现，不是随机波动。'],
    ['若 κ(J2,J3) 也低', '那说明是判据设计有问题，而非标签有偏 —— 这个数是为了排除这种解释。'],
  ];
  read.forEach(([h, d], i) => {
    const y = 4.04 + i * 0.74;
    s.addShape(S.ellipse, { x: 0.8, y: y + 0.06, w: 0.14, h: 0.14, fill: { color: AMBER }, line: { color: AMBER } });
    txt(s, h, { x: 1.08, y, w: 11.3, h: 0.3, fontSize: 10.5, bold: true, color: INK });
    txt(s, d, { x: 1.08, y: y + 0.3, w: 11.3, h: 0.32, fontSize: 9.5, color: INK2, lineSpacingMultiple: 1.26 });
  });

  s.addShape(S.roundRect, { x: 0.78, y: 6.06, w: 11.8, h: 0.56, fill: { color: 'FBF0EF' }, line: { color: 'FBF0EF' }, rectRadius: 0.05 });
  txt(s, '必须说明的限定', { x: 1.0, y: 6.15, w: 1.7, h: 0.28, fontSize: 9.5, bold: true, color: BRICK });
  txt(s, 'κ 是模拟值，不是人工校验。已导出 60 条 CSV 供人工填写，但本轮未完成标注。因此这一口径只能说明「规则之间一致」，不能替代人工金标准。', {
    x: 2.75, y: 6.13, w: 9.6, h: 0.44, fontSize: 9, color: INK2, lineSpacingMultiple: 1.24,
  });

  s.addNotes('模拟抽检是三个口径里最弱的一个，必须明确说明它不是人工校验。κ(J2,J3)=0.707 的作用是排除「判据设计有问题」这个解释。');
}

// ============================================================
// 31. 换标签对照
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第五部分 · 校验与讨论' });
  title(s, '换个标签会更好吗：一次对照实验');

  txt(s, '既然标签标的是「引入」而非「必要」，自然会问：换成「必要」重新训练，指标会不会更好？为此做了一次对照实验。', {
    x: 0.78, y: 1.42, w: 11.8, h: 0.34, fontSize: 10.5, color: INK2,
  });

  const head = ['指标', 'intro（现标签）', 'nec（必要标签）', '差值'].map((t) => ({
    text: t, options: { bold: true, color: PAPER, fill: { color: INK }, fontSize: 10, align: t === '指标' ? 'left' : 'center' },
  }));
  const rows = [
    ['正例率', '55.6%', '32.3%', '—'],
    ['CR（跨长度）', '0.7778', '0.7556', '−0.0222'],
    ['Recall@budget', '0.6370', '0.6210', '−0.0160'],
    ['NDCG@k', '0.2137', '0.2114', '−0.0023'],
    ['配对 p 值', '—', '—', '0.500'],
    ['完全相同的场景', '—', '83.3%', '—'],
  ].map((r, i) => r.map((c, j) => ({
    text: c,
    options: {
      fontSize: 10, align: j === 0 ? 'left' : 'center', color: INK2,
      fill: { color: i % 2 ? MIST : PAPER },
    },
  })));

  s.addTable([head, ...rows], {
    x: 0.78, y: 1.86, w: 7.3,
    colW: [2.3, 1.75, 1.8, 1.45],
    border: { type: 'solid', color: 'E3E7EC', pt: 1 },
    // 表头 + 6 行 = 7 行，0.36 才落在 4.38
    rowH: 0.36, valign: 'middle', fontFace: BODY, margin: 0.05,
  });
  txt(s, '180 个场景上的配对比较，95% CI [−0.083, +0.039]', { x: 0.78, y: 4.52, w: 7.3, h: 0.28, fontSize: 9, color: SLATE });

  s.addShape(S.roundRect, { x: 8.4, y: 1.86, w: 4.18, h: 1.85, fill: { color: MIST }, line: { color: MIST }, rectRadius: 0.06 });
  txt(s, '为什么换标签没有收益', { x: 8.62, y: 1.98, w: 3.75, h: 0.3, fontSize: 10.5, bold: true, color: INK });
  txt(s, '构造场景时 needle 本就优先选 label=1 的轮次，两个标签对 needle 的判定高度重合，nec 没有额外的区分力。', {
    x: 8.62, y: 2.32, w: 3.75, h: 1.2, fontSize: 9.5, color: INK2, lineSpacingMultiple: 1.34,
  });

  s.addShape(S.roundRect, { x: 8.4, y: 3.85, w: 4.18, h: 1.35, fill: { color: 'FBF3E6' }, line: { color: 'FBF3E6' }, rectRadius: 0.06 });
  txt(s, '结论：不改标签', { x: 8.62, y: 3.97, w: 3.75, h: 0.3, fontSize: 10.5, bold: true, color: BRICK });
  txt(s, '力气应该花在 L=512（CR 只有 0.5333）与对话归属特征上，而不是重新定义标签。', {
    x: 8.62, y: 4.31, w: 3.75, h: 0.8, fontSize: 9.5, color: INK2, lineSpacingMultiple: 1.3,
  });

  takeaway(s, '这个对照的价值不在「证明现标签更好」，而在于排除了「换个标签就能大幅提升」这个可能性，避免把时间花错地方。', { y: 5.42, h: 0.34, color: INK, size: 10.5 });
  takeaway(s, '注意 83.3% 的场景结果完全相同 —— 说明两个标签在大多数情况下给出同样的排序，差异集中在少数边界样本上。', { y: 5.86, h: 0.42, color: SLATE, size: 9.5 });

  s.addNotes('换标签对照实验。结论是不改标签：CR 反而略降，p=0.500 不显著，83.3% 场景完全相同。价值在于排除了一种可能性。');
}

// ============================================================
// 32. 结论
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第五部分 · 校验与讨论' });
  title(s, '五条结论');

  const cs = [
    ['打分网络有效', 'test AUC 0.9332 vs 最强零成本基线 0.8172（+0.1160）；L1 跨长度 CR 0.7667 vs 纯检索 0.4778'],
    ['压缩换来的是「同样的 token 装更多有用信息」', '1,489 token / 0.3778 对 7,277 token / 0.3222，压缩 4.9 倍且指标更高'],
    ['瓶颈是对话归属，不是网络容量', 'ORACLE 216 token 即达 0.8444，中间 46.6 个百分点是任务难度上界'],
    ['「浅层」是正解而非妥协', '1 层 99,585 参数最优（0.9352），加深到 4 层参数翻三倍而指标略降'],
    ['标签三重校验方向一致，但有已知偏差', '标的是「引入」而非「必要」，49.2% 轮次可删；位置偏置已用基线控制'],
  ];
  cs.forEach(([h, d], i) => {
    const y = 1.5 + i * 0.95;
    s.addShape(S.ellipse, { x: 0.8, y: y + 0.06, w: 0.2, h: 0.2, fill: { color: i === 2 ? AMBER : SLATE }, line: { color: i === 2 ? AMBER : SLATE } });
    txt(s, h, { x: 1.18, y, w: 11.4, h: 0.34, fontSize: 13, bold: true, color: INK });
    txt(s, d, { x: 1.18, y: y + 0.36, w: 11.4, h: 0.4, fontSize: 10.5, color: INK2, lineSpacingMultiple: 1.28 });
  });

  s.addShape(S.roundRect, { x: 1.18, y: 6.18, w: 11.0, h: 0.5, fill: { color: 'FBF3E6' }, line: { color: 'FBF3E6' }, rectRadius: 0.05 });
  txt(s, '下一步最值得做的：给打分器加入对话归属特征（dialogue_id 或对话级聚类），比调层数、调学习率有用得多。', {
    x: 1.4, y: 6.26, w: 10.6, h: 0.34, fontSize: 10.5, color: BRICK, bold: true,
  });

  s.addNotes('五条结论按重要性排序，第三条是重点：瓶颈是对话归属。最后落到下一步方向。');
}

// ============================================================
// 33. 局限与改进
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第五部分 · 校验与讨论' });
  title(s, '局限与改进方向');

  kicker(s, '四项局限', 0.78, 1.42, 4.0, BRICK);
  const lim = [
    ['L=512 未真正跑满', '90 个场景中只有 23 个真到 512 轮，其余落在 354–486（候选池不足）。长历史结论的强度受此限制。'],
    ['统计功效不足', '180 个槽位对 2.2 个百分点量级的差异无法分辨（B2−B1 p=0.6964）。需要更大规模或更敏感的任务形式。'],
    ['κ 未做人工校验', '模拟抽检只能说明规则间一致，不能替代人工金标准。60 条 CSV 已导出但未完成标注。'],
    ['选做二未做', '记忆摘要压缩方向未实现。打分只做「选或不选」，没有做「改写压缩」。'],
  ];
  lim.forEach(([h, d], i) => {
    const y = 1.74 + i * 1.02;
    s.addShape(S.roundRect, { x: 0.78, y, w: 5.85, h: 0.92, fill: { color: MIST }, line: { color: MIST }, rectRadius: 0.04 });
    txt(s, h, { x: 1.0, y: y + 0.08, w: 5.4, h: 0.28, fontSize: 10.5, bold: true, color: INK });
    txt(s, d, { x: 1.0, y: y + 0.38, w: 5.4, h: 0.5, fontSize: 9, color: INK2, lineSpacingMultiple: 1.24 });
  });

  kicker(s, '三个改进方向（按预期收益排序）', 6.95, 1.42, 5.6, TEAL);
  const imp = [
    ['① 对话归属特征', '当前输入只有单轮句向量，看不到对话边界。加入 dialogue_id 或对话级聚类，直接针对第 25 页诊断出的盲区。', AMBER],
    ['② 摘要式压缩', '从「选或不选」扩展到「改写压缩」，同样 token 预算下可容纳更多信息。这是选做二的方向。', TEAL],
    ['③ 扩大评测规模', '把 L2 从 180 槽位扩到 1,000+ 以提升统计功效，使 2–3 个百分点量级的差异可被检出。', SLATE],
  ];
  imp.forEach(([h, d, col], i) => {
    const y = 1.74 + i * 1.32;
    s.addShape(S.roundRect, { x: 6.95, y, w: 0.06, h: 1.16, fill: { color: col }, line: { color: col } });
    txt(s, h, { x: 7.22, y: y + 0.06, w: 5.3, h: 0.3, fontSize: 11, bold: true, color: INK });
    txt(s, d, { x: 7.22, y: y + 0.4, w: 5.3, h: 0.72, fontSize: 9.5, color: INK2, lineSpacingMultiple: 1.28 });
  });

  takeaway(s, '四项局限都已在报告中量化披露，未做模糊表述；三个改进方向中第一项有第 25 页的实证支撑，优先级最高。', { y: 6.16, h: 0.34, color: SLATE, size: 10 });

  s.addNotes('局限与改进。四项局限都在报告中量化披露。三个改进方向里第一项有实证支撑：第 25 页诊断出打分器看不到对话边界。');
}

// ============================================================
// 34. 结束
// ============================================================
{
  const s = pres.addSlide({ masterName: 'DARK', sectionTitle: '第五部分 · 校验与讨论' });
  txt(s, '谢谢', { x: 0.9, y: 2.35, w: 9, h: 0.9, fontSize: 40, bold: true, color: PAPER, fontFace: HEAD });
  txt(s, '代码、训练日志、消融原始结果与完整报告均已随附，可一条命令复现全部非 LLM 结果。', {
    x: 0.9, y: 3.4, w: 10.5, h: 0.4, fontSize: 13, color: SAND,
  });
  s.addShape(S.line, { x: 0.9, y: 4.15, w: 4.2, h: 0, line: { color: '3D4E6B', width: 1 } });
  txt(s, 'github.com/MinamiFi/machine_learning_homework', { x: 0.9, y: 4.35, w: 10.5, h: 0.34, fontSize: 12.5, color: AMBER, fontFace: MONO });
  txt(s, 'bash exp/scripts/run_all.sh --skip-l2', { x: 0.9, y: 4.72, w: 10.5, h: 0.34, fontSize: 11.5, color: '9AA9C0', fontFace: MONO });
  txt(s, '不含 L2 的全流程约 25 分钟（2 核 CPU / 4 GB 内存，无 GPU）', { x: 0.9, y: 5.1, w: 10.5, h: 0.3, fontSize: 10.5, color: '7A8CA8' });

  s.addNotes('结束页。仓库地址与复现命令。');
}

// ---------- 输出 ----------
if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR, { recursive: true });
/**
 * applyTheme 只写颜色与主题名，字体仍是 pptxgenjs 默认的 Calibri Light / Calibri。
 * 中文靠 PowerPoint 自动回退，在没装中文字体的机器上会变成方框。
 * 所以这里把主题的 major/minor 字体（latin + ea + cs）一并改成中文字体，
 * 让图表里用 '+mn-lt' / '+mj-lt' 的文字也走同一套字体。
 */
async function patchThemeFonts(pptxPath, fontFace) {
  const JSZip = require('jszip');
  const zip = await JSZip.loadAsync(fs.readFileSync(pptxPath));
  // pptxgenjs 生成的包里自带 'ppt/' 与 'ppt/theme/' 两个目录条目，先删掉
  for (const name of Object.keys(zip.files)) {
    if (zip.files[name].dir) delete zip.files[name];
  }
  const themePath = 'ppt/theme/theme1.xml';
  let xml = await zip.file(themePath).async('string');
  for (const tag of ['majorFont', 'minorFont']) {
    // 原文已是 latin + ea + cs 三段（ea/cs 为空）。必须整体替换这三段，
    // 只替换 latin 会在其后追加一对 ea/cs，导致元素重复、XSD 校验失败。
    const re = new RegExp(
      `(<a:${tag}>)<a:latin typeface="[^"]*"[^/]*/><a:ea typeface="[^"]*"/><a:cs typeface="[^"]*"/>`
    );
    const rep =
      `$1<a:latin typeface="${fontFace}"/><a:ea typeface="${fontFace}"/><a:cs typeface="${fontFace}"/>`;
    if (!re.test(xml)) throw new Error(`主题字体结构不符合预期：${tag}`);
    xml = xml.replace(re, rep);
  }
  // 注意：zip.file(path, data) 会自动补建父目录条目（'ppt/'、'ppt/theme/'），
  // 刚删掉的目录条目会被这里重新创建，所以必须传 createFolders: false。
  zip.file(themePath, xml, { createFolders: false });
  // 目录条目（以 / 结尾）会让云服务与部分编辑器打不开或无法共同编辑。
  // 关键在于 generateAsync 的 createFolders: false —— 默认 true 会给每个
  // 带斜杠的路径自动补一个目录条目，所以只重建 zip 是不够的。
  const buf = await zip.generateAsync({
    type: 'nodebuffer',
    compression: 'DEFLATE',
    compressionOptions: { level: 9 },
    createFolders: false,
  });
  fs.writeFileSync(pptxPath, buf);
}

pres.writeFile({ fileName: OUT })
  .then(async () => {
    await applyTheme(OUT, THEME);
    await patchThemeFonts(OUT, HEAD);
    console.log('已生成：' + OUT);
  })
  .catch((e) => {
    console.error('生成失败：', e);
    process.exit(1);
  });
