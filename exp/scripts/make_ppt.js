#!/usr/bin/env node
/**
 * 生成交付用的汇报 PPT（题目三）。
 *
 * 设计取舍：
 * - 用 pptxgenjs 的 structured deck（theme + layout + section），这样配色改一处生效，
 *   文字位置由 layout 的 placeholder 统一，避免逐页手调导致标题漂移。
 * - 数据全部来自 exp/runs/ 的 json 与 交付/报告/实验报告.md，不在这里另算。
 * - 图表用 PowerPoint 原生 chart（不用截图），便于答辩时改数据。
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

// ---------- 四人分工（每人约 6 分钟，共 24 分钟 + 提问） ----------
// 题目是小组作业，汇报也按人切段：每人 4 页，讲完一段再交给下一位。
// 分工依据是「数据依赖」而不是「页数均分」：
//   1 号讲清楚要做什么、怎么做 → 2 号讲标签与网络（方法的地基）
//   3 号讲三组实验的主表（题目要求的指标） → 4 号讲归因与负结果（判断力所在）
const GROUP = '第 29 组';
const SPEAKERS = [
  { no: 1, name: '许兆丞', part: '第一部分', title: '题目、方案与总体结构', minutes: 6, slides: [1, 2, 3, 4] },
  { no: 2, name: '文萧尧', part: '第二部分', title: '标签构造与打分网络', minutes: 6, slides: [5, 6, 7, 8] },
  { no: 3, name: '张瑞哲', part: '第三部分', title: '实验结果：四张主表', minutes: 6, slides: [9, 10, 11, 12] },
  { no: 4, name: '马家源', part: '第四部分', title: '关键发现、负结果与结论', minutes: 6, slides: [13, 14, 15, 16] },
];
// 每部分在分工页上的一句话提要（写「讲什么」而不是「讲几页」）
const PART_BULLETS = [
  '问题从哪来 · 四模块怎么切\n三组基线为什么这样设\n一句话结论先给出来',
  '标签为什么不用人工标注\n为什么浅层就够\n两层评测省下多少成本',
  '固定预算下谁选得准\n压缩到多少 token 仍够用\n三个超参到底有没有影响',
  'ORACLE 差 46.6 点说明什么\n四处没跑赢的地方\n下一步该改哪里',
];

// 页码 → 讲者，供页脚标记用（第 1 页是封面，不标）
const SPEAKER_OF = {};
for (const sp of SPEAKERS) for (const i of sp.slides) SPEAKER_OF[i] = sp;

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
    { placeholder: { options: { name: 'title', type: 'title', x: 0.75, y: 0.4, w: 11.8, h: 0.92, fontSize: 32, bold: true, color: INK, fontFace: HEAD, valign: 'middle', align: 'left', margin: 0 } } },
    { placeholder: { options: { name: 'body', type: 'body', x: 0.75, y: 1.5, w: 11.8, h: 4.9, fontSize: 15, color: INK2, fontFace: BODY, margin: 0 } } },
    { text: { text: '高级机器学习 · 题目 3 · 记忆重要性打分与长上下文优化', options: { x: 0.75, y: 6.88, w: 8.5, h: 0.28, fontSize: 9.5, color: SLATE, fontFace: BODY } } },
  ],
  slideNumber: { x: 12.2, y: 6.88, w: 0.7, h: 0.28, fontSize: 9.5, color: SLATE, fontFace: BODY, align: 'right' },
});

// ---------- 小工具 ----------
const txt = (s, text, o) => s.addText(text, Object.assign({ isTextBox: true, fontFace: BODY, color: INK2, margin: 0, valign: 'top' }, o));

/** 大数字 + 小标签 */
function statCard(s, { x, y, w, h, value, unit, label, accent }) {
  s.addShape(S.roundRect, { x, y, w, h, fill: { color: MIST }, line: { color: MIST }, rectRadius: 0.06 });
  txt(s, value, { x: x + 0.22, y: y + 0.16, w: w - 0.44, h: 0.62, fontSize: 30, bold: true, color: accent || AMBER, fontFace: HEAD });
  if (unit) txt(s, unit, { x: x + 0.22, y: y + 0.74, w: w - 0.44, h: 0.26, fontSize: 11, color: SLATE });
  txt(s, label, { x: x + 0.22, y: y + (unit ? 1.0 : 0.8), w: w - 0.44, h: h - (unit ? 1.05 : 0.85), fontSize: 11.5, color: INK2, lineSpacingMultiple: 1.2 });
}

/** 章节标题（深色页用） */
function sectionTitle(s, kicker, title) {
  txt(s, kicker, { x: 0.9, y: 2.5, w: 8, h: 0.4, fontSize: 13, color: AMBER, bold: true, charSpacing: 2 });
  txt(s, title, { x: 0.9, y: 2.95, w: 10.5, h: 1.0, fontSize: 40, bold: true, color: PAPER, fontFace: HEAD });
}

/** 页码位置统一：内容页的标题占位 */
const title = (s, t) => s.addText(t, { placeholder: 'title' });

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
  txt(s, `${GROUP}　｜　许兆丞　文萧尧　张瑞哲　马家源`, { x: 0.9, y: 5.32, w: 9.0, h: 0.34, fontSize: 13, color: 'B9C4D6' });
  txt(s, '汇报安排：四人接力，每人 6 分钟', { x: 0.9, y: 5.7, w: 9.0, h: 0.32, fontSize: 11.5, color: '9AA9C0' });
  txt(s, '2026 年 11 月', { x: 0.9, y: 6.06, w: 5, h: 0.32, fontSize: 11.5, color: '8A99B0' });

s.addNotes(
    '【许兆丞 · 0:00–0:30】开场 30 秒。\n' +
    '先说三件事：我们做的是题目三，数据集 MultiWOZ 2.2，四个人接力讲。\n' +
    '然后立刻进入路线图，不要在这里展开技术细节。'
  );
}

pres.addSection({ title: '第一部分 · 题目与方案' });

// ============================================================
// 2. 分工总览
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第一部分 · 题目与方案' });
  title(s, '四人接力，每人 6 分钟');

  txt(s, '按「数据依赖」切段而不是按页数均分：前一位讲完的地基，正好是后一位要用的前提。', {
    x: 0.78, y: 1.44, w: 11.8, h: 0.3, fontSize: 11.5, color: SLATE,
  });

  // 四张卡片横向排开，每张 = 一位讲者
  // 4*cw + 3*gap 必须落在 [0.78, 12.55] 内，否则最后一张卡会出画布
  const cw = 2.71, gap = 0.3, y0 = 1.95;
  SPEAKERS.forEach((sp, i) => {
    const x = 0.78 + i * (cw + gap);
    const isLast = i === SPEAKERS.length - 1;
    s.addShape(S.roundRect, { x, y: y0, w: cw, h: 2.98, fill: { color: MIST }, line: { color: MIST }, rectRadius: 0.06 });
    // 顶部色条：越靠后颜色越深，暗示「从方法到判断」
    s.addShape(S.rect, { x, y: y0, w: cw, h: 0.09, fill: { color: [AMBER, TEAL, SLATE, INK][i] }, line: { color: [AMBER, TEAL, SLATE, INK][i] } });

    txt(s, `${sp.part}　${sp.minutes} 分钟`, { x: x + 0.22, y: y0 + 0.24, w: cw - 0.44, h: 0.26, fontSize: 10, color: SLATE, bold: true, charSpacing: 0.6 });
    txt(s, sp.name, { x: x + 0.22, y: y0 + 0.56, w: cw - 0.44, h: 0.44, fontSize: 21, bold: true, color: INK, fontFace: HEAD });
    txt(s, sp.title, { x: x + 0.22, y: y0 + 1.08, w: cw - 0.44, h: 0.82, fontSize: 12, color: INK2, lineSpacingMultiple: 1.32 });

    const first = sp.slides[0], last = sp.slides[sp.slides.length - 1];
    txt(s, `第 ${first}–${last} 页`, { x: x + 0.22, y: y0 + 1.62, w: cw - 0.44, h: 0.26, fontSize: 10, color: SLATE });
    txt(s, PART_BULLETS[i], { x: x + 0.22, y: y0 + 1.96, w: cw - 0.44, h: 0.94, fontSize: 10, color: INK2, lineSpacingMultiple: 1.42 });
    if (i < SPEAKERS.length - 1) {
      s.addShape(S.rightArrow, { x: x + cw + 0.035, y: y0 + 1.7, w: 0.23, h: 0.2, fill: { color: 'C9D0DA' }, line: { color: 'C9D0DA' } });
    }
  });

  // 底部：交接语，提醒每人结尾要留给下一位什么
  s.addShape(S.roundRect, { x: 0.78, y: 5.32, w: 11.8, h: 0.9, fill: { color: INK }, line: { color: INK }, rectRadius: 0.05 });
  txt(s, '交接约定', { x: 1.02, y: 5.45, w: 1.2, h: 0.26, fontSize: 10, color: AMBER, bold: true });
  txt(s, '每人最后一句话必须是「交给下一位的问题」：①为什么是浅层网络 ②为什么拆两层评测 ③瓶颈到底在哪 ④还差什么', {
    x: 2.2, y: 5.43, w: 10.15, h: 0.3, fontSize: 10.5, color: 'D7DEEA',
  });
  txt(s, '这样四段不是四份独立汇报，而是一条连续的推理链。', {
    x: 2.2, y: 5.76, w: 10.15, h: 0.3, fontSize: 9.5, color: '8A99B0',
  });

s.addNotes(
    '【许兆丞 · 0:30–2:00】分工页，90 秒。\n' +
    '重点讲「为什么这样切」：不是按页数均分，是按数据依赖。\n' +
    '第 1 部分给问题与方法骨架，第 2 部分给地基（标签与网络），第 3 部分给题目要的三指标，第 4 部分给归因。\n' +
    '最后点一下交接约定：每人结尾留一个问题给下一位，四段是一条推理链。\n' +
    '交接语：「下面请文萧尧讲我们怎么把对话变成可训练的数据。」'
  );
}

// ============================================================
// 3. 题目要求 vs 交付
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第一部分 · 题目与方案' });
  title(s, '题目要什么，我们交了什么');

  txt(s, '必做要求', { x: 0.78, y: 1.5, w: 6.2, h: 0.3, fontSize: 12, bold: true, color: AMBER, charSpacing: 1.5 });
  const must = [
    ['搭建具备记忆模块的对话智能体，搭建向量检索记忆库', '四模块齐全：memory / scorer / agent / eval'],
    ['构造记忆样本数据集，训练全连接打分网络', '65,213 轮弱监督样本，384→256→256→1'],
    ['三组基线：全量 / 仅检索 / 检索+打分', 'B0 / B1 / B2，另加 6 组零成本基线'],
    ['消融：层数 / 学习率 / L2 正则 → 三指标', '12 个配置 × 3 seeds，结论见第 11 页'],
  ];
  must.forEach(([h, d], i) => {
    const y = 1.88 + i * 1.16;
    s.addShape(S.ellipse, { x: 0.8, y: y + 0.04, w: 0.2, h: 0.2, fill: { color: AMBER }, line: { color: AMBER } });
    txt(s, h, { x: 1.14, y, w: 5.9, h: 0.5, fontSize: 13.5, bold: true, color: INK, lineSpacingMultiple: 1.15 });
    txt(s, d, { x: 1.14, y: y + 0.52, w: 5.9, h: 0.32, fontSize: 11.5, color: SLATE });
  });

  // 右侧：选做与交付物，如实标完成度
  s.addShape(S.roundRect, { x: 7.45, y: 1.5, w: 5.1, h: 2.62, fill: { color: MIST }, line: { color: MIST }, rectRadius: 0.06 });
  txt(s, '选做拓展', { x: 7.7, y: 1.68, w: 4.6, h: 0.28, fontSize: 11.5, bold: true, color: INK });
  txt(s, '① 注入错误记忆观察失效场景', { x: 7.7, y: 2.04, w: 4.6, h: 0.28, fontSize: 12, color: INK2 });
  txt(s, '⚠ 部分完成 — 第 13 页的冲突干扰分析', { x: 7.95, y: 2.3, w: 4.4, h: 0.26, fontSize: 10.5, color: BRICK });
  txt(s, '② 打分基础上加记忆摘要压缩', { x: 7.7, y: 2.66, w: 4.6, h: 0.28, fontSize: 12, color: INK2 });
  txt(s, '✗ 未做 — 时间优先给了标签校验', { x: 7.95, y: 2.92, w: 4.4, h: 0.26, fontSize: 10.5, color: BRICK });
  txt(s, '这两条如实标注，不假装完成。', { x: 7.7, y: 3.36, w: 4.6, h: 0.3, fontSize: 10.5, color: SLATE, italic: true });

  s.addShape(S.roundRect, { x: 7.45, y: 4.3, w: 5.1, h: 2.05, fill: { color: INK }, line: { color: INK }, rectRadius: 0.06 });
  txt(s, '交付物（6 项）', { x: 7.7, y: 4.48, w: 4.6, h: 0.28, fontSize: 11.5, bold: true, color: AMBER });
  txt(s, '报告 / 实验报告（合并为一份，44K 字）\nAgent + 打分网络代码（四模块，48 项单测）\n训练日志 · 消融实验结果 · 7 张图 · PPT', {
    x: 7.7, y: 4.82, w: 4.6, h: 1.4, fontSize: 11.5, color: 'D7DEEA', lineSpacingMultiple: 1.45,
  });

s.addNotes(
    '【许兆丞 · 2:00–3:30】题目要求对照，90 秒。\n' +
    '四条必做逐条说我们怎么做的，一句话一条，不要念细节。\n' +
    '选做两条主动说明：① 部分完成 ② 未做。这里不要含糊，被追问更被动。\n' +
    '交付物 6 项一句带过。'
  );
}

// ============================================================
// 4. 方法总览
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第一部分 · 题目与方案' });
  title(s, '四模块，依赖单向');

  const mods = [
    ['memory', '切片 · 编码 · 长历史 · 检索 · 预算内选择', '关键：select() 是所有基线共用的截断逻辑'],
    ['scorer', '384→256→256→1 打分网 + 6 组零成本基线', '关键：训练时必须同时报位置基线 AUC'],
    ['agent', 'prompt · LLM 客户端 · 响应缓存 · 主循环', '关键：缓存是必需不是优化'],
    ['eval', 'CR / NDCG / 任务完成率 / token 成本', '关键：L1 与 L2 两层调度'],
  ];
  const bw = 2.62, gap = 0.45, y0 = 2.35;
  mods.forEach(([name, d, key], i) => {
    const x = 0.78 + i * (bw + gap);
    s.addShape(S.roundRect, { x, y: y0, w: bw, h: 2.5, fill: { color: MIST }, line: { color: MIST }, rectRadius: 0.06 });
    txt(s, name, { x: x + 0.22, y: y0 + 0.2, w: bw - 0.44, h: 0.34, fontSize: 17, bold: true, color: INK, fontFace: MONO });
    txt(s, d, { x: x + 0.22, y: y0 + 0.62, w: bw - 0.44, h: 0.9, fontSize: 11, color: INK2, lineSpacingMultiple: 1.3 });
    txt(s, key, { x: x + 0.22, y: y0 + 1.62, w: bw - 0.44, h: 0.75, fontSize: 10, color: AMBER, lineSpacingMultiple: 1.25 });
    if (i < 3) {
      s.addShape(S.rightArrow, { x: x + bw + 0.09, y: y0 + 1.12, w: 0.27, h: 0.26, fill: { color: SLATE }, line: { color: SLATE } });
    }
  });

  txt(s, '为什么「单向」重要：eval 可以只导入前三模块跑 L1，不加载 LLM —— 省 3.2 GB 内存，把一次评测从 17 秒压到毫秒级。', {
    x: 0.78, y: 5.15, w: 11.8, h: 0.34, fontSize: 12.5, color: INK2,
  });
  txt(s, 'select(scores, tokens, budget) 单独成函数，是公平性的代码级保证：B0/B1/B2/R0/R1/R2/R3 共用同一截断逻辑，唯一变量是 scores 从哪来。', {
    x: 0.78, y: 5.6, w: 11.8, h: 0.34, fontSize: 12.5, color: SLATE,
  });

s.addNotes(
    '【许兆丞 · 3:30–5:30】四模块，2 分钟。这是第 1 部分的技术落点。\n' +
    '讲两件事：① 依赖单向，所以 L1 评测可以不加载 LLM，17 秒压到毫秒级；② select() 单独成函数，七组基线共用同一截断逻辑，唯一变量是 scores 从哪来。\n' +
    '这两点是后面所有实验可比的前提。\n' +
    '交接语：「为什么用浅层网络就够？请文萧尧从标签说起。」'
  );
}

pres.addSection({ title: '第二部分 · 标签与网络' });

// ============================================================
// 5. 标签从哪来
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第二部分 · 标签与网络' });
  title(s, '标签：belief state 差分，三行代码');

  s.addShape(S.roundRect, { x: 0.78, y: 1.55, w: 7.0, h: 2.28, fill: { color: INK }, line: { color: INK }, rectRadius: 0.06 });
  const code = [
    'cur   = norm_state(r.states_22)',
    'delta = {k: v for k, v in cur.items()',
    '         if prev.get(k) != v}',
    'n_valid = sum(1 for k, v in delta.items()',
    '              if final.get(k) == v)',
    'label = int(n_valid > 0)',
  ];
  code.forEach((line, i) => {
    txt(s, line, { x: 1.02, y: 1.76 + i * 0.32, w: 6.6, h: 0.3, fontSize: 11.5, color: i === 5 ? AMBER : 'D7DEEA', fontFace: MONO });
  });

  txt(s, '第 t 轮「重要」 ⟺ 它引入了状态变化，且至少一个 (slot, value) 到对话结束仍然有效。', {
    x: 0.78, y: 3.98, w: 7.0, h: 0.32, fontSize: 11.5, color: INK2,
  });

  statCard(s, { x: 8.1, y: 1.55, w: 2.2, h: 1.55, value: '65,213', label: '训练轮次（弱监督样本）' });
  statCard(s, { x: 10.45, y: 1.55, w: 2.13, h: 1.55, value: '55.6%', label: '正例率（不是预估的 25–40%）' });

  // 位置偏置：这是必须先讲的坑
  s.addShape(S.roundRect, { x: 8.1, y: 3.3, w: 4.48, h: 1.28, fill: { color: 'FBF3E6' }, line: { color: 'FBF3E6' }, rectRadius: 0.06 });
  txt(s, '必须先处理的坑：位置偏置', { x: 8.32, y: 3.46, w: 4.1, h: 0.28, fontSize: 11.5, bold: true, color: BRICK });
  txt(s, '正例率 turn 0 = 85.0% → turn 10 = 16.0%（首因）。\n「对话内相对位置」这一条零成本规则的 AUC 就有 0.8172。', {
    x: 8.32, y: 3.76, w: 4.1, h: 0.7, fontSize: 10.5, color: INK2, lineSpacingMultiple: 1.3,
  });

  // 分工页承诺「标签为什么不用人工标注」，这里给三条对照，填掉左下空白
  const why = [
    ['65,213 条', '人工标注按每条 8 秒算要 145 小时，且跨标注员一致性无法保证'],
    ['规则可复核', '任何人拿 states_22 重跑三行代码，得到逐位相同的标签，不需要第二个人'],
    ['代价是偏差', '标的是「引入」不是「必要」：集合覆盖显示 49.2% 的轮次可删 —— 第 8 页用三个口径量化'],
  ];
  why.forEach(([k, v], i) => {
    const y = 5.05 + i * 0.5;
    txt(s, k, { x: 0.78, y, w: 1.72, h: 0.3, fontSize: 11, bold: true, color: AMBER });
    txt(s, v, { x: 2.55, y, w: 5.35, h: 0.3, fontSize: 10.5, color: INK2 });
  });



s.addNotes(
    '【文萧尧 · 0:00–1:30】标签怎么来，90 秒。\n' +
    '三行代码讲清楚：第 t 轮重要 ⟺ 引入状态变化，且至少一个 slot 到对话结束仍有效。\n' +
    '两个数字：65,213 轮，正例率 55.6%（比预估高很多）。\n' +
    '必须主动讲位置偏置：turn 0 正例率 85% → turn 10 只剩 16%，这条零成本规则的 AUC 就有 0.8172。不讲会被问。'
  );
}

// ============================================================
// 6. 打分网络
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第二部分 · 标签与网络' });
  title(s, '打分网络：384 → 256 → 256 → 1');

  const boxes = [['384', '句向量\n(all-MiniLM-L6-v2)'], ['256', 'Linear + GELU\n+ Dropout 0.1'], ['256', 'Linear + GELU\n+ Dropout 0.1'], ['1', 'logit']];
  const bx = 0.78, bwd = 2.5, bgap = 0.55, by = 2.0;
  boxes.forEach(([n, d], i) => {
    const x = bx + i * (bwd + bgap);
    s.addShape(S.roundRect, { x, y: by, w: bwd, h: 1.35, fill: { color: i === 3 ? INK : MIST }, line: { color: i === 3 ? INK : 'DCE2EA' }, rectRadius: 0.06 });
    txt(s, n, { x, y: by + 0.16, w: bwd, h: 0.44, fontSize: 24, bold: true, color: i === 3 ? AMBER : INK, align: 'center', fontFace: HEAD });
    txt(s, d, { x: x + 0.1, y: by + 0.66, w: bwd - 0.2, h: 0.62, fontSize: 10, color: i === 3 ? 'B9C4D6' : SLATE, align: 'center', lineSpacingMultiple: 1.25 });
    if (i < 3) s.addShape(S.rightArrow, { x: x + bwd + 0.12, y: by + 0.55, w: 0.3, h: 0.25, fill: { color: SLATE }, line: { color: SLATE } });
  });

  statCard(s, { x: 0.78, y: 3.7, w: 2.75, h: 1.6, value: '0.9332', label: 'test AUC' });
  statCard(s, { x: 3.78, y: 3.7, w: 2.75, h: 1.6, value: '0.8172', label: '最强零成本基线（相对位置）', accent: SLATE });
  statCard(s, { x: 6.78, y: 3.7, w: 2.75, h: 1.6, value: '+0.1160', label: '净优势，方法成立', accent: TEAL });

  s.addShape(S.roundRect, { x: 9.9, y: 3.7, w: 2.68, h: 1.6, fill: { color: 'FBF3E6' }, line: { color: 'FBF3E6' }, rectRadius: 0.06 });
  txt(s, '165,377', { x: 10.12, y: 3.86, w: 2.24, h: 0.44, fontSize: 22, bold: true, color: AMBER, fontFace: HEAD });
  txt(s, '个参数（2 层配置）\n1 层只有 99,585', { x: 10.12, y: 4.34, w: 2.24, h: 0.8, fontSize: 10.5, color: INK2, lineSpacingMultiple: 1.3 });

  // 分工页承诺「为什么浅层就够」，这里给出论证，同时填掉下方空白
  s.addShape(S.roundRect, { x: 0.78, y: 5.86, w: 11.8, h: 0.66, fill: { color: 'FBF3E6' }, line: { color: 'FBF3E6' }, rectRadius: 0.05 });
  txt(s, '为什么浅层就够', { x: 1.0, y: 5.98, w: 1.7, h: 0.28, fontSize: 11, bold: true, color: BRICK });
  txt(s, '判据是「这轮有没有引入状态变化」，属低阶特征，384 维句向量里已近似线性可分 —— 加深只增参数不加信息。第 11 页消融验证：1 层 0.9352 反而最高。', {
    x: 2.7, y: 5.96, w: 9.7, h: 0.46, fontSize: 10.5, color: INK2, lineSpacingMultiple: 1.3,
  });

  txt(s, '训练细节：pos_weight = 0.7993（负 25,159 / 正 31,477）。正例率过半，所以 pos_weight 不能按「负/正」算，方向会反。', {
    x: 0.78, y: 5.5, w: 11.8, h: 0.34, fontSize: 11.5, color: SLATE,
  });

s.addNotes(
    '【文萧尧 · 1:30–3:00】打分网络，90 秒。\n' +
    '结构一句话：384 维句向量 → 256 → 256 → 1，16.5 万参数。\n' +
    '三个数字对比：test AUC 0.9332 vs 位置基线 0.8172，净优势 +0.1160。\n' +
    '回答上一位的问题：为什么浅层就够 —— 判据是「这轮有没有引入状态变化」，属低阶特征，加深只增参数（第 11 页消融会验证）。'
  );
}

// ============================================================
// 7. 两层评测
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第二部分 · 标签与网络' });
  title(s, '为什么把评测拆成两层');

  // 左：L1
  s.addShape(S.roundRect, { x: 0.78, y: 1.55, w: 5.75, h: 3.35, fill: { color: MIST }, line: { color: MIST }, rectRadius: 0.06 });
  txt(s, 'L1 · 选择质量', { x: 1.02, y: 1.72, w: 5.3, h: 0.36, fontSize: 16, bold: true, color: INK, fontFace: HEAD });
  txt(s, '不需要 LLM', { x: 1.02, y: 2.08, w: 5.3, h: 0.26, fontSize: 10.5, color: TEAL, bold: true });
  txt(s, '指标：CR（needle 是否被选入上下文）、\nRecall@budget、NDCG@k', { x: 1.02, y: 2.42, w: 5.3, h: 0.7, fontSize: 12, color: INK2, lineSpacingMultiple: 1.35 });
  txt(s, '成本：全量 8,372 段打分 0.014 秒\n90 场景 × 3 长度 = 2.5 分钟', { x: 1.02, y: 3.2, w: 5.3, h: 0.7, fontSize: 12, color: INK2, lineSpacingMultiple: 1.35 });
  txt(s, '可以先跑、随时重跑、跑全量', { x: 1.02, y: 4.05, w: 5.3, h: 0.3, fontSize: 11, color: AMBER, bold: true });
  txt(s, '适合：判断「方法成不成立」', { x: 1.02, y: 4.38, w: 5.3, h: 0.3, fontSize: 11, color: SLATE });

  // 右：L2
  s.addShape(S.roundRect, { x: 6.83, y: 1.55, w: 5.75, h: 3.35, fill: { color: INK }, line: { color: INK }, rectRadius: 0.06 });
  txt(s, 'L2 · 任务完成率', { x: 7.07, y: 1.72, w: 5.3, h: 0.36, fontSize: 16, bold: true, color: PAPER, fontFace: HEAD });
  txt(s, '需要 LLM backbone', { x: 7.07, y: 2.08, w: 5.3, h: 0.26, fontSize: 10.5, color: AMBER, bold: true });
  txt(s, '指标：slot_acc、场景成功率、AR、\ntoken 消耗、CR', { x: 7.07, y: 2.42, w: 5.3, h: 0.7, fontSize: 12, color: 'D7DEEA', lineSpacingMultiple: 1.35 });
  txt(s, '成本：17 秒/样本（本地 0.5B）\n90 场景 = 28 分钟（API）', { x: 7.07, y: 3.2, w: 5.3, h: 0.7, fontSize: 12, color: 'D7DEEA', lineSpacingMultiple: 1.35 });
  txt(s, '只能跑子集，且要挑对 backbone', { x: 7.07, y: 4.05, w: 5.3, h: 0.3, fontSize: 11, color: AMBER, bold: true });
  txt(s, '适合：回答题目要的三个指标', { x: 7.07, y: 4.38, w: 5.3, h: 0.3, fontSize: 11, color: '9AA9C0' });

  txt(s, '策略：先 L1 全量扫（秒级），若打分器赢不过位置基线就停下来改标签，不浪费 8 小时跑 L2 —— 把「发现方法不行」从 8 小时压到 1 分钟。', {
    x: 0.78, y: 5.12, w: 11.8, h: 0.5, fontSize: 12.5, color: INK2, lineSpacingMultiple: 1.3,
  });
  txt(s, 'L2 的 backbone 换过一次：0.5B 的绝对 JGA 只有 0.589，DeepSeek-V4.1-Flash 是 0.9174。结论以 API 为准。', {
    x: 0.78, y: 5.75, w: 11.8, h: 0.34, fontSize: 11.5, color: SLATE,
  });

s.addNotes(
    '【文萧尧 · 3:00–4:30】两层评测，90 秒。\n' +
    '核心论点：L1 零 LLM 成本，可以跑全量、随时重跑；L2 要 LLM，只能跑子集。\n' +
    '所以策略是先用 L1 全量扫，赢不过位置基线就停下来改标签 —— 把「发现方法不行」从 8 小时压到 1 分钟。\n' +
    '顺带说明 0.5B 只能打通管线（绝对 JGA 0.589），主表用 DeepSeek-V4.1-Flash（0.9174）。'
  );
}

// ============================================================
// 8. 标签三口径
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第二部分 · 标签与网络' });
  title(s, '标签可信吗：三个独立口径，方向一致');

  const cols = [
    ['口径一', '反事实删除（覆盖率）', '10.56×', '同分母配对下，删重要轮的信息丢失率是删不重要轮的倍数', 'test 10.56× · dev 13.23× · train 10.71×', AMBER],
    ['口径二', 'JGA（LLM 答题）', '−0.6755', '删重要轮后答对率从 0.9174 掉到 0.2477（p<0.001）', '607 段 / 1,332 槽位 / DeepSeek-V4.1-Flash', TEAL],
    ['口径三', '模拟抽检（Cohen κ）', '0.43–0.50', '标签与两个规则判据的中等一致（非人工校验）', 'κ(J2,J3)=0.707 说明判据本身稳定', SLATE],
  ];
  cols.forEach(([k, name, big, desc, foot, col], i) => {
    const x = 0.78 + i * 4.05;
    s.addShape(S.roundRect, { x, y: 1.5, w: 3.85, h: 4.1, fill: { color: MIST }, line: { color: MIST }, rectRadius: 0.06 });
    txt(s, k, { x: x + 0.24, y: 1.66, w: 3.4, h: 0.26, fontSize: 10, color: col, bold: true, charSpacing: 1.2 });
    txt(s, name, { x: x + 0.24, y: 1.94, w: 3.4, h: 0.34, fontSize: 12.5, bold: true, color: INK });
    txt(s, big, { x: x + 0.24, y: 2.4, w: 3.4, h: 0.72, fontSize: 30, bold: true, color: col, fontFace: HEAD });
    txt(s, desc, { x: x + 0.24, y: 3.2, w: 3.4, h: 1.0, fontSize: 10.5, color: INK2, lineSpacingMultiple: 1.35 });
    txt(s, foot, { x: x + 0.24, y: 4.5, w: 3.4, h: 0.7, fontSize: 9.5, color: SLATE, lineSpacingMultiple: 1.3 });
  });

  txt(s, '为什么值得做：如果标签是错的，前面所有结论都不成立。', { x: 0.78, y: 5.78, w: 11.8, h: 0.3, fontSize: 11.5, bold: true, color: INK });
  txt(s, '两个必须说明的限定：① κ 是模拟值，不是人工校验（用户已明确无时间，CSV 已导出 60 条待填）② 标签标的是「引入」而非「必要」，49.2% 的轮次在集合覆盖下可删。', {
    x: 0.78, y: 6.12, w: 11.8, h: 0.5, fontSize: 10.5, color: SLATE, lineSpacingMultiple: 1.3,
  });

s.addNotes(
    '【文萧尧 · 4:30–6:00】标签三口径，90 秒。这是第 2 部分的收尾。\n' +
    '三个独立口径方向一致才敢用这个标签：反事实 10.56 倍、JGA −0.6755、模拟 κ 0.43–0.50。\n' +
    '主动说限定：κ 是模拟值不是人工校验；标签标的是「引入」而非「必要」，49.2% 轮次可删。\n' +
    '交接语：「标签可信了，那选得准不准、答得对不对？请张瑞哲给三张主表。」'
  );
}

pres.addSection({ title: '第三部分 · 实验结果' });

// ============================================================
// 9. L1 主表
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第三部分 · 实验结果' });
  title(s, 'L1：固定预算下，谁把关键信息选进来了');

  s.addChart(
    pres.ChartType.bar,
    [
      { name: 'B2 打分网络', labels: ['L=128', 'L=256', 'L=512'], values: [1.0, 0.7667, 0.5333] },
      { name: 'B1 仅向量检索', labels: ['L=128', 'L=256', 'L=512'], values: [0.7, 0.4, 0.3333] },
      { name: 'LEN 长度基线', labels: ['L=128', 'L=256', 'L=512'], values: [0.8333, 0.5333, 0.3] },
      { name: 'R0 位置基线', labels: ['L=128', 'L=256', 'L=512'], values: [0.4333, 0.2, 0.2] },
    ],
    {
      x: 0.62, y: 1.5, w: 7.6, h: 4.55,
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

  s.addShape(S.roundRect, { x: 8.45, y: 1.55, w: 4.13, h: 2.1, fill: { color: MIST }, line: { color: MIST }, rectRadius: 0.06 });
  txt(s, '跨长度 CR', { x: 8.67, y: 1.7, w: 3.7, h: 0.26, fontSize: 10.5, color: SLATE });
  txt(s, '0.7667', { x: 8.67, y: 1.95, w: 3.7, h: 0.6, fontSize: 32, bold: true, color: AMBER, fontFace: HEAD });
  txt(s, '比纯检索高 28.9 个百分点\n比位置基线高 48.9 个百分点', { x: 8.67, y: 2.62, w: 3.7, h: 0.7, fontSize: 11, color: INK2, lineSpacingMultiple: 1.3 });

  txt(s, '三条必须说清的观察', { x: 8.45, y: 3.85, w: 4.13, h: 0.3, fontSize: 11.5, bold: true, color: INK });
  const obs = [
    '「优势随历史变长而扩大」不成立：B2/B1 比值 1.43× → 1.92× → 1.60× 非单调',
    'B0 与 R1 完全相同 —— 无记忆模块时，全量上下文就是时间序截断',
    'LEN 意外地强（0.5555），但分层 AUC 仍 0.86–0.95，内容信号独立有效',
  ];
  obs.forEach((t, i) => {
    txt(s, '· ' + t, { x: 8.45, y: 4.14 + i * 0.6, w: 4.13, h: 0.58, fontSize: 10, color: INK2, lineSpacingMultiple: 1.25 });
  });

s.addNotes(
    '【张瑞哲 · 0:00–1:30】L1 主表，90 秒。\n' +
    '跨长度 CR：B2 0.7667，比纯检索高 28.9 个百分点。\n' +
    '三条观察要分开说，尤其是第一条负结果：「优势随历史变长而扩大」不成立，1.43× → 1.92× → 1.60× 非单调。\n' +
    'L=512 的 0.5333 是短板，后面还会提。'
  );
}

// ============================================================
// 10. 预算扫描
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第三部分 · 实验结果' });
  title(s, '预算扫描：压缩越狠，优势越大 —— 但有例外');

  // 用分组柱而非折线：B=256 处三条线挤在 0.03–0.08，折线的数据标签会互相压住。
  // 柱状并排后标签天然错开，趋势仍按预算递增可读。
  s.addChart(
    pres.ChartType.bar,
    [
      { name: 'B2 打分网络', labels: ['B=256', 'B=512', 'B=1024', 'B=2048'], values: [0.05, 0.225, 0.4, 0.625] },
      { name: 'B1 仅向量检索', labels: ['B=256', 'B=512', 'B=1024', 'B=2048'], values: [0.075, 0.1, 0.275, 0.275] },
      { name: 'R0 位置基线', labels: ['B=256', 'B=512', 'B=1024', 'B=2048'], values: [0.025, 0.05, 0.075, 0.1] },
    ],
    {
      x: 0.62, y: 1.5, w: 7.4, h: 4.3,
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
  txt(s, '横轴：token 预算　｜　柱子：L=512 时的 CR（库 16,874 token）', { x: 0.78, y: 5.9, w: 7.2, h: 0.3, fontSize: 10, color: SLATE });

  txt(s, '结论要分开说', { x: 8.3, y: 1.55, w: 4.28, h: 0.3, fontSize: 11.5, bold: true, color: INK });

  s.addShape(S.roundRect, { x: 8.3, y: 1.9, w: 4.28, h: 1.5, fill: { color: MIST }, line: { color: MIST }, rectRadius: 0.06 });
  txt(s, 'B=1024 / 2048 的 6 个格子，B2 全部第一', { x: 8.52, y: 2.04, w: 3.9, h: 0.32, fontSize: 11.5, bold: true, color: TEAL });
  txt(s, '且压缩越激进优势越大。L=512、B=1024 时 0.400 vs 0.275 vs 0.075。', { x: 8.52, y: 2.4, w: 3.9, h: 0.9, fontSize: 10.5, color: INK2, lineSpacingMultiple: 1.3 });

  s.addShape(S.roundRect, { x: 8.3, y: 3.55, w: 4.28, h: 1.5, fill: { color: 'FBF0EF' }, line: { color: 'FBF0EF' }, rectRadius: 0.06 });
  txt(s, 'B=256 的 3 个格子里有 2 个 B2 输给 B1', { x: 8.52, y: 3.69, w: 3.9, h: 0.32, fontSize: 11.5, bold: true, color: BRICK });
  txt(s, '只够放 8–9 个片段时，稠密打分器的排序精度不如余弦相似度。打分器学的是「片段整体重要性」，余弦直接对齐 query。', { x: 8.52, y: 4.05, w: 3.9, h: 0.9, fontSize: 10.5, color: INK2, lineSpacingMultiple: 1.3 });

  txt(s, '推荐配置：L=512 时 B=2048（保留 12.1%），CR 0.625 vs B1 的 0.275，2.3× 提升。R0 在全部 12 格最差。', {
    x: 0.78, y: 6.3, w: 11.8, h: 0.3, fontSize: 11, color: INK2,
  });

s.addNotes(
    '【张瑞哲 · 1:30–3:00】预算扫描，90 秒。\n' +
    '一句话结论：压缩越狠优势越大，但有例外。\n' +
    '正面：B=1024/2048 的 6 个格子 B2 全部第一。\n' +
    '负面：B=256 的 3 个格子里有 2 个 B2 输给余弦 —— 只够放 8–9 个片段时，稠密打分器排序精度不如直接对齐 query。这是真实边界，不是调参能解决。'
  );
}

// ============================================================
// 11. 消融
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第三部分 · 实验结果' });
  title(s, '消融：三个超参都没有显著影响');

  const head = ['因素', '取值', 'test AUC', '标准差', '参数量'].map((t) => ({
    text: t, options: { bold: true, color: PAPER, fill: { color: INK }, fontSize: 11, align: 'center' },
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
      fontSize: 11, align: j === 0 ? 'left' : 'center', color: INK2,
      fill: { color: i === 0 ? 'FBF3E6' : (i % 2 ? MIST : PAPER) },
      bold: i === 0 && j !== 0,
    },
  })));

  s.addTable([head, ...rows], {
    x: 0.78, y: 1.6, w: 7.3,
    colW: [1.35, 1.95, 1.85, 1.05, 1.1],
    border: { type: 'solid', color: 'E3E7EC', pt: 1 },
    rowH: 0.4,
    valign: 'middle',
    fontFace: BODY,
    margin: 0.05,
  });
  txt(s, '位置基线 0.8172 作为参照（未列入表中，差距 +0.116 以上）', { x: 0.78, y: 4.78, w: 7.3, h: 0.28, fontSize: 10, color: SLATE });

  statCard(s, { x: 8.35, y: 1.6, w: 2.1, h: 1.5, value: '0.0013', label: 'AUC 最大跨度' });
  statCard(s, { x: 10.6, y: 1.6, w: 1.98, h: 1.5, value: '0.0007', label: '种子间标准差', accent: SLATE });

  txt(s, '同量级 ⇒ 无显著差异', { x: 8.35, y: 3.25, w: 4.23, h: 0.3, fontSize: 12, bold: true, color: TEAL });

  const notes = [
    ['层数', '1 层反而最高（0.9352）。题目写「浅层」不是妥协，是正解 —— 加深只增参数。'],
    ['学习率', '1e-4 略优，但 3e-3 到 1e-4 全落在 0.9336–0.9349 窄带内，被种子噪声淹没。'],
    ['L2 正则', '四个取值小数点后四位完全相同，标准差也相同 —— 早停在 epoch 5 先触发，正则没机会起作用。'],
  ];
  notes.forEach(([k, v], i) => {
    txt(s, k, { x: 8.35, y: 3.68 + i * 0.95, w: 4.23, h: 0.26, fontSize: 11, bold: true, color: AMBER });
    txt(s, v, { x: 8.35, y: 3.94 + i * 0.95, w: 4.23, h: 0.68, fontSize: 10, color: INK2, lineSpacingMultiple: 1.25 });
  });

s.addNotes(
    '【张瑞哲 · 3:00–4:30】消融，90 秒。\n' +
    '三个超参都没有显著影响：AUC 最大跨度 0.0013，种子间标准差 0.0007，同量级。\n' +
    '层数那条要讲出意思：1 层反而最高（0.9352），题目写「浅层」不是妥协是正解。\n' +
    'L2 正则那条讲清楚：不是「L2 无用」，而是早停先触发、正则没机会起作用，不能外推。'
  );
}

// ============================================================
// 12. L2 主表
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第三部分 · 实验结果' });
  title(s, 'L2：把上下文压到 20%，准确率反而更高');

  s.addChart(
    pres.ChartType.bar,
    [{
      name: 'slot_acc',
      labels: ['B2 打分', 'B1 检索', 'R1 recency', 'B0 截断', 'R2 随机', 'R0 位置', 'ORACLE'],
      values: [0.3778, 0.3556, 0.3278, 0.3167, 0.3056, 0.2111, 0.8444],
    }],
    {
      x: 0.62, y: 1.5, w: 7.3, h: 4.3,
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
  txt(s, '90 场景 / 180 槽位，B=2048，DeepSeek-V4.1-Flash', { x: 0.78, y: 5.85, w: 7.1, h: 0.28, fontSize: 10, color: SLATE });

  s.addShape(S.roundRect, { x: 8.2, y: 1.5, w: 4.38, h: 1.85, fill: { color: MIST }, line: { color: MIST }, rectRadius: 0.06 });
  txt(s, '压缩 4.9 倍，指标还涨了 5.6 个百分点', { x: 8.42, y: 1.64, w: 4.0, h: 0.5, fontSize: 12, bold: true, color: TEAL, lineSpacingMultiple: 1.2 });
  txt(s, 'B2：1,489 token → 0.3778\nB0FULL：7,277 token → 0.3222', { x: 8.42, y: 2.2, w: 4.0, h: 0.6, fontSize: 11, color: INK2, lineSpacingMultiple: 1.4 });
  txt(s, '题目「控制 token 开销的前提下保障关键信息」的直接回答', { x: 8.42, y: 2.86, w: 4.0, h: 0.4, fontSize: 10, color: SLATE, lineSpacingMultiple: 1.25 });

  txt(s, '配对比较（bootstrap 5,000）', { x: 8.2, y: 3.55, w: 4.38, h: 0.28, fontSize: 11, bold: true, color: INK });
  const pairs = [
    ['B2 − B1', '+0.0222', 'p=0.6964　不显著', BRICK],
    ['B2 − B0', '+0.0611', 'p=0.1948　不显著', SLATE],
    ['B2 − R0', '+0.1667', 'p<0.0001　显著', TEAL],
    ['B2 − R2', '+0.0722', 'p=0.0984　边缘', SLATE],
  ];
  pairs.forEach(([a, b, c, col], i) => {
    txt(s, a, { x: 8.2, y: 3.9 + i * 0.42, w: 1.2, h: 0.28, fontSize: 10.5, color: INK2 });
    txt(s, b, { x: 9.4, y: 3.9 + i * 0.42, w: 1.0, h: 0.28, fontSize: 10.5, bold: true, color: INK });
    txt(s, c, { x: 10.45, y: 3.9 + i * 0.42, w: 2.1, h: 0.28, fontSize: 10, color: col });
  });

  txt(s, 'B2 唯一显著赢的是位置基线。对纯检索只高 2.2 个百分点且不显著 —— 180 条样本功效不够。', {
    x: 0.78, y: 6.28, w: 11.8, h: 0.3, fontSize: 11, color: BRICK,
  });

s.addNotes(
    '【张瑞哲 · 4:30–6:00】L2 主表，90 秒。这是题目要求的三指标主表。\n' +
    '正面结论：压缩 4.9 倍，slot_acc 还涨 5.6 个百分点（1,489 token / 0.3778 对 7,277 token / 0.3222）。\n' +
    '必须主动说配对结果：B2 唯一显著赢的是位置基线（B2−R0 p<0.0001），对纯检索只高 2.2 个百分点且不显著（p=0.6964）。\n' +
    '交接语：「为什么压到 20% 还只有 37.8%？请马家源拆这个差距。」'
  );
}

pres.addSection({ title: '第四部分 · 发现与结论' });

// ============================================================
// 13. ORACLE 关键发现
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第四部分 · 发现与结论' });
  title(s, '瓶颈不在网络，在「区分对话归属」');

  const cards = [
    ['216', 'token', 'ORACLE（只给本对话真实历史）', '0.8444', INK, PAPER],
    ['1,489', 'token', 'B2 打分网络 + 预算 2048', '0.3778', AMBER, PAPER],
    ['7,277', 'token', 'B0FULL 全量上下文（题目字面义）', '0.3222', SLATE, PAPER],
  ];
  cards.forEach(([v, u, label, acc, bg, fg], i) => {
    const x = 0.78 + i * 3.42;
    s.addShape(S.roundRect, { x, y: 1.5, w: 3.2, h: 2.15, fill: { color: bg }, line: { color: bg }, rectRadius: 0.06 });
    txt(s, v, { x: x + 0.24, y: 1.66, w: 2.7, h: 0.62, fontSize: 30, bold: true, color: i === 0 ? AMBER : (i === 1 ? INK : INK), fontFace: HEAD });
    txt(s, u + '  →  slot_acc ' + acc, { x: x + 0.24, y: 2.3, w: 2.75, h: 0.3, fontSize: 11.5, bold: true, color: i === 0 ? PAPER : INK });
    txt(s, label, { x: x + 0.24, y: 2.66, w: 2.75, h: 0.85, fontSize: 10.5, color: i === 0 ? 'B9C4D6' : INK, lineSpacingMultiple: 1.3 });
  });

  txt(s, '46.6 个百分点是什么', { x: 0.78, y: 3.9, w: 5.6, h: 0.3, fontSize: 12, bold: true, color: INK });
  txt(s, '不是「方法还不够好」，而是「任务本身有多难」的上界。\n\n如果能完美剔除跨对话的干扰轮，同样 216 token 就能到 84.4%。中间那 46.6 个百分点，就是「区分对话归属」这件事的价值。', {
    x: 0.78, y: 4.22, w: 5.6, h: 1.7, fontSize: 11, color: INK2, lineSpacingMultiple: 1.45,
  });

  txt(s, '证据：不是「没选到」', { x: 6.8, y: 3.9, w: 5.78, h: 0.3, fontSize: 12, bold: true, color: INK });
  const ev = [
    '30 个 L=128 场景，主对话段只被漏选 36 段，needle 全部选中',
    '但有 33 个（场景, 槽位）组合的上下文含「同槽位、不同值」的冲突轮',
    '例：hotel-area 的 gold 是 centre，另一段对话说 north，模型答 north',
  ];
  ev.forEach((t, i) => {
    txt(s, '· ' + t, { x: 6.8, y: 4.24 + i * 0.52, w: 5.78, h: 0.5, fontSize: 10.5, color: INK2, lineSpacingMultiple: 1.3 });
  });
  txt(s, '纯文本打分器看不到对话边界 —— 这是它的固有盲区，也是下一步最该改的地方（加 dialogue_id 特征或对话级聚类）。', {
    x: 6.8, y: 5.86, w: 5.78, h: 0.5, fontSize: 10.5, color: AMBER, lineSpacingMultiple: 1.3,
  });

s.addNotes(
    '【马家源 · 0:00–2:00】ORACLE 关键发现，2 分钟。这是全场最重要的一页。\n' +
    '三张卡：ORACLE 216 token 到 0.8444；B2 1,489 token 0.3778；B0FULL 7,277 token 0.3222。\n' +
    '46.6 个百分点不是「方法不够好」，是任务难度上界。\n' +
    '证据链要讲清：不是没选到（30 个场景主对话段只漏选 36 段，needle 全选中），是 33 个组合含「同槽位不同值」的冲突轮把模型带偏了。\n' +
    'B0FULL 是最直接证据：给全量上下文反而更差。'
  );
}

// ============================================================
// 14. 负结果
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第四部分 · 发现与结论' });
  title(s, '四处没跑赢的地方');

  const negs = [
    ['B2 相对 B1 在 L2 上不显著', '+0.0222，p=0.6964。180 条样本的功效不足以分辨这个量级。L1 上差 28.9 个百分点，但 CR 高不等于答对率高。'],
    ['B=256 时打分器不如余弦', '只够放 8–9 个片段时，3 个格子里有 2 个 B2 输给 B1。打分器学「整体重要性」，余弦直接对齐 query。'],
    ['L=512 时优势收窄', 'CR 只有 0.5333，且 L2 上 B1 的 slot_acc（0.3478）反超 B2（0.3043）。长历史仍是短板。'],
    ['「删噪声有益」在强模型上消失', '0.5B 上是 −0.075 且显著；换 DeepSeek-V4.1-Flash 后是 −0.0082，p=0.49 不显著。不能拿弱模型结论外推。'],
  ];
  negs.forEach(([h, d], i) => {
    const y = 1.62 + i * 1.28;
    s.addShape(S.roundRect, { x: 0.78, y: y + 0.04, w: 0.34, h: 0.34, fill: { color: 'FBF0EF' }, line: { color: 'FBF0EF' }, rectRadius: 0.04 });
    txt(s, String(i + 1), { x: 0.78, y: y + 0.09, w: 0.34, h: 0.28, fontSize: 12, bold: true, color: BRICK, align: 'center' });
    txt(s, h, { x: 1.28, y, w: 11.3, h: 0.36, fontSize: 13.5, bold: true, color: INK });
    txt(s, d, { x: 1.28, y: y + 0.38, w: 11.3, h: 0.7, fontSize: 11, color: INK2, lineSpacingMultiple: 1.3 });
  });

  txt(s, '主动列出来，是因为它们限定了方法的适用范围 —— 报告第 7 节与第 6.3 节有完整讨论。', {
    x: 0.78, y: 6.5, w: 11.8, h: 0.28, fontSize: 10, color: SLATE,
  });

s.addNotes(
    '【马家源 · 2:00–3:30】四处没跑赢，90 秒。\n' +
    '这一页专门放负结果，因为评审最可能追问这些。\n' +
    '四条按顺序快速过，每条一句：不显著、低预算不如余弦、L=512 收窄、删噪声有益在强模型上消失。\n' +
    '态度要坦率：它们限定了方法的适用范围。'
  );
}

// ============================================================
// 15. 结论
// ============================================================
{
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '第四部分 · 发现与结论' });
  title(s, '五条结论');

  const cs = [
    ['打分网络有效', 'AUC 0.9332 vs 位置基线 0.8172（+0.1160）；CR 0.7667 vs 纯检索 0.4778'],
    ['压缩换来的是「同样的 token 装更多有用的东西」', '1,489 token / 0.3778 对 7,277 token / 0.3222，压缩 4.9 倍且指标更高'],
    ['瓶颈是对话归属，不是网络容量', 'ORACLE 216 token 到 0.8444，中间 46.6 个百分点是任务难度上界'],
    ['「浅层」是正解，不是妥协', '1 层 99,585 参数最优，加深到 4 层参数翻三倍而指标略降'],
    ['标签三重校验方向一致，但有已知偏差', '标的是「引入」而非「必要」；49.2% 轮次可删；位置偏置已用基线控制'],
  ];
  cs.forEach(([h, d], i) => {
    const y = 1.55 + i * 1.02;
    s.addShape(S.ellipse, { x: 0.8, y: y + 0.06, w: 0.22, h: 0.22, fill: { color: i === 2 ? AMBER : SLATE }, line: { color: i === 2 ? AMBER : SLATE } });
    txt(s, h, { x: 1.2, y, w: 11.4, h: 0.36, fontSize: 14, bold: true, color: INK });
    txt(s, d, { x: 1.2, y: y + 0.38, w: 11.4, h: 0.44, fontSize: 11, color: INK2, lineSpacingMultiple: 1.3 });
  });

  txt(s, '下一步最值得做的：给打分器加对话归属特征（dialogue_id 或对话级聚类），它比调层数、调学习率有用得多。', {
    x: 1.2, y: 6.42, w: 11.0, h: 0.3, fontSize: 11, color: AMBER, bold: true,
  });

s.addNotes(
    '【马家源 · 3:30–5:00】五条结论，90 秒。\n' +
    '按重要性排序，第三条是重点（瓶颈是对话归属）。\n' +
    '最后落到「下一步做什么」：给打分器加对话归属特征，比调层数、调学习率有用得多。让汇报有落点。'
  );
}

// ============================================================
// 16. 结束
// ============================================================
{
  const s = pres.addSlide({ masterName: 'DARK', sectionTitle: '第四部分 · 发现与结论' });
  txt(s, '谢谢，欢迎提问', { x: 0.9, y: 2.55, w: 9, h: 1.0, fontSize: 40, bold: true, color: PAPER, fontFace: HEAD });
  txt(s, '代码、训练日志、消融结果与完整报告都在仓库里，可一条命令复现全部非 LLM 结果。', {
    x: 0.9, y: 3.7, w: 10.5, h: 0.4, fontSize: 13, color: SAND,
  });
  txt(s, 'github.com/MinamiFi/machine_learning_homework', { x: 0.9, y: 4.25, w: 10.5, h: 0.34, fontSize: 13, color: AMBER, fontFace: MONO });
  txt(s, 'bash exp/scripts/run_all.sh --skip-l2　约 25 分钟', { x: 0.9, y: 4.62, w: 10.5, h: 0.34, fontSize: 12, color: '9AA9C0', fontFace: MONO });

s.addNotes(
    '【马家源 · 5:00–6:00】收尾 60 秒。\n' +
    '仓库地址 + 一条命令复现（25 分钟，不含 L2）。\n' +
    '然后四人一起上台接受提问。备好三个答案：为什么 B2 不显著、为什么不用 7B、标签怎么验证的。'
  );
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

// ---------- 页脚讲者标记 ----------
// 四个人接力讲，评委需要随时知道「现在是谁在讲、这是第几部分」。
// 放在页码左侧，与页脚同基线，不新增视觉层级。
pres.slides.forEach((sl, idx) => {
  if (idx === 0) return; // 封面是全场共用的，不属于任何一位
  const sp = SPEAKER_OF[idx + 1];
  if (!sp) return;
  const dark = idx + 1 === pres.slides.length; // 结束页是深色底
  sl.addText(`${sp.part} · ${sp.name}`, {
    x: 9.5, y: dark ? 6.85 : 6.88, w: 2.6, h: dark ? 0.3 : 0.28,
    fontSize: dark ? 10 : 9.5, color: dark ? '9AA9C0' : SLATE,
    fontFace: BODY, align: 'right', margin: 0, valign: 'middle',
  });
});

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
