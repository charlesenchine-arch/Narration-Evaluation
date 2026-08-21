const fs = require("fs");
const path = require("path");
const {
  AlignmentType,
  BorderStyle,
  Document,
  ExternalHyperlink,
  Footer,
  Header,
  HeadingLevel,
  PageBreak,
  PageNumber,
  Packer,
  Paragraph,
  ShadingType,
  Table,
  TableCell,
  TableRow,
  TextRun,
  WidthType,
} = require("docx");

const root = path.resolve(__dirname, "..");
const protocolPath = path.join(root, "docs", "QUALITY_EVALUATION_PROTOCOL.md");
const surveyPath = path.join(root, "docs", "dataset_and_baseline_survey.md");
const outputPath = process.argv[2] || path.join(root, "deliverables", "中文叙事质量评估器_论文协议_v2.0.docx");

const NAVY = "17324D";
const VERMILION = "B83C2E";
const PAPER = "F5F1E8";
const PALE_BLUE = "E8EEF3";
const INK = "17212B";
const MUTED = "66717A";
const WHITE = "FFFFFF";
const LINE = "C8C2B5";
const CJK_FONT = {
  ascii: "Noto Sans CJK SC",
  eastAsia: "Noto Sans CJK SC",
  hAnsi: "Noto Sans CJK SC",
  cs: "Noto Sans CJK SC",
};

function inlineRuns(text, options = {}) {
  const runs = [];
  const regex = /\[([^\]]+)\]\((https?:\/\/[^)]+)\)|`([^`]+)`|\*\*([^*]+)\*\*/g;
  let last = 0;
  for (let match; (match = regex.exec(text)); ) {
    if (match.index > last) runs.push(new TextRun({ text: text.slice(last, match.index), ...options }));
    if (match[1]) {
      runs.push(new ExternalHyperlink({
        link: match[2],
        children: [new TextRun({ text: match[1], color: "2563A6", underline: {}, ...options })],
      }));
    } else if (match[3]) {
      runs.push(new TextRun({ text: match[3], font: "Menlo", color: VERMILION, ...options }));
    } else {
      runs.push(new TextRun({ text: match[4], bold: true, ...options }));
    }
    last = regex.lastIndex;
  }
  if (last < text.length) runs.push(new TextRun({ text: text.slice(last), ...options }));
  return runs.length ? runs : [new TextRun({ text, ...options })];
}

function tableFromLines(lines) {
  const rows = lines
    .filter((_, index) => index !== 1)
    .map(line => line.trim().replace(/^\||\|$/g, "").split("|").map(cell => cell.trim()));
  const columns = Math.max(...rows.map(row => row.length));
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    columnWidths: Array(columns).fill(Math.floor(9000 / columns)),
    rows: rows.map((row, rowIndex) => new TableRow({
      tableHeader: rowIndex === 0,
      children: Array.from({ length: columns }, (_, colIndex) => new TableCell({
        width: { size: Math.floor(9000 / columns), type: WidthType.DXA },
        shading: rowIndex === 0 ? { fill: NAVY, type: ShadingType.CLEAR } : undefined,
        margins: { top: 90, bottom: 90, left: 110, right: 110 },
        children: [new Paragraph({
          spacing: { after: 0 },
          children: inlineRuns(row[colIndex] || "", {
            font: CJK_FONT,
            size: 18,
            bold: rowIndex === 0,
            color: rowIndex === 0 ? WHITE : INK,
          }),
        })],
      })),
    })),
    borders: {
      top: { style: BorderStyle.SINGLE, size: 3, color: LINE },
      bottom: { style: BorderStyle.SINGLE, size: 3, color: LINE },
      left: { style: BorderStyle.SINGLE, size: 3, color: LINE },
      right: { style: BorderStyle.SINGLE, size: 3, color: LINE },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: LINE },
      insideVertical: { style: BorderStyle.SINGLE, size: 2, color: LINE },
    },
  });
}

let numberingSequence = 0;

function parseMarkdown(markdown, headingShift = 0, compact = false) {
  const lines = markdown.split(/\r?\n/);
  const children = [];
  let index = 0;
  let compactBody = compact;
  let activeNumberingReference = null;
  while (index < lines.length) {
    const line = lines[index].trim();
    if (!line) { index += 1; continue; }
    if (line.startsWith("|")) {
      activeNumberingReference = null;
      const tableLines = [];
      while (index < lines.length && lines[index].trim().startsWith("|")) tableLines.push(lines[index++]);
      children.push(tableFromLines(tableLines));
      children.push(new Paragraph({ spacing: { after: 120 } }));
      continue;
    }
    const heading = /^(#{1,4})\s+(.+)$/.exec(line);
    if (heading) {
      activeNumberingReference = null;
      const level = Math.min(4, heading[1].length + headingShift);
      children.push(new Paragraph({
        heading: [HeadingLevel.TITLE, HeadingLevel.HEADING_1, HeadingLevel.HEADING_2, HeadingLevel.HEADING_3, HeadingLevel.HEADING_4][level],
        pageBreakBefore: level === 1 && children.length > 0,
        children: inlineRuns(heading[2]),
      }));
      if (/^13\.\s/.test(heading[2])) compactBody = true;
      index += 1;
      continue;
    }
    const bullet = /^-\s+(.+)$/.exec(line);
    if (bullet) {
      activeNumberingReference = null;
      children.push(new Paragraph({
        bullet: { level: 0 },
        spacing: { after: compactBody ? 25 : 45 },
        children: inlineRuns(bullet[1], compactBody ? { size: compact ? 19 : 18 } : {}),
      }));
      index += 1;
      continue;
    }
    const numbered = /^(\d+)\.\s+(.+)$/.exec(line);
    if (numbered) {
      if (!activeNumberingReference) {
        numberingSequence += 1;
        activeNumberingReference = `main-numbering-${numberingSequence}`;
      }
      children.push(new Paragraph({
        numbering: { reference: activeNumberingReference, level: 0 },
        spacing: { after: compactBody ? 25 : 45 },
        children: inlineRuns(numbered[2], compactBody ? { size: compact ? 19 : 18 } : {}),
      }));
      index += 1;
      continue;
    }
    const blockquote = /^>\s*(.*)$/.exec(line);
    if (blockquote) {
      activeNumberingReference = null;
      children.push(new Paragraph({
        indent: { left: 360, right: 240 },
        shading: { fill: PALE_BLUE, type: ShadingType.CLEAR },
        border: { left: { style: BorderStyle.SINGLE, size: 16, color: VERMILION, space: 8 } },
        spacing: { before: 90, after: 120 },
        children: inlineRuns(blockquote[1], { italics: true, color: NAVY, ...(compactBody ? { size: compact ? 19 : 18 } : {}) }),
      }));
      index += 1;
      continue;
    }
    const paragraphLines = [line];
    activeNumberingReference = null;
    index += 1;
    while (index < lines.length) {
      const next = lines[index].trim();
      if (!next || /^(#{1,4})\s+/.test(next) || /^[-|>]\s?/.test(next) || /^\d+\.\s+/.test(next)) break;
      paragraphLines.push(next);
      index += 1;
    }
    children.push(new Paragraph({
      spacing: { after: compactBody ? 80 : 105, line: compactBody ? 310 : 340 },
      children: inlineRuns(paragraphLines.join(" "), compactBody ? { size: compact ? 19 : 18 } : {}),
    }));
  }
  return children;
}

const protocol = fs.readFileSync(protocolPath, "utf8").replace(/^# .+\n/, "").replace(/^更新日期：.+\n/, "");
const survey = fs.readFileSync(surveyPath, "utf8").replace(/^# .+\n/, "").replace(/^更新日期：.+\n/, "");

const tocRows = [
  ["一句话论文主张", "3"], ["1. 核心科学问题", "3"],
  ["2. 论文贡献与非贡献", "3"], ["3. 任务定义", "4"],
  ["4. 数据设计", "5"], ["5. 数据切分", "6"],
  ["6. 质量—来源解耦", "7"], ["7. 模型与训练", "8"],
  ["8. 指标与统计", "8"], ["9. 辅助输出", "9"],
  ["10. 论文主实验表", "9"], ["11. 成功标准", "9"],
  ["12. 可发表版本", "10"], ["13. 执行顺序", "10"],
  ["14. 关键参考", "11"], ["附录 A：数据集与基线调研", "12"],
];

const manualToc = new Table({
  width: { size: 100, type: WidthType.PERCENTAGE },
  columnWidths: [8200, 800],
  borders: {
    top: { style: BorderStyle.NIL }, bottom: { style: BorderStyle.NIL },
    left: { style: BorderStyle.NIL }, right: { style: BorderStyle.NIL },
    insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: "E3DED3" },
    insideVertical: { style: BorderStyle.NIL },
  },
  rows: tocRows.map(([title, page]) => new TableRow({ children: [
    new TableCell({ margins: { top: 100, bottom: 100, left: 0, right: 80 }, children: [new Paragraph({ spacing: { after: 0 }, children: [new TextRun({ text: title, color: NAVY, size: 20 })] })] }),
    new TableCell({ margins: { top: 100, bottom: 100, left: 80, right: 0 }, children: [new Paragraph({ alignment: AlignmentType.RIGHT, spacing: { after: 0 }, children: [new TextRun({ text: page, color: VERMILION, bold: true, size: 20 })] })] }),
  ] })),
});

const cover = [
  new Paragraph({ spacing: { before: 1000, after: 180 }, children: [new TextRun({ text: "NARRATIVE EVALUATION · RESEARCH PROTOCOL", color: VERMILION, bold: true, size: 19, characterSpacing: 80 })] }),
  new Paragraph({ spacing: { after: 220 }, children: [new TextRun({ text: "中文叙事质量评估器", color: NAVY, bold: true, size: 52 })] }),
  new Paragraph({ spacing: { after: 400 }, children: [new TextRun({ text: "论文与实验协议 v2.0", color: INK, size: 30 })] }),
  new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    borders: { top: { style: BorderStyle.NIL }, bottom: { style: BorderStyle.NIL }, left: { style: BorderStyle.NIL }, right: { style: BorderStyle.NIL }, insideHorizontal: { style: BorderStyle.NIL }, insideVertical: { style: BorderStyle.NIL } },
    rows: [new TableRow({ children: [
      new TableCell({ shading: { fill: NAVY, type: ShadingType.CLEAR }, margins: { top: 220, bottom: 220, left: 240, right: 240 }, children: [
        new Paragraph({ spacing: { after: 90 }, children: [new TextRun({ text: "质量—来源解耦  ×  偏好学习  ×  未见生成器", color: WHITE, bold: true, size: 26 })] }),
        new Paragraph({ spacing: { after: 0 }, children: [new TextRun({ text: "主问题：模型判断的是叙事质量，还是人类/机器来源痕迹？", color: "D8E5EF", size: 20 })] }),
      ] }),
    ] })],
  }),
  new Paragraph({ spacing: { before: 650, after: 90 }, children: [new TextRun({ text: "协议状态", color: MUTED, bold: true, size: 18 })] }),
  new Paragraph({ spacing: { after: 220 }, children: [new TextRun({ text: "可执行 pilot 已实现；正式论文结果需完成预注册规模的人类标注与泛化实验。", color: INK, size: 21 })] }),
  new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: "版本日期  2026-08-21", color: MUTED, size: 18 })] }),
  new Paragraph({ spacing: { after: 0 }, children: [new TextRun({ text: "项目仓库  Narration-Evaluation", color: MUTED, size: 18 })] }),
  new Paragraph({ children: [new PageBreak()] }),
  new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("目录")] }),
  manualToc,
  new Paragraph({ children: [new PageBreak()] }),
];

const appendixHeading = new Paragraph({
  heading: HeadingLevel.HEADING_1,
  pageBreakBefore: true,
  children: [new TextRun("附录 A：数据集与基线调研")],
});

const doc = new Document({
  creator: "Narration-Evaluation Research Team",
  title: "中文叙事质量评估器：论文与实验协议 v2.0",
  subject: "Source-quality disentanglement, pairwise preference learning, and unseen-generator evaluation",
  description: "A preregistration-ready protocol for testing whether a Chinese narrative evaluator predicts quality rather than source cues.",
  numbering: { config: Array.from({ length: 40 }, (_, index) => ({
    reference: `main-numbering-${index + 1}`,
    levels: [{ level: 0, format: "decimal", text: "%1.", alignment: AlignmentType.START, style: { paragraph: { indent: { left: 420, hanging: 240 } } } }],
  })) },
  styles: {
    default: {
      document: { run: { font: CJK_FONT, size: 21, color: INK }, paragraph: { spacing: { line: 340 } } },
      heading1: { run: { font: CJK_FONT, size: 32, bold: true, color: NAVY }, paragraph: { spacing: { before: 260, after: 150 }, keepNext: true } },
      heading2: { run: { font: CJK_FONT, size: 26, bold: true, color: NAVY }, paragraph: { spacing: { before: 220, after: 110 }, keepNext: true } },
      heading3: { run: { font: CJK_FONT, size: 22, bold: true, color: VERMILION }, paragraph: { spacing: { before: 170, after: 90 }, keepNext: true } },
      heading4: { run: { font: CJK_FONT, size: 20, bold: true, color: INK }, paragraph: { spacing: { before: 140, after: 70 }, keepNext: true } },
    },
  },
  sections: [{
    properties: {
      page: { size: { width: 11906, height: 16838 }, margin: { top: 1080, right: 1080, bottom: 1000, left: 1080 } },
    },
    headers: { default: new Header({ children: [new Paragraph({
      border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: LINE, space: 6 } },
      children: [new TextRun({ text: "NARRATIVE EVALUATION", color: NAVY, bold: true, size: 16 }), new TextRun({ text: "   /   论文与实验协议 v2.0", color: MUTED, size: 16 })],
    })] }) },
    footers: { default: new Footer({ children: [new Paragraph({
      alignment: AlignmentType.RIGHT,
      children: [new TextRun({ text: "研究协议  ·  ", color: MUTED, size: 16 }), new TextRun({ children: [PageNumber.CURRENT], color: NAVY, bold: true, size: 16 })],
    })] }) },
    children: [
      ...cover,
      ...parseMarkdown(protocol),
      appendixHeading,
      ...parseMarkdown(survey, 1, true),
    ],
  }],
});

fs.mkdirSync(path.dirname(outputPath), { recursive: true });
Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(outputPath, buffer);
  console.log(outputPath);
});
