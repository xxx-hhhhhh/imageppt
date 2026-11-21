import type { ElementStyle, LayoutElement } from '../types/layout';

interface TextRun { text: string; x: number; y: number; width: number; height: number; style: ElementStyle }
interface MeasuredLine { text: string; width: number; height: number; style: ElementStyle; runs: TextRun[] }

export function measuredTextRuns(element: LayoutElement): TextRun[] {
  const data = element.metadata?.measuredTextLine as MeasuredLine | undefined;
  if (!data || !Array.isArray(data.runs) || data.text !== element.text) return [];
  const style = element.style || {};
  if (['fontFamily', 'fontWeight', 'fontStyle', 'color', 'letterSpacing', 'align'].some(k => style[k] !== data.style[k])) return [];
  const scale = (style.fontSize || 24) / (data.style.fontSize || 24);
  return data.runs.map(run => ({ ...run, x: run.x * element.width / data.width, y: run.y * element.height / data.height,
    style: { ...run.style, fontSize: (run.style.fontSize || 24) * scale, letterSpacing: (run.style.letterSpacing || 0) * scale } }));
}
