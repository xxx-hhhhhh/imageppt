import { create } from 'zustand';
import type { LayoutElement, LayoutJSON, ProjectInfo } from '../types/layout';
import { analyzeProject, createProject, exportPptx, saveSlide, uploadImages } from '../services/api';

interface ProjectState {
  project: ProjectInfo | null;
  slides: LayoutJSON[];
  activePage: number;
  selectedIds: string[];
  busy: boolean;
  message: string;
  conversionMode: 'fast' | 'standard' | 'high_quality' | 'maximum';
  create: () => Promise<void>;
  upload: (files: File[]) => Promise<void>;
  analyze: () => Promise<void>;
  setConversionMode: (mode: 'fast' | 'standard' | 'high_quality' | 'maximum') => void;
  persistPage: (page: number, layout: LayoutJSON) => Promise<void>;
  updateActive: (updater: (layout: LayoutJSON) => LayoutJSON) => void;
  setSelection: (ids: string[]) => void;
  setActivePage: (page: number) => void;
  deletePage: (page: number) => void;
  movePage: (page: number, direction: 'up' | 'down') => void;
  addShape: (type: LayoutElement['type']) => void;
  deleteSelected: () => void;
  copySelected: () => void;
  moveLayer: (direction: 'up' | 'down' | 'top' | 'bottom') => void;
  export: () => Promise<string>;
}

const clone = <T,>(value: T): T => JSON.parse(JSON.stringify(value)) as T;

export const useProjectStore = create<ProjectState>((set, get) => ({
  project: null,
  slides: [],
  activePage: 0,
  selectedIds: [],
  busy: false,
  message: '请上传一张或多张图片开始',
  conversionMode: 'standard',
  create: async () => {
    set({ busy: true, message: '正在创建项目…' });
    const project = await createProject();
    set({ project, busy: false, message: '项目已创建，请上传图片' });
  },
  upload: async (files) => {
    const current = get().project;
    if (!current) await get().create();
    const project = get().project;
    if (!project) return;
    set({ busy: true, message: `正在上传 ${files.length} 张图片…` });
    const updated = await uploadImages(project.id, files);
    set({ project: updated, busy: false, message: '图片已上传，点击“开始解析”' });
  },
  analyze: async () => {
    const project = get().project;
    if (!project) return;
    set({ busy: true, message: '正在进行 OCR、版面分析和背景修复…' });
    const result = await analyzeProject(project.id, get().conversionMode);
    const aiLabel = result.aiUsed ? `AI：${result.visionModel || result.visionProvider || '视觉模型'}` : 'AI：基础模式';
    const fallbackLabel = result.fallbackCount ? ` · 已自动切换 ${result.fallbackCount} 次` : '';
    set({ project: result.project, slides: result.slides, activePage: 0, busy: false, message: result.warnings.length ? `OCR：${result.ocrProvider || result.provider} · ${aiLabel}${fallbackLabel} · ${result.warnings[0]}` : `OCR：${result.ocrProvider || result.provider} · ${aiLabel}${fallbackLabel}` });
  },
  setConversionMode: (conversionMode) => set({ conversionMode }),
  persistPage: async (page, layout) => {
    const project = get().project;
    if (!project) return;
    await saveSlide(project.id, page + 1, layout);
    set((state) => ({ slides: state.slides.map((item, index) => index === page ? layout : item) }));
  },
  updateActive: (updater) => set((state) => ({ slides: state.slides.map((item, index) => index === state.activePage ? updater(clone(item)) : item) })),
  setSelection: (selectedIds) => set({ selectedIds }),
  setActivePage: (activePage) => set({ activePage, selectedIds: [] }),
  deletePage: (page) => set((state) => {
    if (state.slides.length <= 1) return { message: '至少保留一个页面' };
    const slides = state.slides.filter((_, index) => index !== page);
    return { slides, activePage: Math.min(state.activePage, slides.length - 1), selectedIds: [] };
  }),
  movePage: (page, direction) => set((state) => {
    const target = direction === 'up' ? page - 1 : page + 1;
    if (target < 0 || target >= state.slides.length) return state;
    const slides = [...state.slides];
    [slides[page], slides[target]] = [slides[target], slides[page]];
    return { slides, activePage: target };
  }),
  addShape: (type) => {
    if (!['rectangle', 'roundedRectangle', 'ellipse', 'line', 'arrow'].includes(type)) return;
    const state = get();
    const layout = state.slides[state.activePage];
    if (!layout) return;
    const id = `manual_${type}_${Date.now()}`;
    const element: LayoutElement = { id, type, x: layout.slide.width * 0.35, y: layout.slide.height * 0.35, width: type === 'line' || type === 'arrow' ? 260 : 240, height: type === 'line' || type === 'arrow' ? 2 : 140, rotation: 0, zIndex: Math.max(0, ...layout.elements.map((item) => item.zIndex)) + 1, style: { fill: '#DCE6F1', stroke: '#17365D', strokeWidth: 2, opacity: 1 } };
    state.updateActive((current) => ({ ...current, elements: [...current.elements, element] }));
  },
  deleteSelected: () => {
    const ids = new Set(get().selectedIds);
    get().updateActive((layout) => ({ ...layout, elements: layout.elements.filter((element) => !ids.has(element.id) || element.type === 'background') }));
    set({ selectedIds: [] });
  },
  copySelected: () => {
    const ids = new Set(get().selectedIds);
    get().updateActive((layout) => {
      const copies = layout.elements.filter((element) => ids.has(element.id) && element.type !== 'background').map((element, index) => ({ ...clone(element), id: `${element.id}_copy_${Date.now()}_${index}`, x: element.x + 20, y: element.y + 20, zIndex: Math.max(0, ...layout.elements.map((item) => item.zIndex)) + index + 1 }));
      return { ...layout, elements: [...layout.elements, ...copies] };
    });
  },
  moveLayer: (direction) => {
    const ids = new Set(get().selectedIds);
    get().updateActive((layout) => {
      const elements = layout.elements.map(clone);
      const selected = elements.filter((element) => ids.has(element.id) && element.type !== 'background');
      if (!selected.length) return layout;
      if (direction === 'top') selected.forEach((element, index) => { element.zIndex = 1000 + index; });
      if (direction === 'bottom') selected.forEach((element, index) => { element.zIndex = 1 + index; });
      if (direction === 'up') selected.forEach((element) => { element.zIndex += 1; });
      if (direction === 'down') selected.forEach((element) => { element.zIndex = Math.max(1, element.zIndex - 1); });
      return { ...layout, elements };
    });
  },
  export: async () => {
    const project = get().project;
    if (!project) throw new Error('项目尚未创建');
    set({ busy: true, message: '正在生成对象级可编辑 PPTX…' });
    const result = await exportPptx(project.id);
    set({ busy: false, message: 'PPTX 导出完成' });
    return result.downloadUrl;
  },
}));
