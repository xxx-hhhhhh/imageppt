import { create } from 'zustand';
import axios from 'axios';
import type { LayoutElement, LayoutJSON, ProjectInfo } from '../types/layout';
import { analyzeProject, approvePage, createProject, downgradePage, exportPptx, saveSlide, uploadImages } from '../services/api';

interface ProjectState {
  project: ProjectInfo | null;
  slides: LayoutJSON[];
  activePage: number;
  selectedIds: string[];
  busy: boolean;
  message: string;
  aiPaused: boolean;
  reviewVersion: number;
  conversionMode: 'fast' | 'standard' | 'high_quality' | 'maximum';
  create: () => Promise<void>;
  upload: (files: File[]) => Promise<void>;
  analyze: () => Promise<void>;
  approveAndNext: () => Promise<void>;
  useBasicFallback: () => Promise<void>;
  reanalyzeCurrent: () => Promise<void>;
  downgradeCurrent: () => Promise<void>;
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
  aiPaused: false,
  reviewVersion: 0,
  conversionMode: 'maximum',
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
    set({ project: updated, slides: [], activePage: 0, aiPaused: false, busy: false, message: '图片已上传，开始逐页解析' });
  },
  analyze: async () => {
    const project = get().project;
    if (!project) return;
    const page = Math.min(get().slides.length + 1, project.imageCount);
    set({ busy: true, aiPaused: false, message: `正在重建第 ${page} / ${project.imageCount} 页…` });
    try {
      const result = await analyzeProject(project.id, get().conversionMode, page);
      set((state) => ({ project: result.project, slides: [...state.slides.slice(0, page - 1), result.slides[0]], activePage: page - 1, busy: false, reviewVersion: state.reviewVersion + 1, message: `第 ${page} 页已重建，请对照原图确认。${result.warnings[0] || ''}` }));
    } catch (error) {
      const detail = axios.isAxiosError(error) ? error.response?.data?.detail : null;
      set({ busy: false, aiPaused: detail?.code === 'AI_UNAVAILABLE', message: typeof detail?.message === 'string' ? detail.message : '本页重建失败，请重试。' });
      throw error;
    }
  },
  approveAndNext: async () => {
    const { project, slides, activePage } = get();
    if (!project || !slides[activePage]) return;
    set({ busy: true, message: `正在确认第 ${activePage + 1} 页…` });
    try {
      await saveSlide(project.id, activePage + 1, slides[activePage]);
      await approvePage(project.id, activePage + 1);
      set({ busy: false, message: `第 ${activePage + 1} 页已通过。` });
      if (activePage + 1 < project.imageCount) await get().analyze();
    } catch (error) {
      set({ busy: false, message: '页面确认失败，请重试。' });
      throw error;
    }
  },
  useBasicFallback: async () => {
    const project = get().project;
    if (!project) return;
    const page = Math.min(get().slides.length + 1, project.imageCount);
    set({ busy: true, aiPaused: false, message: `正在以基础模式处理第 ${page} 页…` });
    try {
      const result = await analyzeProject(project.id, get().conversionMode, page, true);
      set((state) => ({ slides: [...state.slides.slice(0, page - 1), result.slides[0]], activePage: page - 1, busy: false, reviewVersion: state.reviewVersion + 1, message: '本页已使用基础模式，请仔细核对视觉结果。' }));
    } catch (error) {
      set({ busy: false, aiPaused: true, message: '基础模式也未能完成，请检查设置并重试。' });
      throw error;
    }
  },
  reanalyzeCurrent: async () => {
    const { project, activePage, conversionMode } = get();
    if (!project) return;
    set({ busy: true, message: `继续优化第 ${activePage + 1} 页…` });
    try {
      const result = await analyzeProject(project.id, conversionMode, activePage + 1);
      set((state) => ({ slides: state.slides.map((slide, index) => index === activePage ? result.slides[0] : slide), busy: false, reviewVersion: state.reviewVersion + 1, message: '本页已重新优化，请对照检查。' }));
    } catch (error) {
      set({ busy: false, message: '继续优化失败，当前结果已保留。' });
      throw error;
    }
  },
  downgradeCurrent: async () => {
    const { project, activePage } = get();
    if (!project) return;
    set({ busy: true, message: '正在将问题视觉区域改为图片主体与可编辑文字…' });
    try {
      const layout = await downgradePage(project.id, activePage + 1);
      set((state) => ({ slides: state.slides.map((slide, index) => index === activePage ? layout : slide), busy: false, reviewVersion: state.reviewVersion + 1, message: '问题区域已降级，请检查并确认。' }));
    } catch (error) {
      set({ busy: false, message: '问题区域降级失败，当前结果已保留。' });
      throw error;
    }
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
