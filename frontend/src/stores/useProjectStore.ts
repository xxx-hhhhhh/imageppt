import { create } from 'zustand';
import axios from 'axios';
import type { LayoutElement, LayoutJSON, PagePreview, ProjectInfo } from '../types/layout';
import { acceptCurrentResult, approvePage, artifactUrl, createProject, downgradePage, exportPptx, revisePage, saveSlide, uploadImages } from '../services/api';
import { cancelTask, readTask, restoreProject, startTask, taskIsActive, type AnalysisTask, type RestoredProject } from '../services/analysisTasks';

interface ProjectState {
  project: ProjectInfo | null;
  slides: LayoutJSON[];
  pagePreviews: PagePreview[];
  activePage: number;
  selectedIds: string[];
  busy: boolean;
  task: AnalysisTask | null;
  taskDisconnected: boolean;
  cancelAnalysis: () => Promise<void>;
  reconnectAnalysis: () => Promise<void>;
  runAnalysis: (fallback?: boolean) => Promise<void>;
  message: string;
  aiPaused: boolean;
  reviewVersion: number;
  approvedPages: number[];
  conversionMode: 'fast' | 'standard' | 'high_quality' | 'maximum';
  create: () => Promise<void>;
  upload: (files: File[]) => Promise<void>;
  analyze: () => Promise<void>;
  approveAndNext: () => Promise<void>;
  useBasicFallback: () => Promise<void>;
  reanalyzeCurrent: () => Promise<void>;
  downgradeCurrent: () => Promise<void>;
  acceptCurrent: () => Promise<void>;
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
const PROJECT_KEY = 'image2editableppt.projectId';
let initializing: Promise<void> | null = null;
let watching: string | null = null;
function restoredState(data: RestoredProject) {
  return { project: data.project, slides: data.slides, approvedPages: data.approvedPages,
    pagePreviews: data.images.map((image, index) => ({ id: image.id, originalPreviewUrl: image.source_url,
      resultPreviewUrl: data.slides[index] ? `${artifactUrl(data.project.id, index === 0 ? 'reconstructed_preview.png' : `reconstructed_preview_${index + 1}.png`)}?v=${Date.now()}` : null,
      status: data.slides[index] ? 'ready' as const : 'waiting' as const })) };
}
async function watchTask(initial: AnalysisTask) {
  if (watching === initial.id) return;
  watching = initial.id;
  try {
    let task = initial;
    while (true) {
      if (useProjectStore.getState().project?.id !== task.projectId) return;
      useProjectStore.setState({ task, taskDisconnected: false, busy: taskIsActive(task),
        message: taskIsActive(task) ? `第 ${task.page} 页：${task.stageLabel} · 已用 ${task.elapsedSeconds} 秒 / 上限 ${task.deadlineSeconds} 秒` : task.message });
      if (!taskIsActive(task)) {
        if (task.status === 'succeeded') {
          const data = await restoreProject(task.projectId);
          // Keep unsaved edits to other pages; only the analyzed page changes.
          useProjectStore.setState(state => ({ ...restoredState(data),
            slides: data.slides.map((slide, index) => index === task.page - 1 ? slide : state.slides[index] || slide),
            activePage: task.page - 1, reviewVersion: state.reviewVersion + 1, busy: false, aiPaused: false }));
        } else {
          useProjectStore.setState({ busy: false, aiPaused: ['failed', 'timed_out', 'interrupted'].includes(task.status) });
        }
        return;
      }
      await new Promise(resolve => window.setTimeout(resolve, 1500));
      task = await readTask(task.projectId, task.id);
    }
  } catch {
    // Unknown server state is NOT a failed job. Do not offer a duplicate submit.
    useProjectStore.setState({ taskDisconnected: true, busy: true,
      message: '任务连接暂时中断，后台可能仍在处理。点击“恢复连接”或刷新，不会重复开始分析。' });
  } finally { watching = null; }
}

export const useProjectStore = create<ProjectState>((set, get) => ({
  project: null,
  slides: [],
  pagePreviews: [],
  activePage: 0,
  selectedIds: [],
  busy: false,
  task: null,
  taskDisconnected: false,
  message: '请上传一张或多张图片开始',
  aiPaused: false,
  reviewVersion: 0,
  approvedPages: [],
  conversionMode: 'maximum',
  create: async () => {
    if (get().project) return;
    if (initializing) return initializing;
    initializing = (async () => {
      set({ busy: true, message: '正在恢复工作区…' });
      try {
        const id = new URLSearchParams(window.location.search).get('project') || localStorage.getItem(PROJECT_KEY);
        if (id) {
          try {
            const data = await restoreProject(id);
            localStorage.setItem(PROJECT_KEY, id);
            set({ ...restoredState(data), activePage: Math.max(0, (data.task?.page || data.slides.length || 1) - 1),
              busy: taskIsActive(data.task), task: data.task, message: '已恢复上次项目。' });
            if (data.task) void watchTask(data.task);
            return;
          } catch (error) {
            if (!axios.isAxiosError(error) || error.response?.status !== 404) throw error;
            localStorage.removeItem(PROJECT_KEY);
          }
        }
        const project = await createProject();
        localStorage.setItem(PROJECT_KEY, project.id);
        set({ project, busy: false, message: '项目已创建，请上传图片' });
      } catch { set({ busy: false, message: '无法恢复工作区，请检查服务后刷新；没有创建重复项目。' }); }
    })();
    try { await initializing; } finally { initializing = null; }
  },
  upload: async (files) => {
    const current = get().project;
    if (!current) await get().create();
    const project = get().project;
    if (!project) return;
    set({ busy: true, message: `正在上传 ${files.length} 张图片…` });
    try {
      const uploaded = await uploadImages(project.id, files);
      set((state) => ({ project: uploaded.project, pagePreviews: [...state.pagePreviews, ...uploaded.images.map((image) => ({ id: image.id, originalPreviewUrl: image.source_url, resultPreviewUrl: null, status: 'waiting' as const }))], aiPaused: false, busy: false, message: `${files.length} 张图片已添加，可以开始逐页解析` }));
    } catch (error) {
      set({ busy: false, message: '图片上传失败，请重试。' });
      throw error;
    }
  },
  analyze: async () => {
    await get().runAnalysis();
  },
  runAnalysis: async (fallback = false) => {
    const project = get().project;
    if (!project || get().busy || taskIsActive(get().task)) return;
    const page = get().task && get().task!.status !== 'succeeded' ? get().task!.page : Math.min(get().slides.length + 1, project.imageCount);
    set({ busy: true, aiPaused: false, message: `正在重建第 ${page} / ${project.imageCount} 页…` });
    try {
      const task = await startTask(project.id, page, fallback ? 'fast' : get().conversionMode, fallback);
      set({ task, activePage: page - 1 });
      void watchTask(task);
    } catch (error) {
      if (axios.isAxiosError(error) && !error.response) {
        // POST may have reached the server even if its response was lost.
        // Recover its id before allowing another submit or calling it failed.
        try {
          const restored = await restoreProject(project.id);
          if (restored.task && taskIsActive(restored.task)) {
            set({ task: restored.task });
            void watchTask(restored.task);
            return;
          }
        } catch {
          set({ busy: true, taskDisconnected: true,
            message: '未收到提交确认，任务可能已启动。请恢复连接，不要重复提交。' });
          return;
        }
      }
      const detail = axios.isAxiosError(error) ? error.response?.data?.detail : null;
      set({ busy: false, aiPaused: detail?.code === 'AI_UNAVAILABLE', message: typeof detail?.message === 'string' ? detail.message : '本页重建失败，请重试。' });
      throw error;
    }
  },
  cancelAnalysis: async () => {
    const task = get().task;
    if (!task || !taskIsActive(task)) return;
    try {
      const result = await cancelTask(task.projectId, task.id);
      set({ task: result, busy: taskIsActive(result), taskDisconnected: false, message: result.message });
    } catch { set({ taskDisconnected: true, message: '未确认取消成功，请恢复连接后重试；后台任务仍受总时限保护。' }); }
  },
  reconnectAnalysis: async () => {
    const { project } = get();
    if (!project) { await get().create(); return; }
    try {
      const data = await restoreProject(project.id);
      if (data.task) await watchTask(data.task);
      else set({ busy: false, taskDisconnected: false });
    } catch { set({ taskDisconnected: true, message: '服务仍未连接，请检查后再次恢复。' }); }
  },
  approveAndNext: async () => {
    const { project, slides, activePage } = get();
    if (!project || !slides[activePage]) return;
    set((state) => ({ busy: true, approvedPages: [...new Set([...state.approvedPages, activePage + 1])], message: `正在确认第 ${activePage + 1} 页…` }));
    try {
      await saveSlide(project.id, activePage + 1, slides[activePage]);
      await approvePage(project.id, activePage + 1);
      set({ busy: false, message: `第 ${activePage + 1} 页已通过。` });
    } catch (error) {
      set((state) => ({ busy: false, approvedPages: state.approvedPages.filter((page) => page !== activePage + 1), message: '页面确认失败，请重试。' }));
      throw error;
    }
    if (activePage + 1 < project.imageCount) await get().analyze();
  },
  useBasicFallback: async () => {
    await get().runAnalysis(true);
  },
  reanalyzeCurrent: async () => {
    const { project, activePage } = get();
    if (!project) return;
    set({ busy: true, message: `继续优化第 ${activePage + 1} 页…` });
    try {
      const result = await revisePage(project.id, activePage + 1);
      set((state) => ({ slides: state.slides.map((slide, index) => index === activePage ? result.layout : slide), pagePreviews: state.pagePreviews.map((item, index) => index === activePage && result.accepted ? { ...item, resultPreviewUrl: `${artifactUrl(project.id, activePage === 0 ? 'reconstructed_preview.png' : `reconstructed_preview_${activePage + 1}.png`)}?v=${state.reviewVersion + 1}` } : item), busy: false, reviewVersion: state.reviewVersion + 1, message: result.accepted ? `第 ${result.revisionRound} 轮区域优化已改善 ${result.improvedRegions.length} 处，请对照检查。` : `第 ${result.revisionRound} 轮未改善指标，已保留上一轮结果。` }));
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
      set((state) => ({ slides: state.slides.map((slide, index) => index === activePage ? layout : slide), pagePreviews: state.pagePreviews.map((item, index) => index === activePage ? { ...item, resultPreviewUrl: `${artifactUrl(project.id, activePage === 0 ? 'reconstructed_preview.png' : `reconstructed_preview_${activePage + 1}.png`)}?v=${state.reviewVersion + 1}` } : item), busy: false, reviewVersion: state.reviewVersion + 1, message: '问题区域已降级，请检查并确认。' }));
    } catch (error) {
      set({ busy: false, message: '问题区域降级失败，当前结果已保留。' });
      throw error;
    }
  },
  acceptCurrent: async () => {
    const { project, activePage } = get();
    if (!project) return;
    await acceptCurrentResult(project.id, activePage + 1);
    set((state) => ({ reviewVersion: state.reviewVersion + 1, message: '已使用当前结果，可继续编辑或导出。' }));
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
    try {
      const result = await exportPptx(project.id);
      set({ busy: false, message: 'PPTX 导出完成' });
      return result.downloadUrl;
    } catch (error) {
      set({ busy: false, message: '导出失败，请检查页面后重试。' });
      throw error;
    }
  },
}));
