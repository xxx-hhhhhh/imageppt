import { useEffect, useState } from 'react';
import axios from 'axios';
import { AISettingsModal } from '../components/AISettingsModal';
import { EditorCanvas } from '../editor/EditorCanvas';
import { PageList } from '../components/PageList';
import { PropertyPanel } from '../components/PropertyPanel';
import { Toolbar } from '../components/Toolbar';
import { artifactUrl, getVisionStatus, getVisualScore, type VisionStatus } from '../services/api';
import { useProjectStore } from '../stores/useProjectStore';
import type { LayoutJSON } from '../types/layout';

export function EditorPage() {
  const [error, setError] = useState('');
  const [viewMode, setViewMode] = useState<'reconstruction' | 'original' | 'difference'>('reconstruction');
  const [visionStatus, setVisionStatus] = useState<VisionStatus | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [compareSplit, setCompareSplit] = useState(50);
  const [similarity, setSimilarity] = useState<number | null>(null);
  const [revisionStatus, setRevisionStatus] = useState('');
  const [acceptedCurrent, setAcceptedCurrent] = useState(false);
  const { project, slides, activePage, selectedIds, busy, message, aiPaused, reviewVersion, conversionMode, create, upload, analyze, approveAndNext, useBasicFallback, reanalyzeCurrent, downgradeCurrent, setConversionMode, persistPage, updateActive, setSelection, setActivePage, deletePage, movePage, addShape, deleteSelected, copySelected, moveLayer, export: exportProject } = useProjectStore();

  useEffect(() => { if (!project) void create(); }, [project, create]);
  useEffect(() => { void getVisionStatus().then(setVisionStatus).catch(() => setVisionStatus(null)); }, []);
  useEffect(() => {
    if (!project || !slides.length) { setSimilarity(null); return; }
    setAcceptedCurrent(false);
    void getVisualScore(project.id, activePage + 1).then((score) => { setSimilarity(typeof score.overall === 'number' ? score.overall : null); setRevisionStatus(score.revisionStatus || ''); }).catch(() => { setSimilarity(null); setRevisionStatus(''); });
  }, [project?.id, slides.length, activePage, reviewVersion]);
  const page = slides[activePage];
  const handle = async (operation: () => Promise<void>) => { try { setError(''); await operation(); } catch (err) { const detail = axios.isAxiosError(err) ? err.response?.data?.detail : null; setError(typeof detail?.message === 'string' ? detail.message : typeof detail === 'string' ? detail : err instanceof Error ? err.message : '操作失败'); } };
  const saveCurrent = (next: LayoutJSON) => { updateActive(() => next); };
  const exportPpt = async () => { const url = await exportProject(); window.open(url, '_blank'); };
  const comparisonFile = viewMode === 'original' ? (activePage === 0 ? 'original.png' : `original_${activePage + 1}.png`) : 'difference.png';
  const previewFile = activePage === 0 ? 'reconstructed_preview.png' : `reconstructed_preview_${activePage + 1}.png`;

  const renderCompare = () => <div className="comparison-view">
    <div className="comparison-stage">
      <img className="comparison-image comparison-base" src={artifactUrl(project!.id, previewFile)} alt="重建结果" />
      <div className="comparison-original-clip" style={{ width: `${compareSplit}%` }}><img className="comparison-image comparison-original" src={artifactUrl(project!.id, activePage === 0 ? 'original.png' : `original_${activePage + 1}.png`)} alt="原图对比层" /></div>
    </div>
    <label className="compare-slider">原图 <input type="range" min="0" max="100" value={compareSplit} onChange={(event) => setCompareSplit(Number(event.target.value))} /> 重建 <span>{compareSplit}%</span></label>
  </div>;

  return <div className="app-shell">
    <Toolbar busy={busy} conversionMode={conversionMode} viewMode={viewMode} visionStatus={visionStatus} onModeChange={setConversionMode} onViewChange={setViewMode} onOpenSettings={() => setSettingsOpen(true)} onUpload={(files) => void handle(() => upload(files))} onAnalyze={() => void handle(analyze)} onExport={() => void handle(exportPpt)} onAddShape={addShape} onDelete={deleteSelected} onCopy={copySelected} onLayer={moveLayer} />
    <div className="workspace">
      <PageList slides={slides} activePage={activePage} onSelect={setActivePage} onDelete={deletePage} onMove={movePage} />
      <main className="editor-area">
        <div className="status-row"><span>{project?.name || 'Image2EditablePPT'}</span><span className={busy ? 'status busy' : 'status'}>{error || message}</span></div>
        {aiPaused && <div className="review-actions" role="alert"><span>AI 未完成本页规划，处理已暂停。</span><button disabled={busy} onClick={() => void handle(analyze)}>重试 AI</button><button disabled={busy} onClick={() => void handle(useBasicFallback)}>明确使用基础模式</button></div>}
        {page && revisionStatus === 'stagnated' && !acceptedCurrent && <div className="review-actions" role="alert"><span>自动优化已停滞。当前结果已保留。</span><button disabled={busy} onClick={() => void handle(reanalyzeCurrent)}>继续自动优化</button><button disabled={busy} onClick={() => void handle(downgradeCurrent)}>问题区域改为图片主体＋可编辑文字</button><button disabled={busy} onClick={() => setAcceptedCurrent(true)}>使用当前结果</button></div>}
        {page && <div className="review-actions"><span>第 {activePage + 1} / {project?.imageCount || slides.length} 页 · 请确认视觉与可编辑文字</span><button disabled={busy} onClick={() => void handle(approveAndNext)}>{activePage + 1 < (project?.imageCount || 0) ? '通过并处理下一页' : '通过本页'}</button><button disabled={busy} onClick={() => setViewMode('reconstruction')}>手动编辑</button></div>}
        {page ? (viewMode === 'reconstruction' || !project ? <EditorCanvas page={page} selectedIds={selectedIds} onSelection={setSelection} onChange={saveCurrent} /> : viewMode === 'difference' ? renderCompare() : <div className="comparison-view"><img className="comparison-image" src={artifactUrl(project.id, comparisonFile)} alt="原图" /></div>) : <div className="welcome"><div className="welcome-icon">P</div><h1>图片转可编辑 PPT</h1><p>上传 JPG、JPEG、PNG 或 WEBP，然后开始 OCR 解析。</p><label className="dropzone">选择图片<input hidden type="file" multiple accept="image/png,image/jpeg,image/webp" onChange={(event) => { const files = event.currentTarget.files; if (files) void handle(() => upload(Array.from(files))); event.currentTarget.value = ''; }} /></label></div>}
        <div className="editor-footer"><span>OCR: RapidOCR · AI: {visionStatus?.configured && visionStatus.model ? visionStatus.model : '本地模式'} · 视觉相似度：{similarity === null ? '—' : `${Math.round(similarity * 100)}%`}</span><button onClick={() => page && void handle(() => persistPage(activePage, page))} disabled={!page || busy}>保存当前页面</button></div>
      </main>
      <PropertyPanel page={page} selectedIds={selectedIds} onChange={(updater) => { if (page) updateActive(updater); }} />
    </div>
    <AISettingsModal open={settingsOpen} onClose={() => setSettingsOpen(false)} onSaved={() => void getVisionStatus().then(setVisionStatus).catch(() => undefined)} />
  </div>;
}
