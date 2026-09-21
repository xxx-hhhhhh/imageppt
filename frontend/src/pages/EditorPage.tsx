import { useEffect, useState } from 'react';
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
  const { project, slides, activePage, selectedIds, busy, message, conversionMode, create, upload, analyze, setConversionMode, persistPage, updateActive, setSelection, setActivePage, deletePage, movePage, addShape, deleteSelected, copySelected, moveLayer, export: exportProject } = useProjectStore();

  useEffect(() => { if (!project) void create(); }, [project, create]);
  useEffect(() => { void getVisionStatus().then(setVisionStatus).catch(() => setVisionStatus(null)); }, []);
  useEffect(() => {
    if (!project || !slides.length) { setSimilarity(null); return; }
    void getVisualScore(project.id).then((score) => setSimilarity(typeof score.overall === 'number' ? score.overall : null)).catch(() => setSimilarity(null));
  }, [project?.id, slides.length]);
  const page = slides[activePage];
  const handle = async (operation: () => Promise<void>) => { try { setError(''); await operation(); } catch (err) { setError(err instanceof Error ? err.message : '操作失败'); } };
  const saveCurrent = (next: LayoutJSON) => { updateActive(() => next); };
  const exportPpt = async () => { const url = await exportProject(); window.open(url, '_blank'); };
  const comparisonFile = viewMode === 'original' ? 'original.png' : 'difference.png';

  const renderCompare = () => <div className="comparison-view">
    <div className="comparison-stage">
      <img className="comparison-image comparison-base" src={artifactUrl(project!.id, 'reconstructed_preview.png')} alt="重建结果" />
      <div className="comparison-original-clip" style={{ width: `${compareSplit}%` }}><img className="comparison-image comparison-original" src={artifactUrl(project!.id, 'original.png')} alt="原图对比层" /></div>
    </div>
    <label className="compare-slider">原图 <input type="range" min="0" max="100" value={compareSplit} onChange={(event) => setCompareSplit(Number(event.target.value))} /> 重建 <span>{compareSplit}%</span></label>
  </div>;

  return <div className="app-shell">
    <Toolbar busy={busy} conversionMode={conversionMode} viewMode={viewMode} visionStatus={visionStatus} onModeChange={setConversionMode} onViewChange={setViewMode} onOpenSettings={() => setSettingsOpen(true)} onUpload={(files) => void handle(() => upload(files))} onAnalyze={() => void handle(analyze)} onExport={() => void handle(exportPpt)} onAddShape={addShape} onDelete={deleteSelected} onCopy={copySelected} onLayer={moveLayer} />
    <div className="workspace">
      <PageList slides={slides} activePage={activePage} onSelect={setActivePage} onDelete={deletePage} onMove={movePage} />
      <main className="editor-area">
        <div className="status-row"><span>{project?.name || 'Image2EditablePPT'}</span><span className={busy ? 'status busy' : 'status'}>{error || message}</span></div>
        {page ? (viewMode === 'reconstruction' || !project ? <EditorCanvas page={page} selectedIds={selectedIds} onSelection={setSelection} onChange={saveCurrent} /> : viewMode === 'difference' ? renderCompare() : <div className="comparison-view"><img className="comparison-image" src={artifactUrl(project.id, comparisonFile)} alt="原图" /></div>) : <div className="welcome"><div className="welcome-icon">P</div><h1>图片转可编辑 PPT</h1><p>上传 JPG、JPEG、PNG 或 WEBP，然后开始 OCR 解析。</p><label className="dropzone">选择图片<input hidden type="file" multiple accept="image/png,image/jpeg,image/webp" onChange={(event) => { const files = event.currentTarget.files; if (files) void handle(() => upload(Array.from(files))); event.currentTarget.value = ''; }} /></label></div>}
        <div className="editor-footer"><span>OCR: RapidOCR · AI: {visionStatus?.configured && visionStatus.model ? visionStatus.model : '本地模式'} · 视觉相似度：{similarity === null ? '—' : `${Math.round(similarity * 100)}%`}</span><button onClick={() => page && void handle(() => persistPage(activePage, page))} disabled={!page || busy}>保存当前页面</button></div>
      </main>
      <PropertyPanel page={page} selectedIds={selectedIds} onChange={(updater) => { if (page) updateActive(updater); }} />
    </div>
    <AISettingsModal open={settingsOpen} onClose={() => setSettingsOpen(false)} onSaved={() => void getVisionStatus().then(setVisionStatus).catch(() => undefined)} />
  </div>;
}
