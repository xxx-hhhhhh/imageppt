import type { LayoutJSON } from '../types/layout';
import { artifactUrl, assetUrl } from '../services/api';

interface Props { slides: LayoutJSON[]; activePage: number; approvedPages: number[]; projectId?: string; imageCount: number; busy: boolean; onSelect: (page: number) => void; onAdd: (files: File[]) => void; }

export function PageList({ slides, activePage, approvedPages, projectId, imageCount, busy, onSelect, onAdd }: Props) {
  return <aside className="page-list" aria-label="幻灯片页面"><div className="panel-title">幻灯片页面 <span>{imageCount || slides.length} 张</span></div>{slides.length === 0 ? <div className="empty-state">上传图片后<br />这里会显示页面缩略图</div> : <div className="thumbs">{slides.map((slide, index) => {
    const preview = slide.backgroundUrl || slide.elements.find((element) => element.type === 'image' || element.type === 'background')?.src || (projectId ? artifactUrl(projectId, index === 0 ? 'original.png' : `original_${index + 1}.png`) : null);
    return <div key={`${slide.source || 'page'}-${index}`} className={`thumb ${activePage === index ? 'active' : ''}`}><span className="thumb-number">{index + 1}</span><button className="thumb-select" onClick={() => onSelect(index)} aria-label={`查看第 ${index + 1} 页`} aria-current={activePage === index ? 'page' : undefined}><div className="thumb-canvas">{preview && <img src={assetUrl(preview)} alt="" />}{!preview && <span>{index + 1}</span>}</div><span className="thumb-caption">第 {index + 1} 页 <small>{approvedPages.includes(index + 1) ? '已通过' : '待确认'}</small></span></button></div>;
  })}</div>}<label className={`add-page ${busy ? 'disabled' : ''}`} role="button" tabIndex={busy ? -1 : 0} onKeyDown={(event) => { if (!busy && (event.key === 'Enter' || event.key === ' ')) { event.preventDefault(); event.currentTarget.querySelector('input')?.click(); } }}>＋ 添加页面<input hidden disabled={busy} type="file" multiple accept="image/png,image/jpeg,image/webp" onChange={(event) => { if (event.currentTarget.files?.length) onAdd(Array.from(event.currentTarget.files)); event.currentTarget.value = ''; }} /></label></aside>;
}
