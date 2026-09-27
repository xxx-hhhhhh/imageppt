import type { PagePreview } from '../types/layout';
import { assetUrl } from '../services/api';

interface Props { pages: PagePreview[]; activePage: number; approvedPages: number[]; busy: boolean; onSelect: (page: number) => void; onAdd: (files: File[]) => void; }

export function PageList({ pages, activePage, approvedPages, busy, onSelect, onAdd }: Props) {
  return <aside className="page-list" aria-label="幻灯片页面"><div className="panel-title">幻灯片页面 <span>{pages.length} 张</span></div>{pages.length === 0 ? <div className="empty-state">上传图片后<br />这里会显示页面缩略图</div> : <div className="thumbs">{pages.map((page, index) => {
    return <div key={page.id} className={`thumb ${activePage === index ? 'active' : ''}`}><span className="thumb-number">{index + 1}</span><button className="thumb-select" onClick={() => onSelect(index)} aria-label={`查看第 ${index + 1} 页`} aria-current={activePage === index ? 'page' : undefined}><div className="thumb-canvas"><img src={assetUrl(page.originalPreviewUrl)} alt={`第 ${index + 1} 页原图缩略图`} /></div><span className="thumb-caption">第 {index + 1} 页 <small>{approvedPages.includes(index + 1) ? '已通过' : page.status === 'ready' ? '待确认' : '待解析'}</small></span></button></div>;
  })}</div>}<label className={`add-page ${busy ? 'disabled' : ''}`} role="button" tabIndex={busy ? -1 : 0} onKeyDown={(event) => { if (!busy && (event.key === 'Enter' || event.key === ' ')) { event.preventDefault(); event.currentTarget.querySelector('input')?.click(); } }}>＋ 添加页面<input hidden disabled={busy} type="file" multiple accept="image/png,image/jpeg,image/webp" onChange={(event) => { if (event.currentTarget.files?.length) onAdd(Array.from(event.currentTarget.files)); event.currentTarget.value = ''; }} /></label></aside>;
}
