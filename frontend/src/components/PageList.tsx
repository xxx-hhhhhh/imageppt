import type { LayoutJSON } from '../types/layout';
import { assetUrl } from '../services/api';

interface Props { slides: LayoutJSON[]; activePage: number; approvedPages: number[]; onSelect: (page: number) => void; }

export function PageList({ slides, activePage, approvedPages, onSelect }: Props) {
  return <aside className="page-list"><div className="panel-title">页面 <span>{slides.length}</span></div>{slides.length === 0 ? <div className="empty-state">上传图片后<br />这里会显示缩略图</div> : <div className="thumbs">{slides.map((slide, index) => {
    const preview = slide.backgroundUrl || slide.elements.find((element) => element.type === 'image' || element.type === 'background')?.src;
    return <div key={`${slide.source}-${index}`} className={`thumb ${activePage === index ? 'active' : ''}`}><button className="thumb-select" onClick={() => onSelect(index)} aria-current={activePage === index ? 'page' : undefined}><div className="thumb-canvas">{preview && <img src={assetUrl(preview)} alt="" />}{!preview && <span>{index + 1}</span>}</div><span>第 {index + 1} 页 <small>{approvedPages.includes(index + 1) ? '已通过' : '待确认'}</small></span></button></div>;
  })}</div>}</aside>;
}
