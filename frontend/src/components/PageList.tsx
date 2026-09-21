import type { LayoutJSON } from '../types/layout';
import { assetUrl } from '../services/api';

interface Props { slides: LayoutJSON[]; activePage: number; onSelect: (page: number) => void; onDelete: (page: number) => void; onMove: (page: number, direction: 'up' | 'down') => void; }

export function PageList({ slides, activePage, onSelect, onDelete, onMove }: Props) {
  return <aside className="page-list"><div className="panel-title">页面 <span>{slides.length}</span></div>{slides.length === 0 ? <div className="empty-state">上传图片后<br />这里会显示缩略图</div> : <div className="thumbs">{slides.map((slide, index) => {
    const preview = slide.backgroundUrl || slide.elements.find((element) => element.type === 'image' || element.type === 'background')?.src;
    return <div key={`${slide.source}-${index}`} className={`thumb ${activePage === index ? 'active' : ''}`}><button className="thumb-select" onClick={() => onSelect(index)}><div className="thumb-canvas">{preview && <img src={assetUrl(preview)} alt={`Page ${index + 1}`} />}{!preview && <span>{index + 1}</span>}</div><span>Page {index + 1}</span></button><div className="thumb-actions"><button onClick={() => onMove(index, 'up')} disabled={index === 0}>↑</button><button onClick={() => onMove(index, 'down')} disabled={index === slides.length - 1}>↓</button><button onClick={() => onDelete(index)} disabled={slides.length <= 1}>×</button></div></div>;
  })}</div>}</aside>;
}
