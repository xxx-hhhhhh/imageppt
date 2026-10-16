import type { LayoutElement, LayoutJSON } from '../types/layout';

export const visibleElement = (e: LayoutElement) => e.type !== 'group' && e.type !== 'background' &&
  !e.metadata?.suppressed && !e.metadata?.suppressRender && !e.metadata?.ownedBy;

// Semantic membership is shared with export; selecting a module is explicit.
// A normal click must still allow editing an individual photograph or text.
export function moduleIds(page: LayoutJSON, id: string): string[] {
  const element = page.elements.find(e => e.id === id);
  if (!element) return [];
  const legacy = page.elements.find(e => e.metadata?.layerRole === 'container' &&
    (e.id === id || (Array.isArray(e.metadata.moduleMemberIds) && e.metadata.moduleMemberIds.includes(id))));
  const listed = legacy?.metadata?.moduleMemberIds;
  const explicit = legacy ? [legacy.id, ...(Array.isArray(listed) ? listed.filter((id): id is string => typeof id === 'string') : [])] : [];
  const group = element.groupId || element.parent;
  const descendants = new Set<string>(group ? [group] : []);
  const plan = page.metadata?.reconstructionPlan as { modules?: unknown[] } | undefined;
  const visit = (items: unknown[], inside = false) => {
    for (const value of items) {
      if (!value || typeof value !== 'object') continue;
      const item = value as { moduleId?: string; id?: string; children?: unknown[] };
      const id = item.moduleId || item.id;
      const included = inside || (Boolean(group) && id === group);
      if (included && id) descendants.add(id);
      if (Array.isArray(item.children)) visit(item.children, included);
    }
  };
  if (Array.isArray(plan?.modules)) visit(plan.modules);
  return page.elements.filter(e => visibleElement(e) && (explicit.length ? explicit.includes(e.id) :
    group ? descendants.has(e.groupId || e.parent || '') : e.id === id)).map(e => e.id);
}

export function selectionBounds(page: LayoutJSON, ids: string[]) {
  const items = page.elements.filter(e => ids.includes(e.id) && visibleElement(e));
  if (!items.length) return { x: 0, y: 0, width: 0, height: 0 };
  const x = Math.min(...items.map(e => e.x)), y = Math.min(...items.map(e => e.y));
  return { x, y, width: Math.max(...items.map(e => e.x + e.width)) - x,
    height: Math.max(...items.map(e => e.y + e.height)) - y };
}

export function transformSelection(page: LayoutJSON, ids: string[], patch: Partial<ReturnType<typeof selectionBounds>>): LayoutJSON {
  if (Object.values(patch).some(n => !Number.isFinite(n))) return page;
  const before = selectionBounds(page, ids);
  if (!before.width || !before.height) return page;
  const x = Math.max(0, Math.min(page.slide.width - before.width, patch.x ?? before.x));
  const y = Math.max(0, Math.min(page.slide.height - before.height, patch.y ?? before.y));
  const width = Math.max(1, Math.min(page.slide.width - x, patch.width ?? before.width));
  const height = Math.max(1, Math.min(page.slide.height - y, patch.height ?? before.height));
  const sx = width / before.width, sy = height / before.height;
  return { ...page, elements: page.elements.map(e => {
    if (!ids.includes(e.id) || !visibleElement(e)) return e;
    const next = { ...e, x: x + (e.x - before.x) * sx, y: y + (e.y - before.y) * sy,
      width: e.width * sx, height: e.height * sy };
    // Keep AST bbox in sync, not OCR provenance masks (which remain source-space).
    if (e.bbox) next.bbox = { left: next.x, top: next.y, width: next.width, height: next.height };
    if (ids.length > 1 && e.type === 'text' && (sx !== 1 || sy !== 1)) {
      const scale = Math.min(sx, sy);
      next.style = { ...e.style, fontSize: (e.style?.fontSize || 24) * scale,
        letterSpacing: (e.style?.letterSpacing || 0) * scale };
    }
    return next;
  }) };
}
