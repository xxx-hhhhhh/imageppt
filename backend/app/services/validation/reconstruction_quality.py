"""Measured editability is separate from raster similarity and pixel ownership."""
from __future__ import annotations


def reconstruction_quality(layout: dict, raster_audit: dict, preservation: dict) -> dict:
    graph = layout.get('metadata',{}).get('ownershipAudit',{})
    decisions = graph.get('textDecisions',[])
    elements = layout.get('elements',[])
    chars = sum(d.get('sourceCharacterCount',len(d.get('text') or '')) for d in decisions)
    restored = sum(d.get('editableCharacterCount',0) for d in decisions)
    complete = sum(d.get('sourceCharacterCount',1)>0 and d.get('editableCharacterCount',0)==d.get('sourceCharacterCount') for d in decisions)
    partial = sum(0<d.get('editableCharacterCount',0)<d.get('sourceCharacterCount',0) for d in decisions)
    nodes = graph.get('nodes',[])
    area = max(1,layout['slide']['width']*layout['slide']['height'])
    image_nodes = [n for n in nodes if n.get('owner')=='movable_image']
    largest = max((n.get('sourcePixelCount',0) for n in image_nodes),default=0)/area
    unresolved = [{'id':d['id'],'text':d.get('text'),'reason':d.get('reason')}
                  for d in decisions if d.get('editableCharacterCount',0)!=d.get('sourceCharacterCount',1)]
    issues = [{'elementId':d['id'],'problem':'textRetainedAsImage','reason':d['reason']} for d in unresolved]
    if largest>.3:
        issues.append({'problem':'largeRasterCarrier','sourceAreaRatio':round(largest,4)})
    if partial:
        issues.append({'problem':'fragmentedEditableLines','count':partial})
    return {
        'qualityStatus':'partial_editable' if unresolved or largest>.3 else 'requires_visual_review',
        'detectedTextCount':len(decisions),'editableTextCount':complete,
        'nonEditableTextCount':len(decisions)-complete,'nativeTextBoxCount':sum(e.get('owner')=='editable_text' for e in elements),
        'editableTextCoverage':complete/max(1,len(decisions)),
        'editableCharacterCoverage':restored/max(1,chars),'fragmentedEditableLineCount':partial,
        'unresolvedText':unresolved,'largestRasterOwnerAreaRatio':round(largest,4),
        'objectExtractionCoverage':None,'movableVisualCoverage':None,'semanticObjectCoverage':None,
        'visualAreaPreserved':max(0,1-(preservation.get('unownedSourcePixels',0)+preservation.get('unexpectedVisualChangedPixels',0))/area),
        'missingVisualCount':None,'backgroundResidualCount':None,
        'missingAssetCount':raster_audit.get('missingAssetCount'),
        'ghostingCount':raster_audit.get('rasterNativeDuplicateTextCount'),
        'duplicateCount':None,'rasterNativeDuplicateTextCount':raster_audit.get('rasterNativeDuplicateTextCount'),
        'imageAssetCount':len(image_nodes),'shapeCount':sum(e.get('owner')=='native_shape' for e in elements),
        'issuesAfter':issues,
        'metricScope':'Character coverage from authoritative OCR; missing assets and old-ink ghosts measured. Pixel ownership is NOT semantic extraction completeness. Unknown metrics are null; PowerPoint rendering needs separate review.',
    }
