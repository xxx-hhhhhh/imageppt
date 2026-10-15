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
    native = [e for e in elements if e.get('owner')=='editable_text']
    approximate = []
    tight = []
    for element in native:
        fit = element.get('metadata',{}).get('fontMeasurement',{})
        if fit and (fit.get('fontMaskSimilarity',1)<.55 or fit.get('fontAlphaError',0)>.22):
            approximate.append(element['id'])
        style = element.get('style',{})
        if style.get('letterSpacing',0)<-.055*max(1,style.get('fontSize',1)):
            tight.append(element['id'])
    issues.extend({'elementId':i,'problem':'approximateFontNeedsVisualReview'} for i in approximate)
    issues.extend({'elementId':i,'problem':'excessiveNegativeTracking'} for i in tight)
    visuals = [e for e in elements if e.get('owner')=='movable_image']
    uncertain_mattes = [e['id'] for e in visuals if e.get('metadata',{}).get('matteNeedsReview')]
    unresolved_surfaces = [e['id'] for e in visuals if e.get('metadata',{}).get('unresolvedChildSurfacePixels',0)>0]
    issues.extend({'elementId':i,'problem':'uncertainObjectMatte'} for i in uncertain_mattes)
    issues.extend({'elementId':i,'problem':'incompleteOccludedSurface'} for i in unresolved_surfaces)
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
        'fontApproximationWarningCount':len(approximate),'aggressiveTrackingCount':len(tight),
        'recoveredVisualPixelCount':sum(e.get('metadata',{}).get('recoveredVisualPixelCount',0) for e in elements),
        'contourPlanning':graph.get('contourPlanning',{}),
        'transparentContourAssetCount':sum(e.get('metadata',{}).get('visualAssetPolicy')=='supersampled_colour_decontaminated_matte' for e in visuals),
        'uncertainMatteCount':len(uncertain_mattes),'incompleteSurfaceCount':len(unresolved_surfaces),
        'completedChildSurfacePixels':sum(e.get('metadata',{}).get('completedChildSurfacePixels',0) for e in visuals),
        'issuesAfter':issues,
        'metricScope':'Character coverage from authoritative OCR; missing assets and old-ink ghosts measured. Pixel ownership is NOT semantic extraction completeness. Unknown metrics are null; PowerPoint rendering needs separate review.',
    }
