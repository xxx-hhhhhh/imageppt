"""Require repeated short labels to agree on a measured, viable font family."""
from collections import Counter

from app.services.reconstruction.mixed_text import _run_error, restore_text_candidates


def prepare_text_lines(source, items, semantic):
    prepared = {}
    for item in items:
        evidence = {}
        candidates = restore_text_candidates(source, item, semantic.get(item['id']), evidence)
        if len(candidates)>1 and item.get('style',{}).get('fontClass') in {'sans','bold-sans'}:
            families = Counter(e[0]['style']['fontFamily'] for e in candidates)
            alternatives = []
            if len(families)>1:
                for family,_ in families.most_common(2):
                    review = {}
                    uniform = restore_text_candidates(source,item,
                        {**(semantic.get(item['id']) or {}),'_cohortFontFamily':family},review)
                    if uniform and ''.join(e[0]['text'] for e in uniform)==item['text'] and _run_error(uniform)<=_run_error(candidates)+.05:
                        alternatives.append((uniform,review))
                if alternatives:
                    candidates,evidence = min(alternatives,key=lambda pair:_run_error(pair[0]))
                    for element,_,_ in candidates:
                        element['metadata']['lineFontConsistency']='individually_verified_shared_family'
        prepared[item['id']] = (candidates, evidence)
    # Only neighbouring, short, similarly sized labels on the same baseline.
    # Do not homogenize headings/body, serif paragraphs or genuine mixed runs.
    labels = [i for i in items if 2 <= len(str(i.get('text') or '')) <= 12
              and len(prepared[i['id']][0]) == 1
              and i.get('style', {}).get('fontClass') in {'sans', 'bold-sans'}]
    used = set()
    for item in sorted(labels, key=lambda i: (i['y'], i['x'])):
        if item['id'] in used:
            continue
        row = sorted([i for i in labels if i['id'] not in used
                      and abs(i['y']+i['height']/2-item['y']-item['height']/2) < item['height']*.35
                      and .70 < i['height']/item['height'] < 1.43
                      and i.get('style', {}).get('fontClass') == item.get('style', {}).get('fontClass')],
                     key=lambda i: i['x'])
        groups = [[]]
        for other in row:
            if groups[-1] and other['x']-(groups[-1][-1]['x']+groups[-1][-1]['width']) > max(item['height']*4, min(other['width'],groups[-1][-1]['width'])*1.1):
                groups.append([])
            groups[-1].append(other)
        for group in groups:
            if len(group) < 3:
                continue
            used.update(i['id'] for i in group)
            families = Counter(prepared[i['id']][0][0][0]['style']['fontFamily'] for i in group)
            family, votes = families.most_common(1)[0]
            if votes < 2:
                continue
            updates = {}
            for other in group:
                old, _ = prepared[other['id']]
                if old[0][0]['style']['fontFamily'] == family:
                    continue
                evidence = {}
                hints = {**(semantic.get(other['id']) or {}), '_cohortFontFamily': family}
                candidate = restore_text_candidates(source, other, hints, evidence)
                if len(candidate) != 1 or _run_error(candidate) > _run_error(old)+.08:
                    updates = None
                    break
                updates[other['id']] = (candidate, evidence)
            if updates is None:
                continue
            prepared.update(updates)
            for other in group:
                for element, _, _ in prepared[other['id']][0]:
                    element['metadata']['fontCohort'] = {
                        'members': [i['id'] for i in group], 'family': family,
                        'policy': 'neighbour_consensus_with_individual_glyph_validation'}
    return prepared
