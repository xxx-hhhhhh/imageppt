"""Real Chromium workflow with explicit mocked long-running jobs; no paid calls."""
import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'temp' / 'task-browser-verification'
    output.mkdir(parents=True, exist_ok=True)
    project_id = 'f' * 32
    fixture = {'project': {'id': project_id, 'name': '任务恢复测试', 'createdAt': '2026-10-14', 'imageCount': 2},
               'slides': [{'version': '1.1', 'slide': {'width': 400, 'height': 220}, 'elements': []}],
               'images': [{'id': str(i), 'source_url': '/api/projects/c8a83bb098684dc284ba1f2e7b597f86/artifacts/source.png'} for i in range(2)],
               'approvedPages': [1], 'task': None}
    task = {'id': 'a'*32, 'projectId': project_id, 'page': 2, 'mode': 'maximum', 'status': 'running',
            'stage': 'vision', 'stageLabel': 'AI 页面结构理解', 'elapsedSeconds': 42,
            'deadlineSeconds': 600, 'message': '任务进行中'}
    counts = {'submit': 0, 'cancel': 0, 'create': 0}
    state = {'disconnect': False}
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={'width': 1600, 'height': 1000})
        def route(request):
            url = request.request.url
            method = request.request.method
            if method == 'POST' and url.endswith('/analysis-tasks'):
                counts['submit'] += 1
                task.update(status='running', message='任务进行中')
                fixture['task'] = task.copy()
                request.fulfill(status=202, json=task)
            elif url.endswith('/cancel'):
                counts['cancel'] += 1
                task.update(status='cancelled', message='已取消分析，原图及已完成页面未改动。')
                fixture['task'] = task.copy()
                request.fulfill(json=task)
            elif '/analysis-tasks/' in url:
                if state['disconnect']:
                    state['disconnect'] = False
                    request.abort('connectionfailed')
                else:
                    request.fulfill(json=task)
            elif url.endswith('/'+project_id):
                request.fulfill(json=fixture)
            else:
                request.continue_()
        context.route('**/api/projects/'+project_id+'**', route)
        def create_guard(route):
            if route.request.method == 'POST':
                counts['create'] += 1
                route.fulfill(status=500, json={'detail': 'Unexpected duplicate project'})
            else:
                route.continue_()
        context.route('**/api/projects', create_guard)
        page = context.new_page()
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto('http://127.0.0.1:5173/?project='+project_id)
        analyze = page.get_by_role('button', name='AI解析 / 开始转换')
        expect(analyze).to_be_enabled()
        page.evaluate('document.fonts.ready')
        page.wait_for_function("() => [...document.images].filter(i => i.offsetParent !== null).every(i => i.complete && i.naturalWidth > 0)")
        page.screenshot(path=str(output/'before.png'), full_page=True)
        analyze.click()
        expect(page.get_by_role('button', name='取消分析')).to_be_visible()
        expect(page.get_by_label('分析任务进度')).to_contain_text('AI 页面结构理解')
        expect(page.get_by_label('分析任务进度')).to_contain_text('600')
        page.screenshot(path=str(output/'running.png'), full_page=True)
        page.reload()
        expect(page.get_by_role('button', name='取消分析')).to_be_visible()
        expect(page.get_by_label('分析任务进度')).to_contain_text('第 2 页')
        assert counts['submit'] == 1 and counts['create'] == 0
        second = context.new_page()
        second.goto('http://127.0.0.1:5173/')
        expect(second.get_by_role('button', name='取消分析')).to_be_visible()
        assert counts['submit'] == 1 and counts['create'] == 0
        second.close()
        state['disconnect'] = True
        expect(page.get_by_role('button', name='恢复连接')).to_be_visible(timeout=20000)
        page.get_by_role('button', name='恢复连接').click()
        expect(page.get_by_role('button', name='恢复连接')).to_have_count(0)
        page.get_by_role('button', name='取消分析').click()
        expect(page.get_by_role('button', name='取消分析')).to_have_count(0)
        expect(analyze).to_be_enabled()
        assert counts['cancel'] == 1
        analyze.click()
        expect(page.get_by_role('button', name='取消分析')).to_be_visible()
        task.update(status='timed_out', message='本页超过总时限，已停止分析；旧结果保留。')
        fixture['task'] = task.copy()
        expect(page.get_by_role('button', name='重试 AI')).to_be_visible(timeout=15000)
        page.screenshot(path=str(output/'timeout.png'), full_page=True)
        page.get_by_role('button', name='重试 AI').click()
        expect(page.get_by_role('button', name='取消分析')).to_be_visible()
        task.update(status='succeeded', message='本页重建完成，请对照原图检查。')
        fixture['task'] = task.copy()
        fixture['slides'].append(fixture['slides'][0].copy())
        expect(page.get_by_role('heading', name='第 2 页 · 逐页确认')).to_be_visible(timeout=15000)
        expect(page.get_by_role('button', name='通过本页')).to_be_visible()
        page.set_viewport_size({'width': 1100, 'height': 800})
        page.screenshot(path=str(output/'success-narrow.png'), full_page=True)
        assert counts == {'submit': 3, 'cancel': 1, 'create': 0}
        assert not errors, errors
        browser.close()
    report = {'mockedProviderJobs': True, 'refreshRecovery': True, 'newTabRecovery': True,
              'cancel': True, 'reconnect': True, 'timeoutRetry': True, 'secondPageSuccess': True,
              'requests': counts, 'pageErrors': errors}
    (output/'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
