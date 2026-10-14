"""Opt-in live local API smoke: cancel, then sequential real-image pages.

Uses a new diagnostic record; never rewrites a user's existing project.
Page 2 uses configured Qwen. Run only with the user's paid-API permission.
"""
import argparse
import json
import time
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('first', type=Path)
    parser.add_argument('second', type=Path)
    parser.add_argument('--resume-project', help='Retry page 2 of an existing diagnostic record only')
    args = parser.parse_args()
    with httpx.Client(base_url='http://127.0.0.1:8000', timeout=20) as client:
        if args.resume_project:
            project = args.resume_project
        else:
            response = client.post('/api/projects', json={'name': '真实两页任务回归'})
            response.raise_for_status()
            project = response.json()['id']
        base = '/api/projects/'+project
        print(json.dumps({'projectId': project}), flush=True)
        if not args.resume_project:
            response = client.post(base+'/images', files=[('files', (p.name, p.read_bytes(), 'image/png')) for p in (args.first, args.second)])
            response.raise_for_status()
            task = client.post(base+'/analysis-tasks', json={'page': 1, 'mode': 'fast'}).json()
            cancel = client.post(base+'/analysis-tasks/'+task['id']+'/cancel')
            cancel.raise_for_status()
            assert cancel.json()['status'] == 'cancelled'
            assert client.get(base).json()['slides'] == []
        results = []
        for page, mode in ([(2, 'maximum')] if args.resume_project else [(1, 'fast'), (2, 'maximum')]):
            response = client.post(base+'/analysis-tasks', json={'page': page, 'mode': mode})
            response.raise_for_status()
            task = response.json()
            duplicate = client.post(base+'/analysis-tasks', json={'page': page, 'mode': mode})
            assert duplicate.json()['id'] == task['id']
            assert client.get(base).json()['task']['id'] == task['id']
            stage = None
            until = time.monotonic()+630
            while task['status'] in {'running', 'publishing'} and time.monotonic()<until:
                if task['stage'] != stage:
                    print(json.dumps({'page': page, 'stage': task['stage'], 'elapsed': task['elapsedSeconds']}), flush=True)
                    stage = task['stage']
                time.sleep(2)
                response = client.get(base+'/analysis-tasks/'+task['id'])
                response.raise_for_status()
                task = response.json()
            print(json.dumps(task, ensure_ascii=False), flush=True)
            results.append(task)
            if task['status'] != 'succeeded':
                break
            client.post(base+f'/pages/{page}/approve').raise_for_status()
        result = {'projectId': project, 'cancelledBeforePublish': not args.resume_project, 'pages': results}
        if all(t['status']=='succeeded' for t in results):
            response = client.post(base+'/export/pptx')
            response.raise_for_status()
            result['export'] = response.json()
        output = Path(__file__).resolve().parents[1]/'outputs'/project/('live_task_verification_retry.json' if args.resume_project else 'live_task_verification.json')
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
