from pathlib import Path

import app.main as api
import pytest
from app.models.project_store import ProjectStore
from app.services.pptx.master_export import (
    MASTER_FILENAME,
    MasterExportError,
    _asset,
    export_master,
)
from fastapi.testclient import TestClient


def test_master_export_keeps_approval_and_separate_download(monkeypatch, tmp_path):
    store = ProjectStore(tmp_path)
    monkeypatch.setattr(api, 'store', store)
    monkeypatch.setattr(api, 'OUTPUTS_DIR', tmp_path)
    project = store.create('export example')['id']
    store.add_image(project, {'id':'page'})
    store.save_slide(project, 1, {'slide':{'width':1280,'height':720},'elements':[]})
    directory = tmp_path/project
    (directory/'editable.pptx').write_bytes(b'standard')

    def master_export(path, layouts):
        assert path == directory and len(layouts) == 1
        output = path/MASTER_FILENAME
        output.write_bytes(b'alternate')
        return output, {'valid':True}

    monkeypatch.setattr(api, 'export_master', master_export)
    client = TestClient(api.app)
    route = f'/api/projects/{project}/export/pptx'
    assert client.post(route, params={'engine':'ppt-master'}).status_code == 409
    assert client.post(route, params={'engine':'other'}).status_code == 422
    store.approve_page(project,1)
    response = client.post(route, params={'engine':'ppt-master'})
    assert response.status_code == 200
    assert client.get(response.json()['downloadUrl']).content == b'alternate'
    assert client.get(route).content == b'standard'


@pytest.mark.parametrize('src', ['/media/assets/def/x.png', '/media/assets/abc/..', 'file:///example.png'])
def test_master_export_only_resolves_current_project_media(tmp_path, src):
    with pytest.raises(MasterExportError):
        _asset(tmp_path/'abc', src)


def test_missing_skill_leaves_existing_export_untouched(monkeypatch, tmp_path):
    monkeypatch.setenv('PPT_MASTER_SKILL_DIR', str(tmp_path/'absent'))
    standard = Path(tmp_path/'editable.pptx')
    standard.write_bytes(b'prior export')
    with pytest.raises(MasterExportError, match='未找到 PPT Master'):
        export_master(tmp_path, [])
    assert standard.read_bytes() == b'prior export'
