import io
import json
import zipfile
import pytest
from PIL import Image

XML = '<annotations><image name="a.png" width="100" height="100"><box label="car" xtl="1" ytl="2" xbr="30" ybr="40"/></image><image name="b.png" width="100" height="100"><points label="pose" points="1,1;3,3"/></image></annotations>'

def upload(client, xml=XML):
    return client.post(
        '/api/datasets/upload',
        data={'name': 'Real import fixture'},
        files=[('annotations', ('labels.xml', xml, 'application/xml'))]
    )

def test_health_and_empty_database(client):
    assert client.get('/api/health').json()['status'] == 'ok'
    assert client.get('/api/datasets').json() == []
    assert client.get('/docs').status_code == 200

def test_upload_dataset_and_details(client):
    response = upload(client)
    assert response.status_code == 200, response.text
    d = response.json()
    dataset_id = d['id']
    assert d['sample_count'] == 2

    # Fetch samples
    samples = client.get(f'/api/datasets/{dataset_id}/samples').json()
    assert len(samples) == 2

    # Fetch sample detail
    detail = client.get(f"/api/samples/{samples[0]['id']}").json()
    assert len(detail['annotations']) == 1
    assert detail['annotations'][0]['shape_type'] == 'BBOX_2D'

    # Delete dataset
    assert client.delete(f'/api/datasets/{dataset_id}').status_code == 200
    assert client.get('/api/datasets').json() == []

def test_image_upload_matching(client):
    image = io.BytesIO()
    Image.new('RGB', (100, 100), (120, 120, 120)).save(image, format='PNG')
    response = client.post(
        '/api/datasets/upload',
        data={'name': 'image test'},
        files=[
            ('annotations', ('a.xml', XML)),
            ('media', ('a.png', image.getvalue(), 'image/png'))
        ]
    )
    assert response.status_code == 200, response.text
    rows = client.get(f"/api/datasets/{response.json()['id']}/samples").json()
    assert rows[0]['media_url'] is not None
    assert client.get(rows[0]['media_url']).status_code == 200

def test_malformed_and_unsupported_upload(client):
    assert upload(client, '<broken').status_code == 422
    assert upload(client, '<annotations/>').status_code == 422
    assert client.post(
        '/api/datasets/upload',
        data={'name': 'x', 'format': 'BAD'},
        files={'annotations': ('a.txt', 'text')}
    ).status_code == 422
    assert client.get('/api/datasets').json() == []

def test_browser_empty_optional_media_field(client):
    response = client.post(
        '/api/datasets/upload',
        data={'name': 'no media'},
        files=[('annotations', ('labels.xml', XML)), ('media', ('', b''))]
    )
    assert response.status_code == 200, response.text

def test_zip_traversal_rejected(client):
    data = io.BytesIO()
    with zipfile.ZipFile(data, 'w') as archive:
        archive.writestr('../escape.xml', XML)
    response = client.post(
        '/api/datasets/upload',
        data={'name': 'unsafe'},
        files={'annotations': ('a.zip', data.getvalue())}
    )
    assert response.status_code == 422 and 'Unsafe ZIP' in response.text

def test_zip_success(client):
    data = io.BytesIO()
    with zipfile.ZipFile(data, 'w') as archive:
        archive.writestr('nested/labels.xml', XML)
    response = client.post(
        '/api/datasets/upload',
        data={'name': 'archive'},
        files={'annotations': ('a.zip', data.getvalue())}
    )
    assert response.status_code == 200, response.text
    dataset_id = response.json()['id']
    assert client.delete(f'/api/datasets/{dataset_id}').status_code == 200

@pytest.mark.parametrize('format,filename,content,expected', [
    (
        'COCO',
        'labels.json',
        json.dumps({
            'images': [{'id': 1, 'file_name': 'a.png', 'width': 100, 'height': 100}],
            'categories': [{'id': 1, 'name': 'car'}],
            'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [1, 1, 20, 20]}]
        }),
        'BBOX_2D'
    ),
    ('YOLO', 'a.txt', '0 0.5 0.5 0.2 0.2', 'BBOX_2D'),
    ('KITTI', 'a.txt', 'Car 0.1 1 0 0 0 10 10 1.5 1.8 4 1 2 30 0.5', 'CUBOID_3D'),
])
def test_other_formats_through_upload(client, format, filename, content, expected):
    image = io.BytesIO()
    Image.new('RGB', (100, 100), (120, 120, 120)).save(image, format='PNG')
    response = client.post(
        '/api/datasets/upload',
        data={'name': format, 'format': format},
        files=[
            ('annotations', (filename, content)),
            ('media', ('a.png', image.getvalue(), 'image/png'))
        ]
    )
    assert response.status_code == 200, response.text
    d = response.json()
    assert d['task_type'] == expected and d['annotation_count'] == 1
