import json
import pytest
from PIL import Image
from app.parsers import CVATParser,COCOParser,YOLOParser,KITTIParser,valid_geometry,task_type,detect_format

def write(tmp_path,name,text):
    p=tmp_path/name
    p.write_text(text,encoding='utf-8')
    return p

def test_cvat_mixed_and_metadata(tmp_path):
    p=write(tmp_path,'labels.xml','<annotations><image id="0" name="a.png" width="100" height="100"><box label="car" xtl="1" ytl="2" xbr="20" ybr="30" occluded="1"><attribute name="weather">rain</attribute></box><polygon label="road" points="0,0;50,0;0,50"/><points label="pose" points="10,20;30,40"/></image></annotations>')
    parser=CVATParser();s=parser.parse([p],[])[0]
    assert [a['shape_type'] for a in s['annotations']]==['BBOX_2D','POLYGON','KEYPOINT']
    assert s['annotations'][0]['attributes']=={'weather':'rain'}
    assert s['annotations'][0]['occluded'] is True
    assert 'visibility' not in s['annotations'][2]['geometry']['points'][0]
    assert task_type(s['annotations'])=='MIXED'
    assert detect_format([p])=='CVAT'

def test_invalid_bbox_and_polygon_skipped(tmp_path):
    p=write(tmp_path,'a.xml','<annotations><image name="a" width="10" height="10"><box label="x" xtl="5" ytl="0" xbr="1" ybr="2"/><polygon label="x" points="0,0;1,1;2,2"/></image></annotations>')
    parser=CVATParser();assert parser.parse([p],[])[0]['annotations']==[]
    assert len(parser.warnings)==2

@pytest.mark.parametrize('geometry',[dict(x1=1,y1=1,x2=1,y2=2),dict(x1=0,y1=0,x2=float('nan'),y2=2)])
def test_geometry_rejects_invalid(geometry):
    assert not valid_geometry('BBOX_2D',geometry)

def test_coco_tasks_and_rle_preservation(tmp_path):
    data={'images':[{'id':1,'file_name':'a.png','width':100,'height':100}], 'categories':[{'id':1,'name':'person','keypoints':['nose']}], 'annotations':[{'id':1,'image_id':1,'category_id':1,'bbox':[1,2,10,20]},{'id':2,'image_id':1,'category_id':1,'segmentation':[[0,0,10,0,0,10]]},{'id':3,'image_id':1,'category_id':1,'keypoints':[4,5,1]},{'id':4,'image_id':1,'category_id':1,'segmentation':{'size':[100,100],'counts':'abc'}}]}
    p=write(tmp_path,'a.json',json.dumps(data));s=COCOParser().parse([p],[])[0]
    assert len(s['annotations'])==4
    assert s['annotations'][0]['geometry']['x2']==11
    assert s['annotations'][2]['geometry']['points'][0]['visibility']==1
    assert s['annotations'][3]['geometry']['rle']['counts']=='abc'

def test_yolo_denormalization(tmp_path):
    image=tmp_path/'a.png';Image.new('RGB',(100,200)).save(image)
    p=write(tmp_path,'a.txt','0 0.5 0.5 0.2 0.4\nwrong line')
    classes=write(tmp_path,'classes.txt','vehicle')
    parser=YOLOParser();s=parser.parse([p,classes],[image])[0]
    assert s['annotations'][0]['geometry']==dict(x1=40,y1=60,x2=60,y2=140)
    assert s['annotations'][0]['label']=='vehicle'
    assert len(parser.warnings)==1

def test_yolo_missing_media_fails(tmp_path):
    with pytest.raises(ValueError,match='matching image'):
        YOLOParser().parse([write(tmp_path,'a.txt','0 0.5 0.5 0.2 0.2')],[])

def test_kitti_coordinates_and_unavailable_occlusion(tmp_path):
    p=write(tmp_path,'000001.txt','Car 0.2 3 0 0 0 10 10 1.5 1.8 4 1 2 30 0.5')
    a=KITTIParser().parse([p],[])[0]['annotations'][0]
    assert a['shape_type']=='CUBOID_3D'
    assert a['geometry']['dimensions']['length']==4
    assert a['occluded'] is None
    assert 'bottom center' in a['source_metadata']['coordinate_system']

def test_cvat_video_tracks_and_meta(tmp_path):
    xml_content = '''<annotations>
        <version>1.1</version>
        <meta>
            <task>
                <id>42</id>
                <name>Test Task</name>
                <size>2</size>
                <original_size><width>640</width><height>480</height></original_size>
            </task>
            <segments>
                <segment><id>99</id></segment>
            </segments>
        </meta>
        <track id="1" label="Vehicle">
            <box frame="0" outside="0" occluded="0" xtl="10" ytl="20" xbr="50" ybr="60" />
            <box frame="1" outside="1" occluded="0" xtl="10" ytl="20" xbr="50" ybr="60" />
        </track>
    </annotations>'''
    p = write(tmp_path, 'annotations.xml', xml_content)
    parser = CVATParser()
    samples = parser.parse([p], [])
    assert parser.cvat_task_id == 42
    assert parser.cvat_job_id == 99
    assert len(samples) == 2
    assert samples[0]['file_name'] == 'frame_000000.PNG'
    assert len(samples[0]['annotations']) == 1
    assert samples[0]['annotations'][0]['label'] == 'Vehicle'
    assert samples[0]['annotations'][0]['geometry'] == {'x1': 10.0, 'y1': 20.0, 'x2': 50.0, 'y2': 60.0}
    # Frame 1 has outside=1 so no active annotation
    assert len(samples[1]['annotations']) == 0

