import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from fastapi.testclient import TestClient
from src.interfaces.multipage_app import create_app,current
from src.services import portrait_masks


class MaskAPITests(unittest.TestCase):
    def test_create_confirm_select_delete_and_legacy_redirect(self):
        (Path(__file__).resolve().parents[1]/'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'tmp') as directory,patch.object(portrait_masks,'ROOT',Path(directory)),patch.object(portrait_masks,'STORE',Path(directory)/'masks'),patch('src.services.face_profile_service._validate_single_face'):
            app=create_app();app.dependency_overrides[current]=lambda:{'id':4,'username':'TEST'}
            with TestClient(app) as client:
                self.assertEqual(client.get('/masks').status_code,200)
                self.assertEqual(client.get('/lora-training',follow_redirects=False).headers['location'],'/masks')
                self.assertEqual(client.post('/api/lora-training/jobs',json={}).status_code,404)
                stream=io.BytesIO();Image.new('RGB',(600,800)).save(stream,format='PNG')
                data={'name':'API mask','mode':'natural','crop_json':json.dumps(dict(x=0,y=100,width=600,height=600))}
                response=client.post('/api/masks',data=data,files={'image':('image.png',stream.getvalue(),'image/png')})
                self.assertEqual(response.status_code,200,response.text);key=response.json()['key']
                self.assertEqual(client.get('/api/masks/'+key+'/preview').status_code,200)
                self.assertEqual(client.post('/api/masks/active',json={'mask_id':key}).status_code,400)
                self.assertEqual(client.post('/api/masks/'+key+'/confirm',json={'accepted':True}).status_code,200)
                self.assertEqual(client.get('/api/masks').json()['masks'][0]['is_active'],True)
                app.dependency_overrides[current]=lambda:{'id':9,'username':'OTHER'}
                self.assertEqual(client.delete('/api/masks/'+key).status_code,403)
                app.dependency_overrides[current]=lambda:{'id':4,'username':'TEST'}
                self.assertEqual(client.delete('/api/masks/'+key).status_code,200)
                self.assertEqual(client.get('/api/masks').json()['masks'],[])


if __name__=='__main__':unittest.main()
