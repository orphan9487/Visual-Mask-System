import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from src.services import portrait_masks as masks


class PortraitMaskTests(unittest.TestCase):
    def setUp(self):
        (Path(__file__).resolve().parents[1]/'tmp').mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'tmp')
        self.root=Path(self.temp.name)
        self.patches=[patch.object(masks,'ROOT',self.root),patch.object(masks,'STORE',self.root/'data/portrait_masks'),patch('src.services.face_profile_service._validate_single_face')]
        for p in self.patches:p.start()
    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()
    def create(self,name='My mask',mode='natural'):
        stream=io.BytesIO();Image.new('RGB',(800,600),'red').save(stream,format='PNG')
        return masks.create(4,'TEST',name,stream.getvalue(),'image/png',json.dumps(dict(x=100,y=0,width=600,height=600)),mode)
    def test_square_validation(self):
        image=Image.new('RGB',(800,600))
        for crop in [dict(x=0,y=0,width=500,height=400),dict(x=1600,y=0,width=600,height=600),dict(x=0,y=0,width=200,height=200)]:
            with self.assertRaises(ValueError):masks.square_crop(image,json.dumps(crop))
        self.assertEqual(masks.square_crop(image,json.dumps(dict(x=0,y=0,width=600,height=600))).size,(1024,1024))
    def test_white_padding_preserves_photo_and_rejects_empty_crop(self):
        image=Image.new('RGB',(400,600),'red')
        output=masks.square_crop(image,json.dumps(dict(x=-100,y=0,width=600,height=600)))
        self.assertEqual(output.size,(1024,1024))
        self.assertEqual(output.getpixel((0,512)),(255,255,255))
        self.assertEqual(output.getpixel((512,512)),(255,0,0))
        self.assertEqual(output.getpixel((1023,512)),(255,255,255))
        with self.assertRaises(ValueError):masks.square_crop(image,json.dumps(dict(x=-400,y=-300,width=256,height=256)))
    def test_lifecycle(self):
        result=self.create();key=result['key']
        self.assertIsNone(masks.active_key(4))
        with self.assertRaises(ValueError):masks.set_active(key,4)
        with self.assertRaises(ValueError):masks.confirm(key,4,False)
        masks.confirm(key,4,True);self.assertEqual(masks.active_key(4),key)
        with Image.open(masks.preview(key,4)) as image:self.assertEqual(image.size,(1024,1024))
        masks.delete(key,4);self.assertIsNone(masks.active_key(4));self.assertEqual(masks.list_masks(4),[])
        self.assertTrue(list((self.root/'data/portrait_masks/4').glob('*/base.png')))
    def test_ownership(self):
        key=self.create()['key']
        for operation in [lambda:masks.preview(key,9),lambda:masks.delete(key,9),lambda:masks.set_active(key,9)]:
            with self.assertRaises(PermissionError):operation()
        with self.assertRaises(ValueError):masks.read('../../escape')
    def test_delete_active_chooses_remaining_ready(self):
        first=self.create('first')['key'];masks.confirm(first,4,True)
        second=self.create('second')['key'];masks.confirm(second,4,True)
        masks.delete(second,4);self.assertEqual(masks.active_key(4),first)
    def test_auto_uses_new_upload_not_active_reference(self):
        first=self.create()['key'];masks.confirm(first,4,True)
        with patch.object(masks,'identity_similarity',return_value=.7),patch('src.integrations.liveportrait_expression.generate_from_expression_profile',side_effect=lambda instruction:instruction['identity']['source_override']) as render:
            result=self.create(mode='auto')
            self.assertEqual(result['method'],'neutral_attempt');render.assert_called_once()
            record=masks.read(result['key'],4)
            self.assertEqual(render.call_args.args[0]['identity']['source_override'],record['source'])
            self.assertNotEqual(record['source'],masks.read(first,4)['base'])
            self.assertEqual(masks.active_key(4),first)
    def test_legacy_import_once_without_weights(self):
        source=self.root/'old.png';Image.new('RGB',(400,600)).save(source)
        folder=self.root/'data/lora_jobs';folder.mkdir(parents=True)
        (folder/'old.json').write_text(json.dumps(dict(id='old',user_id=4,status='ready',selected_base=str(source),base_source='user_confirmed_neutral',mask_name='old mask')),encoding='utf-8')
        masks.import_legacy(4,'TEST');listed=masks.list_masks(4);self.assertEqual(len(listed),1)
        masks.delete(listed[0]['key'],4);masks.import_legacy(4,'TEST');self.assertEqual(masks.list_masks(4),[])


if __name__=='__main__':unittest.main()
