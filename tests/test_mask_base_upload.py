import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from src.services import mask_base_upload as service


class UploadTests(unittest.TestCase):
    def image(self):
        stream=io.BytesIO();Image.new('RGB',(400,600)).save(stream,format='PNG');return stream.getvalue()

    def test_preview_does_not_replace_selected_base(self):
        job={'id':'a'*32,'status':'ready'}
        with tempfile.TemporaryDirectory() as directory, patch.object(service,'ROOT',Path(directory)),patch.object(service,'owned',return_value=job),patch.object(service,'save') as save,patch('src.services.face_profile_service._validate_single_face'),patch('src.integrations.liveportrait_expression.generate_from_expression_profile',side_effect=lambda instruction:instruction['identity']['source_override']):
            result=service.upload_base('a'*32,4,self.image(),'image/png',mode='natural')
            with Image.open(job['pending_base']['source']) as image:
                self.assertEqual(image.size,(400,600))
            self.assertNotIn('selected_base',job);self.assertFalse(result['ready']);save.assert_called_once()
            with self.assertRaises(LookupError):service.confirm_base('a'*32,4,'wrong')
            with self.assertRaises(ValueError):service.confirm_base('a'*32,4,result['token'])
            service.confirm_base('a'*32,4,result['token'],True)
            self.assertEqual(job['base_source'],'user_confirmed_neutral');self.assertNotIn('pending_base',job)

    def test_rejected_face_does_not_change_base(self):
        job={'selected_base':'old.png'}
        with tempfile.TemporaryDirectory() as directory,patch.object(service,'ROOT',Path(directory)),patch.object(service,'owned',return_value=job),patch.object(service,'save') as save,patch('src.services.face_profile_service._validate_single_face',side_effect=ValueError('no face')):
            with self.assertRaises(ValueError):service.upload_base('a'*32,4,self.image(),'image/png')
            self.assertEqual(job['selected_base'],'old.png');save.assert_not_called()
            self.assertFalse(list(Path(directory).rglob('*.png')))

    def test_permission_check_precedes_save(self):
        with patch.object(service,'owned',side_effect=PermissionError()):
            with self.assertRaises(PermissionError):service.upload_base('a'*32,9,self.image(),'image/png')

    def test_auto_reuses_reference_without_rendering(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'models/user_loras'/('a'*32);folder.mkdir(parents=True)
            reference=folder/'reference.png';Image.new('RGB',(400,600)).save(reference)
            job={'selected_base':str(reference),'base_source':'user_confirmed_neutral'}
            with patch.object(service,'ROOT',root),patch.object(service,'owned',return_value=job),patch.object(service,'save'),patch.object(service,'identity_similarity',return_value=.7),patch('src.services.face_profile_service._validate_single_face'),patch('src.integrations.liveportrait_expression.generate_from_expression_profile') as render:
                result=service.upload_base('a'*32,4,self.image(),'image/png')
                self.assertEqual(result['method'],'reuse_confirmed_reference');render.assert_not_called()
                self.assertEqual(job['selected_base'],str(reference))

    def test_first_upload_is_only_an_attempt(self):
        job={'status':'ready'}
        with tempfile.TemporaryDirectory() as directory,patch.object(service,'ROOT',Path(directory)),patch.object(service,'owned',return_value=job),patch.object(service,'save'),patch.object(service,'identity_similarity',return_value=.7),patch('src.services.face_profile_service._validate_single_face'),patch('src.integrations.liveportrait_expression.generate_from_expression_profile',side_effect=lambda i:i['identity']['source_override']):
            result=service.upload_base('a'*32,4,self.image(),'image/png')
            self.assertEqual(result['method'],'automatic_neutral_attempt');self.assertNotIn('selected_base',job)

    def test_identity_mismatch_preserves_old_base(self):
        with tempfile.TemporaryDirectory() as directory:
            reference=Path(directory)/'old.png';Image.new('RGB',(400,600)).save(reference)
            job={'selected_base':str(reference),'base_source':'user_confirmed_neutral'}
            with patch.object(service,'ROOT',Path(directory)),patch.object(service,'owned',return_value=job),patch.object(service,'save') as save,patch.object(service,'identity_similarity',return_value=.1),patch('src.services.face_profile_service._validate_single_face'):
                with self.assertRaises(ValueError):service.upload_base('a'*32,4,self.image(),'image/png')
                self.assertEqual(job['selected_base'],str(reference));save.assert_not_called()


if __name__=='__main__':unittest.main()
