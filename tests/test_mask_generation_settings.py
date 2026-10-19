import unittest
from unittest.mock import patch
from src.services.mask_generation_settings import update,settings

class SettingsTests(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(settings({})['lora_weight'],.7)
        self.assertEqual(settings({})['seed'],42)

    def test_owner(self):
        with patch('src.services.mask_generation_settings.read_job',return_value={'user_id':4}):
            with self.assertRaises(PermissionError):update('a'*32,5,{})

    def test_invalid_source(self):
        job=dict(user_id=4,status='ready',paths=['a','b','c'])
        with patch('src.services.mask_generation_settings.read_job',return_value=job):
            with self.assertRaises(ValueError):update('a'*32,4,{'source_index':3})

    def test_save_settings(self):
        job=dict(user_id=4,status='ready',paths=['a','b','c'])
        with patch('src.services.mask_generation_settings.read_job',return_value=job),patch('src.services.mask_generation_settings.checkpoint_path',return_value='weight'),patch('pathlib.Path.is_file',return_value=True),patch('src.services.mask_generation_settings.save') as save:
            result=update('a'*32,4,dict(source_index=1,checkpoint=200,lora_weight=.5,strength=.3,seed=42))
            self.assertEqual(result['source_index'],1);save.assert_called_once()

if __name__=='__main__':unittest.main()
