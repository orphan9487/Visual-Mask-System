import unittest
from unittest.mock import MagicMock, patch
from src.services.mask_delete import delete_mask

class MaskDeleteTests(unittest.TestCase):
    def test_other_owner(self):
        with patch('src.services.mask_delete.read_job',return_value={'user_id':5}):
            with self.assertRaises(PermissionError):delete_mask('a'*32,4,'TEST')

    def test_training_guard(self):
        with patch('src.services.mask_delete.read_job',return_value={'user_id':4,'status':'training'}):
            with self.assertRaises(ValueError):delete_mask('a'*32,4,'TEST')

    def test_active_mask_falls_back_and_preserves_job(self):
        conn=MagicMock();cur=conn.__enter__.return_value.cursor.return_value.__enter__.return_value
        cur.fetchone.side_effect=[(1,),('faceid_user_4',)]
        job={'user_id':4,'status':'ready','lora_path':'keep.safetensors'}
        with patch('src.services.mask_delete.read_job',return_value=job),patch('src.services.mask_delete.get_connection',return_value=conn),patch('src.services.mask_delete.save') as save:
            result=delete_mask('a'*32,4,'TEST')
            self.assertEqual(result['active_mask'],'faceid_user_4')
            self.assertEqual(job['status'],'deleted');self.assertEqual(job['lora_path'],'keep.safetensors');save.assert_called_once()

if __name__=='__main__':unittest.main()
