import unittest
from unittest.mock import MagicMock,patch
from src.services.lora_training import submit

class ReuseTests(unittest.TestCase):
    def test_same_photos_create_independent_named_jobs(self):
        photos=[dict(id=i,status='ready',path='photo'+str(i),mask_name='old name') for i in (1,2,3)]
        jobs_dir=MagicMock();jobs_dir.glob.return_value=[]
        with patch('db.training_repo.list_training_photos',return_value=photos),patch('src.services.lora_training.JOBS',jobs_dir),patch('src.services.lora_training.save'),patch('src.services.lora_training.subprocess.Popen'):
            a=submit(4,'TEST','第一組',[1,2,3]);b=submit(4,'TEST','第二組',[1,2,3])
        self.assertNotEqual(a['id'],b['id'])
        self.assertEqual(a['paths'],b['paths'])
        self.assertNotEqual(a['mask_name'],b['mask_name'])
        self.assertTrue(all(p['status']=='ready' for p in photos))

if __name__=='__main__':unittest.main()
