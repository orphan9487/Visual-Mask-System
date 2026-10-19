import unittest
from unittest.mock import patch
from src.services.mask_bases import choose,owned

class BaseTests(unittest.TestCase):
    def test_owner(self):
        with patch('src.services.mask_bases.read_job',return_value={'user_id':8}):
            with self.assertRaises(PermissionError):owned('a'*32,4)
    def test_choose(self):
        job={'base_status':'ready','base_candidates':['candidate-a','candidate-b']}
        with patch('src.services.mask_bases.owned',return_value=job),patch('pathlib.Path.is_file',return_value=True),patch('src.services.mask_bases.save') as save:
            self.assertEqual(choose('a'*32,4,1),{'selected':1})
            self.assertEqual(job['selected_base'],'candidate-b');save.assert_called_once()
    def test_invalid_candidate(self):
        with patch('src.services.mask_bases.owned',return_value={'base_status':'ready','base_candidates':[]}):
            with self.assertRaises(ValueError):choose('a'*32,4,0)

if __name__=='__main__':unittest.main()
