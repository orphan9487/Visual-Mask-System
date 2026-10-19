import sys
import types
import unittest
from unittest.mock import MagicMock,patch
from pathlib import Path
import numpy as np
import cv2
from src.services.face_profile_service import _validate_single_face


class FaceValidationRetryTests(unittest.TestCase):
    def run_validation(self,detections):
        analyzer=MagicMock();analyzer.get.side_effect=detections
        module=types.ModuleType('insightface.app');module.FaceAnalysis=MagicMock(return_value=analyzer)
        with patch.dict(sys.modules,{'insightface.app':module}),patch('numpy.fromfile',return_value=np.array([],dtype=np.uint8)),patch('cv2.imdecode',return_value=np.zeros((1024,1024,3),dtype=np.uint8)):
            _validate_single_face(Path('test.png'))
        return analyzer

    def test_zero_then_one_retries_with_padding(self):
        analyzer=self.run_validation([[],[object()]])
        self.assertEqual(analyzer.get.call_count,2)
        self.assertEqual(analyzer.get.call_args.args[0].shape,(1536,1536,3))
    def test_one_face_does_not_retry(self):
        self.assertEqual(self.run_validation([[object()]]).get.call_count,1)
    def test_multiple_faces_are_not_retried_or_silently_selected(self):
        with self.assertRaisesRegex(ValueError,'2'):self.run_validation([[object(),object()]])
    def test_no_face_has_distinct_message(self):
        with self.assertRaisesRegex(ValueError,'未偵測到'):self.run_validation([[],[]])
    def test_multiple_faces_after_padding_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'2'):self.run_validation([[],[object(),object()]])


if __name__=='__main__':unittest.main()
