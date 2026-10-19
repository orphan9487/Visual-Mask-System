import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from src.services.pipeline_service import PipelineService,PipelineAnalysis


class PortraitPipelineTests(unittest.TestCase):
    def test_neutral_copies_base_and_sadness_only_calls_liveportrait(self):
        (Path(__file__).resolve().parents[1]/'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'tmp') as directory:
            root=Path(directory);base=root/'base.png';Image.new('RGB',(512,512)).save(base)
            generated=root/'sadness.png';Image.new('RGB',(512,512),'blue').save(generated)
            service=PipelineService(output_root=root)
            key='portrait_4_'+'a'*32
            mask={'status':'ready','username':'TEST','base':str(base)}
            def analysis(emotion):return PipelineAnalysis({'emotion':emotion,'instruction':{'emotion':emotion,'identity':{'mask_id':key}}},'test',[],'TEST')
            with patch('src.services.portrait_masks.read',return_value=mask),patch('src.integrations.liveportrait_expression.generate_from_expression_profile',return_value=str(generated)) as live,patch('src.integrations.liveportrait_expression.stylize_with_sd15',side_effect=AssertionError('SD1.5 must not run')):
                result=asyncio.run(service.generate(analysis('neutral')))
                self.assertEqual(result.output_path.read_bytes(),base.read_bytes());live.assert_not_called()
                result=asyncio.run(service.generate(analysis('sadness')))
                self.assertEqual(result.output_path,generated)
                self.assertEqual(live.call_args.args[0]['identity']['source_override'],str(base))
                self.assertNotIn('lora_path',live.call_args.args[0]['identity'])


if __name__=='__main__':unittest.main()
