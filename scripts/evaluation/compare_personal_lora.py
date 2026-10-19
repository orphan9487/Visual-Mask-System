"""Fixed-seed offline comparison; does not change saved mask settings."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))

def main():
    from src.services.mask_generation_settings import read_job,settings,checkpoint_path
    from src.integrations.liveportrait_expression import generate_from_expression_profile,stylize_with_sd15
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True)
    parser.add_argument('--emotion',default='anger',choices=['anger','joy','sadness','neutral','fear','surprise','disgust'])
    parser.add_argument('--checkpoints',nargs='+',type=int,default=[100,200,300])
    parser.add_argument('--strengths',nargs='+',type=float,default=[.22,.3,.4])
    parser.add_argument('--weights',nargs='+',type=float,default=[.7])
    args=parser.parse_args();job=read_job(args.job);config=settings(job)
    identity={'mask_id':'lora_'+job['id'],'trigger':job['trigger'],'source_override':job['paths'][config['source_index']],'seed':config['seed']}
    source=generate_from_expression_profile({'emotion':args.emotion,'identity':identity})
    records=[]
    for checkpoint in args.checkpoints:
        for weight in args.weights:
            for strength in args.strengths:
                output=stylize_with_sd15(source,args.emotion,strength,{**identity,'lora_path':checkpoint_path(job,checkpoint),'lora_weight':weight})
                records.append(dict(checkpoint=checkpoint,weight=weight,strength=strength,seed=config['seed'],output=output,source=source))
                print(records[-1],flush=True)
    destination=Path(source).with_suffix('.comparison.json');destination.write_text(json.dumps(records,indent=2),encoding='utf-8');print(destination)

if __name__=='__main__':main()
