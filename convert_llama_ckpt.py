import torch
from pathlib import Path






CHUNK_DIM_0 = [
    
    'output.weight',
    'attention.wq.weight',
    'attention.wk.weight',
    'attention.wv.weight',
    
    'feed_forward.w3.weight',
    'feed_forward.w1.weight',
    
]


CHUNK_DIM_1 = [
    'tok_embeddings.weight',
    'attention.wo.weight',
    'feed_forward.w2.weight',
]

def _is_target_weight(key, patterns):
    is_target = False
    for p in patterns:
        if p in key:
            is_target = True
            break
    return is_target





if __name__ == '__main__':
    import argparse


    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--ckpt-dir', type=str,
        help='llama official ckpt dir to convert'
    )
    parser.add_argument(
        '--save-dir', type=str,
        help='dir for saving the converted pth',
    )
    parser.add_argument(
        '--num-rank', type=int, default=1,
        help='How many ranks for parallelization'
    )
    
    args = parser.parse_args()

    num_rank = args.num_rank
    ckpt_dir = Path(args.ckpt_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_files = sorted(ckpt_dir.glob("*.pth"))
    checkpoints = [torch.load(ckpt) for ckpt in ckpt_files]
    
    coverted_ckpts = [dict() for rank in range(num_rank)]


    for k,v in checkpoints[0].items():
        new_key = k
        
        if _is_target_weight(new_key, CHUNK_DIM_1):
            
            weights = torch.cat([c[k] for c in checkpoints],dim=1)
            
            assert weights.size(1) % num_rank == 0
            splited_weights = torch.chunk(weights, num_rank,dim=1)
            
            for rank in range(num_rank):
                coverted_ckpts[rank].update({new_key:splited_weights[rank]})
            # continue
        elif _is_target_weight(new_key, CHUNK_DIM_0):
            
            weights = torch.cat([c[k] for c in checkpoints],dim=0)
            
            assert weights.size(0) % num_rank == 0
            splited_weights = torch.chunk(weights, num_rank,dim=0)
            for rank in range(num_rank):
                coverted_ckpts[rank].update({new_key:splited_weights[rank]}) 
            # continue
        else:
            for rank in range(num_rank):
                coverted_ckpts[rank][new_key] = v
            
        
        
    for rank, ckpt in enumerate(coverted_ckpts) :
        filename = f'consolidated.0{rank}.pth'
        torch.save(ckpt, save_dir/filename)
        

    import shutil
    
    shutil.copy(ckpt_dir/'params.json', save_dir/'params.json')