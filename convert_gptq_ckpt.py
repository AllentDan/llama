import torch
from pathlib import Path

MAPPINGS = [
    ('lm_head', 'output'),
    ('model.norm','norm'),
    ('model.embed_tokens', 'tok_embeddings'),
    ('model.layers', 'layers'),
    ('self_attn.k_proj', 'attention.wk'),
    ('self_attn.q_proj', 'attention.wq'),
    ('self_attn.v_proj', 'attention.wv'),
    ('self_attn.o_proj', 'attention.wo'),
    ('mlp.down_proj','feed_forward.w2'),
    ('mlp.gate_proj','feed_forward.w1'),
    ('mlp.up_proj','feed_forward.w3'),
    ('input_layernorm','attention_norm'),
    ('post_attention_layernorm','ffn_norm'),
    ('self_attn.rotary_emb','attention.rotary_emb'),
]


CHUNK_DIM_1 = [
    'tok_embeddings.weight',
    
    'attention.wq.qweight',
    'attention.wk.qweight',
    'attention.wv.qweight',
    'attention.wq.scales',
    'attention.wk.scales',
    'attention.wv.scales',
    'attention.wq.qzeros',
    'attention.wk.qzeros',
    'attention.wv.qzeros',
    'feed_forward.w3.qweight',
    'feed_forward.w1.qweight',
    'feed_forward.w3.scales',
    'feed_forward.w1.scales',
    'feed_forward.w3.qzeros',
    'feed_forward.w1.qzeros',
]


CHUNK_DIM_0 = [
    'output.weight',
    'attention.wo.qweight',
    'attention.wo.scales',
    'attention.wo.qzeros',
    'feed_forward.w2.qweight',
    'feed_forward.w2.scales',
    'feed_forward.w2.qzeros',
    # 'attention.wq.bias',
    # 'attention.wq.bias',
    # 'attention.wq.bias',
    # 'attention.wq.scales',
    # 'attention.wk.scales',
    # 'attention.wv.scales',
    # 'feed_forward.w3.scales',
    # 'feed_forward.w1.scales',
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
        '--ckpt', type=str,
        help='llama ckpt(from gptq) to convert'
    )
    parser.add_argument(
        '--save-dir', type=str,
        help='dir for saving the converted pth',
    )
    parser.add_argument(
        '--params-json', type=str,
        help='llama official params json',
    )
    parser.add_argument(
        '--num-rank', type=int, default=1,
        help='How many ranks for parallelization'
    )
    
    args = parser.parse_args()

    num_rank = args.num_rank

    if args.ckpt.endswith('safetensors'):
        from safetensors.torch import load_file as safe_load
        source_ckpt = safe_load(args.ckpt)
    else:
        source_ckpt = torch.load(args.ckpt)
        
    coverted_ckpts = [dict() for rank in range(num_rank)]


    for k,v in source_ckpt.items():
        new_key = k
        for m in MAPPINGS:
            new_key = new_key.replace(m[0],m[1])
            
        if 'bias' in new_key:
            assert v.sum() == 0
            continue
            
            
        if _is_target_weight(new_key, CHUNK_DIM_1) and num_rank > 1:
            assert v.size(1) % num_rank == 0
            splited_weights = torch.chunk(v, num_rank,dim=1)
            
            for rank in range(num_rank):
                coverted_ckpts[rank].update({new_key:splited_weights[rank]})
            # continue
        elif _is_target_weight(new_key, CHUNK_DIM_0) and num_rank > 1 and v.size(0) != 1:
            assert v.size(0) % num_rank == 0
            splited_weights = torch.chunk(v, num_rank,dim=0)
            for rank in range(num_rank):
                coverted_ckpts[rank].update({new_key:splited_weights[rank]}) 
            # continue
        else:
            for rank in range(num_rank):
                coverted_ckpts[rank][new_key] = v
        print(new_key, 'Done', v.shape,' > ' ,coverted_ckpts[rank][new_key].shape)
    save_dir = Path(args.save_dir)
        
    save_dir.mkdir(parents=True, exist_ok=True)
    for rank, ckpt in enumerate(coverted_ckpts) :
        filename = f'consolidated.0{rank}.pth'
        torch.save(ckpt, f'{args.save_dir}/{filename}')
        
    import shutil
    
    shutil.copy(args.params_json, save_dir/'params.json')