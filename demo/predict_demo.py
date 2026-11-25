import torch
import numpy as np
import os
import pandas as pd
from transformers import AutoTokenizer
from model import LMFFT
from run import Arguments

if __name__ == "__main__":
    args = Arguments()
    args.model_name_or_path = "./pretrained_model"
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.max_seq_len = 1028
    args.ws = 9
    args.model_dim = 1280
    
   
    demo_tsv_path = "./demo/data/demo.tsv"
    
    # 1. 读取序列
    df = pd.read_csv(demo_tsv_path, sep='\t')
    seq_col = next((c for c in ["Prot_seq", "sequence", "seq"] if c in df.columns), df.columns[0])
    sequences = df[seq_col].astype(str).tolist()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # 2. 加载模型
    model = LMFFT(args).to(args.device)
    weight_path = "./pkls/fused_model_Dataset1.pkl" 
    
    state_dict = torch.load(weight_path, map_location=args.device)
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 3. 开始推理
    csv_data = []
    thr = 0.84

    with torch.no_grad():
        for i, seq in enumerate(sequences):
            seq_spaced = " ".join(seq.upper())
            input_ids = tokenizer(seq_spaced).input_ids
            
            if len(input_ids) > args.max_seq_len:
                input_ids = input_ids[:args.max_seq_len]
            
            input_tensor = torch.tensor([input_ids]).to(args.device)
            
            try:
                out_pred = model(input_tensor) 
            except Exception as e:
                print(f"Error at protein {i+1}: {e}")
                continue

            prob_pos = torch.softmax(out_pred, dim=-1)[:, 1].cpu().detach().numpy()
            prob_pos = prob_pos[1:-1] if len(prob_pos) > 2 else prob_pos
            
            pred_labels = (prob_pos > thr).astype(int)
            label_str = "".join(map(str, pred_labels))
            
            csv_data.append({
                "Protein_ID": f"Protein_{i+1:03d}",
                "Length": len(label_str),
                "Predicted_Labels": label_str
            })

    # 4. 保存
    out_df = pd.DataFrame(csv_data)
    out_df.to_csv("./demo/demo-predict.csv", index=False)
    print("✅ 推理完成，结果已保存至 ./demo/demo-predict.csv")
