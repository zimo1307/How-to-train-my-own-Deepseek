import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_PATH  = "deepseek_1.3b_base_local"
CKPT_PATH  = "/scratch/network/xz4883/deepseek_output_cont_used_original/checkpoint-3000"  # 改成你的
CKPT_PATH1 =  "/scratch/network/xz4883/aaadeepseek_output/checkpoint-6000"
prompts = [
    "请用人民日报风格，用3句话概述“推进新型工业化”的核心要点：",
    "围绕主题‘乡村振兴’，列出三条具有可操作性的实施建议：",
    "围绕主题“中美关系”，列出三条积极、建设性的表述要点：",
    "三体文明关于引力透镜事件的官方态度是什么？",
    "请列出‘霍格沃茨重建工程’新闻报道中的三个重点：",
]

def load_model(path):
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return tok, model

def gen(tok, model, text):
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True, temperature=0.8, top_p=0.9,
            pad_token_id=tok.eos_token_id
        )
    return tok.decode(out[0], skip_special_tokens=True)

if __name__ == "__main__":
    tok_base,  m_base  = load_model(BASE_PATH)
    tok_ckpt,  m_ckpt  = load_model(CKPT_PATH)
    tok_ckpt1,  m_ckpt1  = load_model(CKPT_PATH1)

    for i, p in enumerate(prompts, 1):
        print(f"\n==== Case {i} | Prompt ====\n{p}\n")
        print("---- Base ----")
        print(gen(tok_base, m_base, p))
        print("\n---- Finetuned ----")
        print(gen(tok_ckpt, m_ckpt, p))
        print("\n---- Fakenews ----")
        print(gen(tok_ckpt1, m_ckpt1, p))
