import os, time, signal, atexit
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, TrainerCallback, set_seed
)
from transformers.trainer_utils import get_last_checkpoint

# ====== 基本配置 ======
MODEL_PATH = "deepseek_1.3b_base_local"  # 你的本地模型目录
DATA_PATH  = "/scratch/network/xz4883/harrypotter.txt"  # ← 改成你的 HP 文本路径
OUT_DIR    = "/scratch/network/xz4883/aaadeepseek_output_hp"  # 沿用原目录即可断点续训

SAVE_STEPS = 800          # 建议 400~1000 之间，4 小时内至少落盘 3~5 次
SAVE_TOTAL_LIMIT = 3      # 至少留 2~3 份，防止单点损坏
BLOCK_SIZE = 512

set_seed(42)

# ====== 数据集（小样本内存版；全量可换 IterableDataset）======
class TxtDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=512):
        self.examples = []
        with open(file_path, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        for line in lines:
            tok = tokenizer(
                line,
                truncation=True,
                max_length=block_size,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = tok["input_ids"].squeeze(0)
            attn = tok["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[attn == 0] = -100  # PAD不计loss
            self.examples.append(
                {"input_ids": input_ids, "attention_mask": attn, "labels": labels}
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

# ====== “瘦身”回调：只删优化器等大文件，保留 trainer_state.json 以便续训 ======
class PruneCheckpointCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.exists(ckpt_dir):
            subs = [os.path.join(args.output_dir, d)
                    for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
            if subs:
                ckpt_dir = max(subs, key=os.path.getmtime)
            else:
                return
        for fname in ["optimizer.pt", "scheduler.pt", "rng_state.pth"]:
            fpath = os.path.join(ckpt_dir, fname)
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except Exception as e:
                    print(f"[PruneCheckpoint] remove {fname} failed: {e}")

# ====== 初始化 tokenizer / model ======
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
model.config.use_cache = False
model.gradient_checkpointing_enable()

# ====== 数据集 ======
dataset = TxtDataset(tokenizer, DATA_PATH, BLOCK_SIZE)

# ====== 训练参数（A100: bf16 + tf32）======
training_args = TrainingArguments(
    output_dir=OUT_DIR,
    overwrite_output_dir=False,       # 防止误覆盖，以便续训
    num_train_epochs=30,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    weight_decay=0.01,

    logging_dir=os.path.join(OUT_DIR, "logs"),
    logging_steps=10,

    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    save_safetensors=True,
    save_on_each_node=False,          # 只在 rank0 保存

    report_to="tensorboard",

    bf16=True,                        # A100 推荐
    fp16=False,
    tf32=True,

    dataloader_num_workers=2,
    dataloader_pin_memory=True,

    ddp_find_unused_parameters=False,
    gradient_checkpointing=True,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    callbacks=[PruneCheckpointCallback()],
)

# ====== 捕获超时/中断信号，临终再存一次 ======
def _save_and_exit(signum=None, frame=None):
    try:
        if trainer.is_world_process_zero():
            path = os.path.join(OUT_DIR, "checkpoint-interrupt")
            os.makedirs(path, exist_ok=True)
            trainer.save_model(path)
            print(f"[Signal] Saved interrupt checkpoint to {path}")
    finally:
        # 立刻退出，避免被强制 KILL
        os._exit(0)

signal.signal(signal.SIGTERM, _save_and_exit)
signal.signal(signal.SIGINT,  _save_and_exit)
atexit.register(_save_and_exit)  # 进程正常结束也存一份

if __name__ == "__main__":
    # 自动查找最新 checkpoint 续训
    last_ckpt = get_last_checkpoint(OUT_DIR) if os.path.isdir(OUT_DIR) else None
    if last_ckpt:
        print(f"[Resume] Found checkpoint: {last_ckpt}")
    trainer.train(resume_from_checkpoint=last_ckpt)

    # 结束前再存一次（只在rank0做）
    if trainer.is_world_process_zero():
        trainer.save_model()
        print("[Finish] Final model saved.")
