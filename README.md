
---

# How to Train My Own DeepSeek

This repository documents the workflow for running and training **DeepSeek** models on the Princeton **Adroit** HPC cluster.

> **Note**
> All commands below use my student ID `xz4513` as an example.
> Replace it with **your own NetID** when running.

---

## 1. Connect to Princeton Adroit

### macOS / Linux

```bash
ssh xz4513@adroit.princeton.edu
```

### Windows

If you encounter connection issues, try:

```bash
ssh -m hmac-sha2-256 xz4513@adroit.princeton.edu
```

After entering your password, Duo two-factor authentication may be required.
Passwords **will not be shown** in the terminal while typing—just type and press **Enter**.

---

## 2. Request Scratch Storage (Recommended)

Submit a ticket to Princeton **IT Service** to request a larger scratch quota.

* Suggested size: **400 GB** (1 TB is typically not approved).

---

## 3. Navigate to Scratch Directory

```bash
cd /scratch/network/xz4513/
```

---

## 4. Prepare the DeepSeek Models

Download the DeepSeek model(s) to your **local** computer first.

* Recommended: both **deepseek-1.3b** and **deepseek-7b**

  * *7b* gives better performance
  * *1.3b* is faster and ideal for debugging

### Upload to Adroit

Run the following **on your local terminal**:

macOS / Linux

```bash
scp D:\deepseek\deepseek-llm-7b-base.zip \
    xz4513@adroit.princeton.edu:/scratch/network/xz4513/
```

Windows (if errors occur)

```bash
scp -o MACs=hmac-sha2-256 -r D:\deepseek\deepseek-llm-7b-base.zip \
    xz4513@adroit.princeton.edu:/scratch/network/xz4513/
```

---

## 5. Create & Activate a Conda Environment

Make sure you are in the scratch directory:

```bash
module load anaconda3/2024.6
export CONDA_PKGS_DIRS=/scratch/network/xz4513/conda_pkgs
export TMPDIR=/scratch/network/xz4513/tmp
export HOME=/scratch/network/xz4513/conda_home

conda activate /scratch/network/xz4513/deepseek_env
```

---

## 6. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 7. Use Princeton GPUs

You can access GPUs in **two** ways.

### (a) Submit a Slurm Job

Create a script like `pretrain.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --output=/scratch/network/xz4513/pretrain_output.log
#SBATCH --error=/scratch/network/xz4513/pretrain_error.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00

cd /scratch/network/xz4513/
module purge
module load anaconda3/2024.6
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /scratch/network/xz4513/deepseek_env

python pretrain.py
```

Submit:

```bash
sbatch pretrain.slurm
```

### (b) Interactive Debug Session

```bash
salloc --partition=gpu --gres=gpu:nvidia_a100:1 \
       --cpus-per-task=8 --mem=80G --time=04:00:00
```

This is useful for quick debugging.
If it hangs, just wait—GPU queues can be long during peak hours.

---

## 8. Two Training Modes

DeepSeek can be customized in two main ways:

| Mode                                     | Analogy           | Description                                                             |
| ---------------------------------------- | ----------------- | ----------------------------------------------------------------------- |
| **RAG (Retrieval-Augmented Generation)** | Open-book exam    | The model remains general but can search a knowledge base for answers.  |
| **Pretraining**                          | Closed-book study | The model “learns” your dataset and answers without external retrieval. |

---

### A. RAG Mode

Prepare:

* A text file with all knowledge, e.g. `Hogwarts.txt`
* An embedding model (already uploaded)

Steps:

```bash
python build_embedding_index_Hogwarts.py   # creates Hogwarts_index.faiss & Hogwarts_chunks.npy
python chat_with_deepseek.py               # interactive chat (for salloc)
# or
python h_test2.py                           # batch query (for Slurm)
```

---

### B. Pretrain Mode

Prepare:

* A text file with all training content, e.g. `harrypotter.txt`

Steps:

```bash
python pretrain_harrypotter.py
```

* With **DeepSeek-1.3b**, a 2 GB training text, and **4× A100 GPUs**, training takes \~7 hours.

After training, test and compare:

```bash
python try_rmrb.py
```

`checkpoint-xxxx` directories are intermediate saves.

---

## 9. Monitoring Training

* Training loss typically ranges **0 – 3** and is not a perfect quality metric.
* Rough guideline:

  * Cohesive single-topic text → loss around **0.5**
  * Mixed/noisy topics → loss may rise to **1–2+**

---

### 🎯 Success Criterion

If models trained on different datasets give noticeably different answers to the **same query**, congratulations—you have successfully fine-tuned DeepSeek! 🎉

---

This README provides all key steps for running and customizing DeepSeek on Princeton’s Adroit cluster. Adjust IDs, file paths, and parameters as needed for your own environment.
