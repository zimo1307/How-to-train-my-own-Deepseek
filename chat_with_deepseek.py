from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "deepseek_1.3b_base_local"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 当前使用的设备是: {device.upper()}")

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    local_files_only=True
).to(device)

chat_history = []

print("DeepSeek Chat Ready! Type 'exit' to quit")

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    chat_history.append({"role": "user", "content": user_input})

    # 构建 prompt，首轮加入角色设定防止模型跑偏
    prompt = ""
    if len(chat_history) == 1:
        prompt += "<|user|>\n You are a very helpful AI assistant \n<|assistant|>\n Hello, what can I do for you today? \n"

    for turn in chat_history:
        if turn["role"] == "user":
            prompt += f"<|user|>\n{turn['content']}\n"
        else:
            prompt += f"<|assistant|>\n{turn['content']}\n"
    prompt += "<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    new_text = decoded[len(prompt):].strip()
    response = new_text.split("<|")[0].strip()

    print(f"Deepseek: {response}")
    chat_history.append({"role": "assistant", "content": response})
