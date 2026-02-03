import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers.models.qwen2.modeling_qwen2 as qwen2_mod
from cutile.ops.matmul import launch_matmul
from cutile.modules.Qwen2MLP import MyQwen2MLP


# 替换transformers中的Qwen2MLP
qwen2_mod.Qwen2MLP = MyQwen2MLP

model_name = "Qwen/Qwen2.5-1.5B"

def load_model():
    if torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA for inference.")
    else:
        device = "cpu"
        print("CUDA not available, using CPU for inference.")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True
    )
    return tokenizer, model

def generate_response(tokenizer, model, prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response

def chat_loop(tokenizer, model):
    print("欢迎使用本地LLM聊天！输入'exit'退出。")
    conversation_history = []
    while True:
        user_input = input("你: ")
        if user_input.lower() == 'exit':
            break
        conversation_history.append(f"{user_input}")
        prompt = "\n".join(conversation_history) + "\n"
        response = generate_response(tokenizer, model, prompt)
        print(f"AI: {response}")
        conversation_history.append(f"{response}")

if __name__ == "__main__":
    print("加载模型中...")
    tokenizer, model = load_model()
    print("模型加载完成。")
    chat_loop(tokenizer, model)