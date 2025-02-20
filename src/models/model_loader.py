from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_path="./models/model"):
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer
