import torch
from transformers import CLIPTextModel, CLIPTokenizer, BertTokenizer, BertModel

def sd_null_condition(path):
    text = "透明物体"
    text_encoder = CLIPTextModel.from_pretrained(path, subfolder="text_encoder", revision=None)
    tokenizer = CLIPTokenizer.from_pretrained(path, subfolder="tokenizer", revision=None)
    with torch.no_grad():
        empty_inputs = tokenizer(text, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
        emptyembed = text_encoder(empty_inputs.input_ids)[0]
    del text_encoder, tokenizer
    return emptyembed


if __name__ == '__main__':
    pass