import torch
from transformers import BertTokenizer, BertModel, T5Tokenizer, T5EncoderModel

def get_hunyuan_text_embeddings(
    prompt: str,
    model_root_path: str,
    device: str = "cuda",
    max_length_bert: int = 77,
    max_length_t5: int = 256,
):
    """
    根据 pretrain_dit 的目录结构加载 BERT 和 T5，并生成 HunyuanDiT 所需的嵌入。
    """
    
    # 1. 加载 BERT (对应 text_encoder 和 tokenizer 文件夹)
    # 注意：HunyuanDiT 的 text_encoder 通常是基于 BERT 结构的
    bert_tokenizer = BertTokenizer.from_pretrained(model_root_path, subfolder="tokenizer")
    bert_model = BertModel.from_pretrained(model_root_path, subfolder="text_encoder").to(device)

    # 2. 加载 T5 (对应 text_encoder_2 和 tokenizer_2 文件夹)
    t5_tokenizer = T5Tokenizer.from_pretrained(model_root_path, subfolder="tokenizer_2")
    t5_model = T5EncoderModel.from_pretrained(model_root_path, subfolder="text_encoder_2").to(device)

    with torch.no_grad():
        # --- A. 准备 encoder_hidden_states (BERT) ---
        bert_inputs = bert_tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length_bert,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        
        bert_output = bert_model(**bert_inputs)
        # BERT 的输出取 last_hidden_state
        encoder_hidden_states = bert_output.last_hidden_state
        text_embedding_mask = bert_inputs.attention_mask

        # --- B. 准备 encoder_hidden_states_t5 (T5) ---
        t5_inputs = t5_tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length_t5,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        t5_output = t5_model(**t5_inputs)
        # T5EncoderModel 的输出取 last_hidden_state
        encoder_hidden_states_t5 = t5_output.last_hidden_state
        text_embedding_mask_t5 = t5_inputs.attention_mask

    # 清理显存
    del bert_model, bert_tokenizer, t5_model, t5_tokenizer
    torch.cuda.empty_cache()

    return (
        encoder_hidden_states,      # 传给 encoder_hidden_states
        text_embedding_mask,        # 传给 text_embedding_mask
        encoder_hidden_states_t5,   # 传给 encoder_hidden_states_t5
        text_embedding_mask_t5      # 传给 text_embedding_mask_t5
    )

if __name__ == '__main__':
    # 测试代码
    path = "/home/ldp/LXW/SemFlow-xu/delta-FM/TransSegFlow/dataset/pretrain_dit" # 您的实际路径
    prompt = "透明物体"
    
    try:
        emb1, mask1, emb2, mask2 = get_hunyuan_text_embeddings(prompt, path)
        print("成功生成嵌入！")
        print(f"BERT shape: {emb1.shape}") # 预期 (1, 77, 1024)
        print(f"T5 shape:   {emb2.shape}") # 预期 (1, 256, 2048)
    except Exception as e:
        print(f"错误: {e}")