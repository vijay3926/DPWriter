import re
import functools
import json, jsonlines
from tqdm import tqdm, trange
import requests
from transformers import AutoTokenizer
import random
import numpy as np
from nltk.util import ngrams



def process_batch_info(batch_completions, batch_extra_info):
    # 首先创建分组字典，key为question，value为对应index的列表
    groups = {}
    
    # 遍历batch_extra_info，按照question进行分组
    for i, info in enumerate(batch_extra_info):
        question = info['question']
        
        # 如果这个question还没有在字典中，创建新的列表
        if question not in groups:
            groups[question] = []
        
        # 将当前索引添加到对应question的分组中
        groups[question].append(i)
    
    # 根据分组创建最终的all_group_prompt2completions结构
    all_group_prompt2completions = {}
    
    for question, indices in groups.items():
        group_data = {
            'question': question,
            'completions': [],
            'extra_info': []
        }
        
        # 根据索引收集对应的completions和extra_info
        for idx in indices:
            # 获取对应的completion并清理特殊标记
            completion = batch_completions[idx]
            completion = completion.replace("<|im_end|>", "")
            completion = completion.replace("<|endoftext|>", "")
            completion = completion.replace("<|eot_id|>", "")
            group_data['completions'].append(completion)
            group_data['extra_info'].append(batch_extra_info[idx])
        
        all_group_prompt2completions[question] = group_data
    
    return all_group_prompt2completions



@functools.lru_cache(maxsize=128)
def _compute_group_metrics_cached(group_texts_tuple):
    group_texts = list(group_texts_tuple)
    
    # Tokenize all texts
    tokenized_texts = []
    for t in group_texts:
        # Simple heuristic for tokenization
        is_english = all(ord(c) < 128 for c in t[:100])
        if is_english:
            tokens = t.split()
        else:
            # Use all non-whitespace characters for Chinese/Mixed
            tokens = [c for c in t if c.strip()]
        tokenized_texts.append(tokens)
        
    # Compute N-grams (2 and 3) for each text
    text_ngrams = []
    for tokens in tokenized_texts:
        s = set()
        if len(tokens) >= 2:
            s.update(ngrams(tokens, 2))
        if len(tokens) >= 3:
            s.update(ngrams(tokens, 3))
        if len(tokens) >= 4:
            s.update(ngrams(tokens, 4))
        text_ngrams.append(s)
        
    contributions = []
    
    for i in range(len(group_texts)):
        my_ngrams = text_ngrams[i]
        
        # Union of all other texts' ngrams
        other_ngrams = set()
        for j in range(len(group_texts)):
            if i == j:
                continue
            other_ngrams.update(text_ngrams[j])
            
        # Unique ngrams
        unique = my_ngrams - other_ngrams
        unique_count = len(unique)
        
        # Length normalization: unique_count / total_tokens
        # This prevents longer texts from having an unfair advantage
        total_tokens = len(tokenized_texts[i])
        
        if total_tokens == 0:
            contributions.append(0.0)
        else:
            contributions.append(unique_count / total_tokens)
            
    return contributions

def compute_loo_contribution(text, group_texts):
    """
    计算单个文本对群体的 Leave-One-Out 多样性贡献
    
    Args:
        text: 要计算贡献的单个文本
        group_texts: 包含 text 的完整文本群组列表
    
    Returns:
        dict: {
            'contribution': 贡献度（正值表示该文本增加多样性）,
        }
    """
    # 使用缓存的计算函数
    # 这样可以避免重复计算整个组的贡献度
    
    group_texts_tuple = tuple(group_texts)
    contributions = _compute_group_metrics_cached(group_texts_tuple)

    # 获取目标文本的原始贡献度
    try:
        target_idx = group_texts.index(text)
        target_contribution = contributions[target_idx]
    except ValueError as e:
        print(f"Error in diversity contribution calculation, return 0.")
        return {
            'contribution': 0.0,
            'normalized_contribution': 0.0,
        }

    # 基于分数的归一化
    valid_scores = [c for c in contributions if c != -float('inf')]
    
    if not valid_scores:
        normalized_contribution = 0.0
    elif target_contribution == -float('inf'):
        normalized_contribution = 0.0
    else:
        min_score = min(valid_scores)
        max_score = max(valid_scores)
        
        if max_score == min_score:
            normalized_contribution = 1.0
        else:
            normalized_contribution = (target_contribution - min_score) / (max_score - min_score)
    
    return {
        'contribution': target_contribution if target_contribution != -float('inf') else 0.0,
        'normalized_contribution': normalized_contribution,
    }


def skywork_reward_w_div06(task, completion, reference, task_extra_info, batch_info=None):
    """
    给定 prompt 和 response, 返回 reward 分数
    prompt: str, response: str, base_url: str) -> float:
    """

    new_tokens = [
        "<think>", "</think>", "<goal>", "</goal>", "<info>", "</info>", 
        "<struct>", "</struct>", "<lang>", "</lang>", "<pres>", "</pres>"
    ]
    for new_tk in new_tokens:
        if new_tk not in completion:
            print(f"In rewarding: token {new_tk} no in completion, return -50!")
            return -50.0

    completion = completion.replace("<|im_end|>", "")
    completion = completion.replace("<|endoftext|>", "")
    completion = completion.replace("<|eot_id|>", "")

    model_name_or_path = "/YOUR_PATH/Skywork-Reward-V2-Llama-3.1-8B"
    # base_urls = [f"http://127.0.0.1:{18000 + i}/classify" for i in range(4)]
    base_urls = [f"http://xx.xx.xx.xx:{18000 + i}/classify" for i in range(4)]
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    MAX_TRIES = 3
    if not completion:
        print("No completion provided in kwargs.")

    prompt = task_extra_info['question'].split("\n\nuser\n\n")[-1]
    response = completion.split('</think>')[-1] # .strip()
    # print(f"prompt: {prompt}, response: {response}")

    # 构造对话格式
    conv = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    conv_formatted = tokenizer.apply_chat_template(conv, tokenize=False)

    # 去掉 BOS token（如果存在）
    if tokenizer.bos_token is not None and conv_formatted.startswith(tokenizer.bos_token):
        conv_formatted = conv_formatted[len(tokenizer.bos_token):]

    # 构造请求
    payload = {"model": model_name_or_path, "text": [conv_formatted]}
    reward = 0.0
    while MAX_TRIES > 0:
        try:
            base_url = random.choice(base_urls)
            # print(f"Sending request to {base_url}")
            api_response = requests.post(base_url, json=payload).json()
            reward = api_response[0]["embedding"][0]
            break
        except Exception as e:
            print(f"Error during rewarding request: {e}")
            MAX_TRIES -= 1
            if MAX_TRIES == 0:
                print("Max retries reached. Returning default rewards.")
                return reward

    ### for diversity
    if batch_info:
        (batch_completions, batch_extra_info) = batch_info
        all_group_prompt2completions = process_batch_info(batch_completions, batch_extra_info)
        group_completions = all_group_prompt2completions[prompt]['completions']
        group_responses = [x.split("</think>")[-1].strip() for x in group_completions]
        div_contribution = compute_loo_contribution(response.strip(), group_responses)['normalized_contribution']
        # print(reward, div_contribution)
        if reward > 10:
            div_lambda = 0.6
            reward = (1 - div_lambda) * reward + div_lambda * reward * div_contribution

    return reward



if __name__ == "__main__":
    pass

