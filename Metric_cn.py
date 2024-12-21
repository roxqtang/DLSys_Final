import re
from typing import List, Dict
import numpy as np
import json

def tokenize_chinese_text(text) -> List[str]:
    # Tokenizes Chinese text by splitting it into individual characters
    return [char for char in text]

def split_into_sentences(original_text: List[str]) -> List[List[str]]:
    """
    Split a list of Chinese texts into sentences.
    
    Args:
        original_text (List[str]): List of Chinese texts to be split
        
    Returns:
        List[List[str]]: List of lists where each inner list contains sentences from one text
    """
    all_sentences = []
    
    for text in original_text:
        # Split on common Chinese sentence endings (。！？；), preserve the punctuation
        sentences = re.split('([。！？；])', text)
        
        # Pair up sentences with their punctuation
        paired = [''.join(i) for i in zip(sentences[0::2], sentences[1::2] + [''] * (len(sentences[0::2]) - len(sentences[1::2])))]
        
        # Remove empty strings and whitespace
        cleaned = [s.strip() for s in paired if s.strip()]
        
        all_sentences.append(cleaned)
    
    return all_sentences

def get_lcs_table(ref_tokens: List[str], pred_tokens: List[str]) -> np.ndarray:
    """
    Compute the Longest Common Subsequence table (2 points)
    """
    ref_length = len(ref_tokens)
    pred_length = len(pred_tokens)
    # Implement dynamic programming for the LCS problem
    lcs_table = np.zeros((ref_length+1, pred_length+1))
    for i in range(1, ref_length+1):
        for j in range(1, pred_length+1):
            if ref_tokens[i-1] == pred_tokens[j-1]:
                lcs_table[i, j] = lcs_table[i-1, j-1] + 1
            else:
                lcs_table[i, j] = max(lcs_table[i-1, j], lcs_table[i, j-1])
    return lcs_table

def compute_rouge_l(reference: List[str], prediction: List[str], beta: float = 1.2) -> Dict[str, float]:
    """
    Basic ROUGE-L computation (4 points)
    """
    if not reference or not prediction:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    # First find the LCS
    lcs_table = get_lcs_table(reference, prediction)
    lcs_length = lcs_table[-1, -1]
    recall = lcs_length / len(reference) if len(reference) > 0 else 0.0
    precision = lcs_length / len(prediction) if len(prediction) > 0 else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def compute_rouge_lsum(reference: List[str], prediction: List[str], beta: float = 1.2) -> Dict[str, float]:
    """
    Compute ROUGE-LSum score (5 points)
    """
    try:
        # Split into sentences,
        # and now we have a list of sentence tokens
        # for both reference and prediction sentences
        ref_sentences = split_into_sentences(reference)
        pred_sentences = split_into_sentences(prediction)
        ref_length = sum(map(len, ref_sentences))
        pred_length = sum(map(len, pred_sentences))
        
        used_preds = set()
        best_match_length = 0
        # Calculate LCS for each reference sentence
        for ref_sentence in ref_sentences:
            # Find the best matching prediction sentence
            best_match = None
            best_lcs_length = 0
            for idx, pred_sentence in enumerate(pred_sentences):
                if idx not in used_preds:
                    lcs_table = get_lcs_table(ref_sentence, pred_sentence)
                    lcs_length = lcs_table[-1, -1]
                    if lcs_length > best_lcs_length:
                        best_lcs_length = lcs_length
                        best_match = idx
            
            if best_match is not None:
                used_preds.add(best_match)
                best_match_length += best_lcs_length

        # Calculate final scores
        recall = best_match_length / ref_length if ref_length > 0 else 0
        precision = best_match_length / pred_length if pred_length > 0 else 0
        f1 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    except Exception as e:
        print(f"Error in ROUGE-LSum computation: {e}")
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

if __name__ == "__main__":
    file_path = '/scratch/qt2094/DLSYS/DLSys_Final/eval_output/llama3_gen_eval_DQ.json'
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    total_rouge_lsum = 0
    total_rouge_l = 0
    for sample in data:
        ref_output = sample['ref_output']
        pred_output = sample['pred_output']

        # Tokenize the reference and prediction outputs
        tokenized_ref_output = tokenize_chinese_text(ref_output)
        tokenized_pred_output = tokenize_chinese_text(pred_output)
        
        # Compute ROUGE-LSum and ROUGE-L scores
        rouge_lsum = compute_rouge_lsum(tokenized_ref_output, tokenized_pred_output)
        rouge_l = compute_rouge_l(tokenized_ref_output, tokenized_pred_output)
        print(f'rouge_lsum: {rouge_lsum}')
        print(f'rouge_l: {rouge_l}')
        total_rouge_lsum += rouge_lsum['f1']
        total_rouge_l += rouge_l['f1']
    print(f'avg rouge_lsum: {total_rouge_lsum/len(data)}')
    print(f'avg rouge_l: {total_rouge_l/len(data)}')