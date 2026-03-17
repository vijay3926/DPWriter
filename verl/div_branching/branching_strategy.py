import torch
import numpy as np
import logging
import re
from collections import Counter
from typing import List, Tuple, Dict, Set, Optional
from verl import DataProto
from tensordict import TensorDict
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from nltk.util import ngrams

logger = logging.getLogger(__name__)



class BranchingRolloutStrategy:
    def __init__(self, config, tokenizer, k_branches=32, diversity_metric="ngram"):
        self.config = config
        self.tokenizer = tokenizer
        self.k_branches = k_branches
        self.diversity_metric = diversity_metric
        
        # Define segments (prefix, start_token, end_token)
        # prefix: The intermediate string to append before the start_token
        self.segments_zh = [
            ("<think>\n为了有创意地回复，我先从多个方面进行发散分析，然后一步一步细致思考。\n", "<goal>目标与受众：", "</goal>"),
            ("\n", "<info>信息与视角：", "</info>"),
            ("\n", "<struct>结构与逻辑：", "</struct>"),
            ("\n", "<lang>语言与风格：", "</lang>"),
            ("\n", "<pres>呈现与体验：", "</pres>")
        ]

        self.segments_en = [
            ("<think>\nTo respond creatively, I first conduct a divergent analysis from multiple perspectives and then think carefully step by step.\n", "<goal>Goal and audience:", "</goal>"),
            ("\n", "<info>Information and perspective:", "</info>"),
            ("\n", "<struct>Structure and logic:", "</struct>"),
            ("\n", "<lang>Language and style:", "</lang>"),
            ("\n", "<pres>Presentation and experience:", "</pres>")
        ]
        
        # Use 8 GPUs by default as requested
        self.emb_api_ports = list(range(8000, 8008))
        self.emb_api_urls = [f"http://{config.actor_rollout_ref.rollout.get('emb_host', '0.0.0.0')}:{p}/embed" for p in self.emb_api_ports]

    def get_segments(self, lang='zh'):
        if lang == 'en':
            return self.segments_en
        return self.segments_zh
    
    # def get_ngrams(self, text: str, n: int = 3) -> set:
    #     tokenized = self.tokenizer.tokenize(text)
    #     s = set(ngrams(tokenized, 2))
    #     s.update(ngrams(tokenized, 3))
    #     return s
    
    def get_ngrams_batch(self, texts: List[str]) -> List[set]:
        """Batch version of get_ngrams - more efficient for multiple texts. Includes caching."""
        # Initialize cache if needed
        if not hasattr(self, 'ngram_cache'):
            self.ngram_cache = {}

        result = []
        for text in texts:
            if text in self.ngram_cache:
                result.append(self.ngram_cache[text])
            else:
                tokenized = self.tokenizer.tokenize(text)
                # s = set(ngrams(tokenized, 2))
                # s.update(ngrams(tokenized, 3))
                
                s = set()
                # 包含1-4gram
                for n in range(2, 5):  # range(1, 5) 生成 1, 2, 3, 4
                    s.update(ngrams(tokenized, n))

                self.ngram_cache[text] = s
                result.append(s)
        return result

    def extract_last_tag_content(self, text: str) -> str:
        """
        Efficiently extract the content of the last tag <tag>content</tag> from the text.
        Searches from the end of the string to avoid parsing the full text.
        """
        # Find the last closing tag start "</"
        end_tag_start = text.rfind("</")
        if end_tag_start == -1:
            return text
            
        # Find the closing bracket ">" of the end tag
        end_tag_end = text.find(">", end_tag_start)
        if end_tag_end == -1:
            return text
            
        # Extract the tag name, e.g. "goal" from "</goal>"
        # end_tag_start points to '<', so +2 is after '/'
        tag_name = text[end_tag_start + 2 : end_tag_end]
        
        # Construct the start tag string, e.g. "<goal>"
        start_tag = f"<{tag_name}>"
        
        # Find the corresponding start tag searching backwards from the end tag
        start_tag_idx = text.rfind(start_tag, 0, end_tag_start)
        if start_tag_idx == -1:
            return text
            
        # Content is between end of start tag and start of end tag
        content_start = start_tag_idx + len(start_tag)
        content = text[content_start : end_tag_start]
        return content

    def _batch_validate_contents(self, texts: List[str], end_tag_str: str) -> List[bool]:
        """
        Batch validation of contents - more efficient than individual calls.
        Returns a list of booleans indicating validity for each text.
        """
        results = []
        for text in texts:
            # Quick check first (no tokenization needed)
            if not text.endswith(end_tag_str):
                results.append(False)
                continue
            
            content = self.extract_last_tag_content(text)
            text_length = len(content)
            text_stripped = content.strip()
            
            if not content or not text_stripped or len(text_stripped) < 10:
                results.append(False)
                continue
                
            if text_length > 10 and (content.count('\n') / text_length > 0.2):
                results.append(False)
                continue
            
            # Only tokenize if passed quick checks
            tokens = self.tokenizer.tokenize(content)
            if not tokens:
                results.append(False)
                continue
            token_counts = Counter(tokens)
            most_common_ratio = token_counts.most_common(1)[0][1] / len(tokens)
            
            results.append(most_common_ratio <= 0.3)
        
        return results

    def remove_wrcot_tokens(self, text):
        # Use pre-compiled regex for better performance
        if not hasattr(self, '_wrcot_pattern'):
            rm_tokens = [
                "<goal>目标与受众：", "</goal>", "<goal>Goal and audience: ", 
                "<info>信息与视角：", "</info>", "<info>Information and perspective: ",
                "<struct>结构与逻辑：", "</struct>", "<struct>Structure and logic: ", 
                "<lang>语言与风格：", "</lang>", "<lang>Language and style: ",
                "<pres>呈现与体验：", "</pres>", "<pres>Presentation and experience: "
            ]
            # Escape special regex characters and join with |
            escaped = [re.escape(t) for t in rm_tokens]
            self._wrcot_pattern = re.compile('|'.join(escaped))
            
            head_tokens = [
                "目标与受众：", "Goal and audience: ", 
                "信息与视角：", "Information and perspective: ",
                "结构与逻辑：", "Structure and logic: ", 
                "语言与风格：", "Language and style: ",
                "呈现与体验：", "Presentation and experience: "
            ]
            escaped_head = [re.escape(t) for t in head_tokens]
            self._head_pattern = re.compile('^(' + '|'.join(escaped_head) + ')')
        
        text = self._wrcot_pattern.sub('', text)
        text = self._head_pattern.sub('', text)
        return text

    def select_diverse_indices_ngram(self, candidates_text: List[str], n_select: int, end_tag_str: str, norm_by_length: bool = True) -> List[int]:
        """
        Select n_select indices from candidates_text based on N-gram novelty.
        Works for both Chinese and English by using character-level N-grams.
        Note: candidates_text contains the <end_token> at the end!
        """
        assert len(candidates_text) >= n_select, "Candidates text must contain at least 'n_select' elements!"
        
        # Batch validate all candidates at once
        validity_results = self._batch_validate_contents(candidates_text, end_tag_str)
        valid_indices_set = {i for i, is_valid in enumerate(validity_results) if is_valid}

        # Extract the last tag content and clean it - batch process
        contents = [self.remove_wrcot_tokens(self.extract_last_tag_content(t)) for t in candidates_text]
        
        # Pre-compute n-grams for all candidates in batch
        candidates_ngrams = self.get_ngrams_batch(contents)
        
        selected_indices = []
        global_seen_ngrams = set()
        
        # Prioritize valid indices
        if len(valid_indices_set) >= n_select:
            remaining_indices = valid_indices_set.copy()
        else:
            # If not enough valid candidates, use all candidates
            logging.warning(f"Not enough valid candidates ({len(valid_indices_set)}) for {n_select} selections. Using all candidates.")
            remaining_indices = set(range(len(candidates_text)))

        for _ in range(n_select):
            best_idx = -1
            max_new_score = -1
            
            # Greedy selection: pick the candidate that adds the most NEW n-grams
            for idx in remaining_indices:
                current_ngrams = candidates_ngrams[idx]
                
                # Calculate novelty: count of ngrams not yet in global_seen_ngrams
                # len(A - B) is faster than iterating
                new_ngrams_count = len(current_ngrams - global_seen_ngrams)
                
                if not norm_by_length:
                    # We want to maximize the number of new n-grams introduced
                    # If tie, prefer the one encountered earlier (stable sort equivalent)
                    if new_ngrams_count > max_new_score:
                        max_new_score = new_ngrams_count
                        best_idx = idx
                else:
                    # Normalize by total ngrams to remove length bias
                    total_ngrams = len(current_ngrams)
                    score = new_ngrams_count / (total_ngrams + 1e-9)
                    if score > max_new_score:
                        max_new_score = score
                        best_idx = idx
            
            # Fallback: if all candidates have 0 novelty (e.g. exact duplicates)
            if best_idx == -1:
                # Pick any from remaining
                if remaining_indices:
                    # best_idx = list(remaining_indices)[0]
                    best_idx = next(iter(remaining_indices))
                else:
                    # Should not happen if logic is correct, but for safety
                    best_idx = 0
            
            selected_indices.append(best_idx)
            remaining_indices.discard(best_idx)
            
            # Update the global set of seen n-grams with the newly selected candidate's ngrams
            global_seen_ngrams.update(candidates_ngrams[best_idx])

        return selected_indices

    def select_diverse_indices_emb(self, candidates_text: List[str], n_select: int, end_tag_str: str) -> List[int]:
        """
        Select n_select indices from candidates_text based on semantic diversity using external embedding API.
        Note: candidates_text contains the <end_token> at the end!
        """
        assert len(candidates_text) >= n_select, "Candidates text must contain at least 'n_select' elements!"
        total_candidates = len(candidates_text)

        # Batch validate
        validity_results = self._batch_validate_contents(candidates_text, end_tag_str)
        valid_indices_set = {i for i, is_valid in enumerate(validity_results) if is_valid}

        # Extract the last tag content
        contents = [self.remove_wrcot_tokens(self.extract_last_tag_content(t)) for t in candidates_text]
        
        # If not enough valid candidates, use all
        if len(valid_indices_set) < n_select:
            logging.warning(f"Not enough valid candidates ({len(valid_indices_set)}) for {n_select} selections. Using all candidates.")
            valid_indices_set = set(range(total_candidates))

        embeddings = self._get_embeddings_from_api(contents)
        
        # Diversity Selection (Greedy Max-Min)
        selected_indices = []
        # Start with the first VALID candidate
        first_idx = next(iter(valid_indices_set))
        selected_indices.append(first_idx)
        
        # sim_to_selected: [N, 1]
        sim_to_selected = embeddings @ embeddings[first_idx].T
        current_max_sims = sim_to_selected.copy() # We want to minimize this max similarity aka maximize min distance
        
        candidates_indices = valid_indices_set.copy()
        # candidates_indices.remove(first_idx)
        candidates_indices.discard(first_idx)

        # Convert to list for faster iteration
        remaining_list = list(candidates_indices)
        
        for _ in range(n_select - 1):
            if not remaining_list:
                break
                
            # We want the candidate that has the SMALLEST 'max_sim' to the existing set
            # Wait, standard MaxSum diversity is usually about maximizing sum of distances.
            # MaxMin (k-Center) is: select x s.t. min_dist(x, S) is maximized.
            # min_dist(x, S) = 1 - max_sim(x, S)
            # So we want to maximize (1 - max_sim) => minimize max_sim
            
            # current_max_sims holds the max similarity of each candidate to ANY selected
            remaining_indices = list(candidates_indices)
            
            # Find index in remaining that has the smallest max_sim
            # (i.e. it is furthest away from its nearest neighbor in S)
            
            # Extract values for remaining
            # Inefficient to index array every time if N is large, but N is small here (~K=32)
            best_next_idx = -1
            min_max_sim = 2.0 # Cosine sim is [-1, 1]
            
            for idx in remaining_indices:
                if current_max_sims[idx] < min_max_sim:
                    min_max_sim = current_max_sims[idx]
                    best_next_idx = idx
            
            if best_next_idx == -1:
                best_next_idx = remaining_indices[0] # Should not happen

            # Vectorized find minimum
            remaining_sims = current_max_sims[remaining_list]
            min_idx_in_remaining = np.argmin(remaining_sims)
            best_next_idx = remaining_list[min_idx_in_remaining]
                
            selected_indices.append(best_next_idx)
            # candidates_indices.remove(best_next_idx)
            remaining_list.remove(best_next_idx)

            # Update current_max_sims
            new_sims = embeddings @ embeddings[best_next_idx]
            # current_max_sims = np.maximum(current_max_sims, new_sims)
            np.maximum(current_max_sims, new_sims, out=current_max_sims)

        return selected_indices

    def select_most_different_from_context(self, candidates_text: List[str], context_text: List[str], end_tag_str: str, method: str = "ngram") -> int:
        """
        Select one index from candidates_text that is most different from context_text.
        """
        assert len(candidates_text) >= 1, "Candidates text must contain at least 'ONE' elements!"

        # Batch validate
        validity_results = self._batch_validate_contents(candidates_text, end_tag_str)
        valid_indices_set = {i for i, is_valid in enumerate(validity_results) if is_valid}

        # Extract content
        cand_contents = [self.remove_wrcot_tokens(self.extract_last_tag_content(t)) for t in candidates_text]
        context_contents = [self.remove_wrcot_tokens(self.extract_last_tag_content(t)) for t in context_text]
        
        if not valid_indices_set:
            logging.warning(f"No valid candidates ({len(valid_indices_set)}) for 'ONE' selections. Using all candidates.")
            valid_indices_set = set(range(len(cand_contents)))

        if method == "ngram":
            # Build context ngrams set
            context_ngrams = set()
            for ngram_set in self.get_ngrams_batch(context_contents):
                context_ngrams.update(ngram_set)
            
            # Batch compute candidate ngrams
            cand_ngrams = self.get_ngrams_batch(cand_contents)
            
            best_idx = -1
            max_new_score = -1
            
            for idx in valid_indices_set:
                new_ngrams_count = len(cand_ngrams[idx] - context_ngrams)
                
                if new_ngrams_count > max_new_score:
                    max_new_score = new_ngrams_count
                    best_idx = idx
            
            if best_idx == -1:
                best_idx = next(iter(valid_indices_set))
            return best_idx

        elif method == "emb":
            # Get embeddings for all
            all_texts = cand_contents + context_contents
            all_embeddings = self._get_embeddings_from_api(all_texts)
            
            cand_embeddings = all_embeddings[:len(cand_contents)]
            context_embeddings = all_embeddings[len(cand_contents):]
            
            sims = cand_embeddings @ context_embeddings.T # [N_cand, N_ctx]
            max_sims = np.max(sims, axis=1) # [N_cand]

            # Vectorized find minimum among valid indices
            valid_list = list(valid_indices_set)
            valid_max_sims = max_sims[valid_list]
            best_idx = valid_list[np.argmin(valid_max_sims)]
            
            return best_idx
            
        else:
            raise ValueError(f"Unknown diversity method: {method}")

    def select_diverse_indices(self, candidates_text: List[str], n_select: int, end_tag_str: str, method: str = "ngram") -> List[int]:
        """
        Select n_select indices from candidates_text based on diversity.
        """
        if method == "ngram":
            return self.select_diverse_indices_ngram(candidates_text, n_select, end_tag_str)
        elif method == "emb":
            return self.select_diverse_indices_emb(candidates_text, n_select, end_tag_str)
        else:
            raise ValueError(f"Unknown diversity method: {method}")


    def _get_embeddings_from_api(self, candidates_text: List[str]) -> np.ndarray:
        """Get embeddings from API with connection pooling and retry logic. Includes caching."""
        # Initialize cache if needed
        if not hasattr(self, 'embedding_cache'):
            self.embedding_cache = {}

        # Identify unique texts that are missing from cache
        unique_missing = []
        seen_missing = set()
        for text in candidates_text:
            if text not in self.embedding_cache and text not in seen_missing:
                unique_missing.append(text)
                seen_missing.add(text)
        
        # Fetch missing texts if any
        if unique_missing:
            # Distribute texts to available APIs
            num_apis = len(self.emb_api_urls)
            total_candidates = len(unique_missing)
            chunk_size = (total_candidates + num_apis - 1) // num_apis
            if chunk_size < 1: chunk_size = 1
            
            chunks = []
            for i in range(0, total_candidates, chunk_size):
                chunk = unique_missing[i:i + chunk_size]
                if chunk:
                    chunks.append(chunk)

            # Reuse session for connection pooling
            if not hasattr(self, '_session'):
                self._session = requests.Session()
                # Disable proxies for local connections
                self._session.trust_env = False

            def call_api(url, texts, chunk_idx):
                try:
                    response = self._session.post(
                        url, 
                        json={"texts": texts}, 
                        proxies={"http": None, "https": None},
                        timeout=30
                    )
                    response.raise_for_status()
                    return (chunk_idx, response.json()['embeddings'])
                except Exception as e:
                    logger.error(f"Failed to get embeddings from {url}: {e}")
                    return (chunk_idx, None)

            # Pre-allocate result array
            results = {}
            embed_dim = None
            
            # Check if we have any existing embeddings to guess dim
            if self.embedding_cache:
                embed_dim = len(next(iter(self.embedding_cache.values())))

            with ThreadPoolExecutor(max_workers=num_apis) as executor:
                futures = []
                for i, chunk in enumerate(chunks):
                    url = self.emb_api_urls[i % num_apis]
                    futures.append(executor.submit(call_api, url, chunk, i))
                
                for future in as_completed(futures):
                    chunk_idx, res = future.result()
                    if res is None:
                        # Fallback: random embeddings
                        logger.warning("API failure, using random embeddings for chunk")
                        if embed_dim is None:
                            embed_dim = 1024  # Default assumption
                        results[chunk_idx] = np.random.rand(len(chunks[chunk_idx]), embed_dim).tolist()
                    else:
                        results[chunk_idx] = res
                        if embed_dim is None and res:
                            embed_dim = len(res[0])
            
            # Update cache with new results
            for i, chunk in enumerate(chunks):
                if i in results:
                    embeddings_list = results[i]
                    for text, emb in zip(chunk, embeddings_list):
                        self.embedding_cache[text] = np.array(emb, dtype=np.float32)
        
        # Reconstruct in order from cache
        embeddings = []
        for text in candidates_text:
            embeddings.append(self.embedding_cache[text])
        
        return np.array(embeddings, dtype=np.float32)

    def _make_input_batch(self, inputs_list, meta_info_base):
        # inputs_list: list of input_ids tensors (1D)
        max_len = max([t.size(0) for t in inputs_list])
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        # Optimization: Pre-allocate tensor
        batch_size = len(inputs_list)
        b_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
        
        for i, t in enumerate(inputs_list):
            cur_len = t.size(0)
            # Left padding
            b_ids[i, max_len-cur_len:] = t
        
        batch_dict = TensorDict({
            'input_ids': b_ids,
        }, batch_size=b_ids.size(0))
        
        dp = DataProto(batch=batch_dict)
        dp.meta_info = meta_info_base.copy()
        return dp

    def update_candidates_with_diversity(self, is_first_step, output_batch, expansion_inputs, current_candidates, mapping_info, end_tag, gen_n, group_size):
        out_ids = output_batch # only segment
        
        batch_size = len(current_candidates)
        end_tag_str = end_tag
        
        curr_out_idx = 0
        new_candidates_lists = [[] for _ in range(batch_size)]
        
        for i, mapping in enumerate(mapping_info):
            b_idx = mapping['b_idx']
            parent_seq = expansion_inputs[i]
            
            # Collect all generated sequences for this specific parent candidate
            for _ in range(gen_n):
                child_response = out_ids[curr_out_idx]
                curr_out_idx += 1
                
                # Prepend parent
                if not isinstance(child_response, torch.Tensor):
                    child_response = torch.tensor(child_response, dtype=torch.long)
                
                full_child = torch.cat([parent_seq, child_response])
                new_candidates_lists[b_idx].append(full_child)

        ### Now select diversity for each original batch item
        next_gen_candidates = []
        
        for b_idx in range(batch_size):
            candidates_pool = new_candidates_lists[b_idx]
            pool_texts = self.tokenizer.batch_decode(candidates_pool, skip_special_tokens=False)
            # ['xxxxx</end_token>', ...] (keep the <end_token> to distinguish the degenerated ones)
            
            selected_cands = []
            
            if is_first_step:
                selected_indices = self.select_diverse_indices(pool_texts, group_size, end_tag_str, self.diversity_metric)
                for s_idx in selected_indices:
                    selected_cands.append(candidates_pool[s_idx].clone().detach())
            else:
                # Split into chunks (one per parent)
                chunks = []
                chunk_texts = []
                for i in range(0, len(candidates_pool), gen_n):
                    chunks.append(candidates_pool[i:i+gen_n])
                    chunk_texts.append(pool_texts[i:i+gen_n])
                
                if len(chunks) != group_size:
                    logger.warning(f"Expected {group_size} chunks but got {len(chunks)}. Fallback to original logic.")
                    selected_indices = self.select_diverse_indices(pool_texts, group_size, end_tag_str, self.diversity_metric)
                    for s_idx in selected_indices:
                        selected_cands.append(candidates_pool[s_idx].clone().detach())
                else:
                    for i in range(group_size):
                        current_chunk_texts = chunk_texts[i]
                        
                        # Context is all other texts
                        context_texts = []
                        for j in range(group_size):
                            if i != j:
                                context_texts.extend(chunk_texts[j])
                        
                        best_idx = self.select_most_different_from_context(current_chunk_texts, context_texts, end_tag_str, self.diversity_metric)

                        selected_cands.append(chunks[i][best_idx].clone().detach())
            
            next_gen_candidates.append(selected_cands)
            
        return next_gen_candidates

    def finalize_rollout(self, actor_rollout_wg, current_candidates, gen_batch, initial_prompts):
        batch_size = len(current_candidates)
        # Flatten candidates
        final_inputs = []
        all_gen_seqs = []
        gen_seq_lens = []
        
        # Batch decode all candidates at once
        all_cand_ids = []
        for b_idx in range(batch_size):
            for cand_ids in current_candidates[b_idx]:
                all_cand_ids.append(cand_ids)
                final_inputs.append(cand_ids)
        
        # Single batch decode call
        all_decode_strs = self.tokenizer.batch_decode(all_cand_ids, skip_special_tokens=False)
        
        for _decode_str in all_decode_strs:
            # ### For Qwen-4B
            # _gen_seq = _decode_str.split("<|im_start|>assistant\n")[-1]
            # ### For llama-3.2-inst
            # _gen_seq = _gen_seq.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1]
            if "<|im_start|>assistant\n" in _decode_str:
                # Qwen format
                _gen_seq = _decode_str.split("<|im_start|>assistant\n")[-1]
            elif "<|start_header_id|>assistant<|end_header_id|>\n\n" in _decode_str:
                # Llama format
                _gen_seq = _decode_str.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1]
            else:
                # Fallback or error
                _gen_seq = _decode_str
            all_gen_seqs.append(_gen_seq)
            gen_seq_lens.append(len(self.tokenizer.encode(_gen_seq, add_special_tokens=False)))

        logging.info(f"Processing final {len(final_inputs)} branching rollout...")
        final_batch = self._make_input_batch(final_inputs, gen_batch.meta_info)
        gen_kwargs = {
            "finalize_branching": True,
            "top_p": self.config.actor_rollout_ref.rollout.top_p,
            "temperature": self.config.actor_rollout_ref.rollout.temperature,
            "n": 1,
            "min_tokens": 1,
            "max_tokens": self.config.data.max_response_length - min(gen_seq_lens),
            # This is to meet the max_tokens requirement
        }
        
        final_batch.meta_info['do_sample'] = True
        final_batch.meta_info.update(gen_kwargs)

        # Call vLLM
        output_batch = actor_rollout_wg.generate_sequences(final_batch)
        # type(output_batch.non_tensor_batch['final_branching_res']): <class 'numpy.ndarray'>
        final_branching_res = output_batch.non_tensor_batch['final_branching_res']
        final_branching_res = filter_oov_tokens(final_branching_res, len(self.tokenizer))

        assert len(final_branching_res) == len(all_gen_seqs), "Batchsize mismatch in final branching results."

        # Batch decode final results
        final_decoded = self.tokenizer.batch_decode(final_branching_res, skip_special_tokens=False)
        full_res = [all_gen_seqs[i] + final_decoded[i] for i in range(len(final_branching_res))]

        # Prepare outputs
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        
        # 1. Expand Prompts (Left Padded to max_prompt_length)
        old_prompts = gen_batch.batch['input_ids'].to('cpu') # [B, L]
        
        # Ensure max_prompt_length
        target_prompt_len = self.config.data.max_prompt_length
        
        # repeats = [len(c) for c in current_candidates]
        repeats = self.config.actor_rollout_ref.rollout.n
        prompts = torch.repeat_interleave(old_prompts, repeats=torch.tensor(repeats), dim=0)
        
        # 2. Process Responses (Right Padded to max_response_length)
        target_response_len = self.config.data.max_response_length
        
        # Batch encode responses
        responses_encoded = [self.tokenizer.encode(res_str, add_special_tokens=False) for res_str in full_res]
        
        responses = torch.full((len(responses_encoded), target_response_len), pad_token_id, dtype=torch.long)
        for i, res_ids in enumerate(responses_encoded):
            if len(res_ids) > target_response_len:
                res_ids = res_ids[:target_response_len]
            responses[i, :len(res_ids)] = torch.tensor(res_ids, dtype=torch.long)
        
        # 3. Concatenate (Prompt + Response = [256, 4096])
        input_ids = torch.cat([prompts, responses], dim=1)
        
        # 4. Attention Mask & Position IDs (vectorized)
        attention_mask = (input_ids != pad_token_id).long()
        position_ids = torch.cumsum(attention_mask, dim=1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        
        # 5. Non-Tensor Batch
        batch_size_final = input_ids.size(0)
        non_tensor_batch = {
            'tools_kwargs': np.array([{} for _ in range(batch_size_final)], dtype=object),
            'interaction_kwargs': np.array([{} for _ in range(batch_size_final)], dtype=object)
        }

        # 6. Construct DataProto
        batch = TensorDict({
            'prompts': prompts,
            'responses': responses,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids
        }, batch_size=batch_size_final)
        
        dp = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
        if hasattr(output_batch, 'meta_info') and 'timing' in output_batch.meta_info:
            dp.meta_info = {'timing': output_batch.meta_info['timing']}

        # Clean up caches
        if hasattr(self, 'embedding_cache'):
            self.embedding_cache = {}
        if hasattr(self, 'ngram_cache'):
            self.ngram_cache = {}
        logging.warning("Diversity caches cleaned up.")
        
        return dp

def filter_oov_tokens(token_sequences, vocab_size):
    """    
    Args:
        token_sequences: numpy array，shape=(batch_size,)，dtype=object，includes token tuple/list
        vocab_size: int, tokenizer vocab size
    
    Returns:
        numpy array，the same shape and dtype, while contains non-OOV tokens
    """    
    # Use list comprehension for better performance
    result = np.array(
        [tuple(token for token in seq if token < vocab_size) for seq in token_sequences],
        dtype=object
    )
    return result


def sync_rollout_with_branching(actor_rollout_wg, gen_batch, config, tokenizer, branching_strategy, group_size):
    """
    Perform synchronized rollout with branching strategy.
    
    Args:
        actor_rollout_wg: The rollout worker group.
        gen_batch: Initial batch containing prompts.
        config: Configuration object.
        tokenizer: Tokenizer.
        branching_strategy: Instance of BranchingRolloutStrategy.
        group_size: The target number of rollouts per prompt (n).
        
    Returns:
        final_gen_batch: DataProto containing the generated responses.
    """
    
    k_branches = branching_strategy.k_branches
    
    # Extract initial input_ids (tensor)
    input_ids = gen_batch.batch['input_ids'].cpu() # (B, L)
    attention_mask = gen_batch.batch['attention_mask'].cpu()
    
    batch_size = input_ids.size(0)
    
    # Store initial candidates as list of tensors
    # We should strip left padding for easier concatenation
    current_candidates = []
    initial_prompts = [] # Store initial prompts for final reconstruction
    
    for i in range(batch_size):
        # find start of real sequence
        mask = attention_mask[i]
        valid_indices = torch.nonzero(mask == 1, as_tuple=True)[0]
        if len(valid_indices) > 0:
            start_idx = valid_indices[0].item()
            real_ids = input_ids[i, start_idx:]
        else:
            real_ids = input_ids[i] # keeping as is if all 0 (weird)
        current_candidates.append([real_ids])
        initial_prompts.append(real_ids)
    
    # Detect language for each batch item
    batch_langs = []
    for p_ids in initial_prompts:
        text = tokenizer.decode(p_ids)
        # Simple heuristic: if any Chinese char, assume ZH, else EN
        if any(u'\u4e00' <= c <= u'\u9fff' for c in text):
            batch_langs.append('zh')
        else:
            batch_langs.append('en')

    num_segments = len(branching_strategy.get_segments('zh'))

    logging.info(f"Branching samples for thinking segment...")
    for seg_idx in range(num_segments):
        ### 1. Expand
        expansion_inputs = [] 
        new_mapping_info = [] # flattened list of what each row corresponds to
        # We need to flatten the candidates to a single batch         
        for b_idx in range(batch_size):
            cands = current_candidates[b_idx]
            num_cands = len(cands)
            n_samples = max(1, k_branches // num_cands)
            
            # Determine append tensor for this item
            lang = batch_langs[b_idx]
            prefix, start_tag, end_tag = branching_strategy.get_segments(lang)[seg_idx]
            append_str = prefix + start_tag
            append_ids = tokenizer.encode(append_str, add_special_tokens=False)
            append_tensor = torch.tensor(append_ids, dtype=torch.long)
            
            for c_idx, curr_ids in enumerate(cands):
                # Append padding + start tag
                new_ids = torch.cat([curr_ids, append_tensor])
                expansion_inputs.append(new_ids)
                new_mapping_info.append({'b_idx': b_idx, 'n_samples': n_samples})

        # Pad and Generate
        if not expansion_inputs:
            break
        
        step_batch = branching_strategy._make_input_batch(expansion_inputs, gen_batch.meta_info)
        
        # end_tag is shared across languages, use one to get stop_token_ids
        _, _, end_tag_ref = branching_strategy.get_segments('zh')[seg_idx]
        stop_token_ids = tokenizer.encode(end_tag_ref, add_special_tokens=False)
        
        gen_kwargs = {
            "branching": True,
            "n": new_mapping_info[0]['n_samples'], 
            "stop_token_ids": stop_token_ids,
            "top_p": branching_strategy.config.actor_rollout_ref.rollout.top_p,
            "temperature": config.trainer.get("branching_temperature", branching_strategy.config.actor_rollout_ref.rollout.temperature),
            "min_tokens": 1,
            "max_tokens": 192, # 128+64
        }
        
        step_batch.meta_info['do_sample'] = True
        step_batch.meta_info['branch_step'] = seg_idx
        step_batch.meta_info.update(gen_kwargs)
        
        # Call vLLM
        output_batch = actor_rollout_wg.generate_sequences(step_batch)
        filtered_tokens = filter_oov_tokens(
            output_batch.non_tensor_batch[f'branch_step{seg_idx}_res'], 
            len(tokenizer)
        )
        output_batch.non_tensor_batch[f'branch_step{seg_idx}_res'] = filtered_tokens
        
        ### 2. Process Output & Select Diversity
        current_candidates = branching_strategy.update_candidates_with_diversity(
            is_first_step=(seg_idx == 0),
            output_batch=output_batch.non_tensor_batch[f'branch_step{seg_idx}_res'],
            expansion_inputs=expansion_inputs,
            current_candidates=current_candidates,
            mapping_info=new_mapping_info,
            end_tag=end_tag_ref,
            gen_n=gen_kwargs['n'],
            group_size=group_size
        )
        
        has_end_tags = [sum([x.endswith(end_tag_ref) for x in tokenizer.batch_decode(current_candidates[i])]) for i in range(batch_size)]
        avg_has_end_tags = sum(has_end_tags) / batch_size
        logging.warning(f"Segment {seg_idx}. Average has_end_tags: {avg_has_end_tags} / {config.actor_rollout_ref.rollout.n}")

    # Final generation until EOS
    return branching_strategy.finalize_rollout(
        actor_rollout_wg=actor_rollout_wg,
        current_candidates=current_candidates,
        gen_batch=gen_batch,
        initial_prompts=initial_prompts
    )






