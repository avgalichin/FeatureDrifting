from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from statistics import mean

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

from tqdm import tqdm


@dataclass
class PromptPair:
    """Data class for chat/base prompt pairs"""
    chat_prompt: str
    base_prompt: str
    expected_response: str  # Ground truth response for alignment


@dataclass
class ConversationTurn:
    """Data class for conversation turns"""
    content: str
    role: str


@dataclass
class ModelOutputs:
    """Data class for model outputs"""
    logits: torch.Tensor  # Shape: (seq_len, vocab_size)
    activations: torch.Tensor  # Shape: (seq_len, d)
    sae_reconstruction: torch.Tensor  # Shape: (seq_len, d)
    sae_features: torch.Tensor  # Shape: (seq_len, n_features) - active features
    tokens: List[int]
    response_start_idx: int
    response_end_idx: int


@dataclass
class FeaturePair:
    """Activations of Sparse Coder features."""
    chat_features: torch.Tensor  # Shape: (seq_len, n_features)
    base_features: torch.Tensor  # Shape: (seq_len, n_features)


class DriftingAnalyzer:
    def __init__(self, 
                 chat_model_name: str, 
                 base_model_name: str,
                 sae_path: str,
                 sae_id: str,
                 act_layer: int,
                 k: int = 100,
                 ap_at_k: int = 30,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 use_chat_format: bool = True):
        self.device = device
        self.chat_model_name = chat_model_name
        self.base_model_name = base_model_name
        self.act_layer = act_layer
        self.k = k
        self.ap_at_k = ap_at_k
        self.use_chat_format = use_chat_format

        print(f"Loading chat model: {chat_model_name}")
        self.chat_tokenizer = AutoTokenizer.from_pretrained(chat_model_name)
        self.chat_model = AutoModelForCausalLM.from_pretrained(
            chat_model_name, 
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        
        print(f"Loading base model: {base_model_name}")
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16, 
            device_map=device
        )

        print(f"Loading SAE: {sae_path}")
        self.chat_sae = SAE.from_pretrained(sae_path, sae_id, device)
        self.base_sae = SAE.from_pretrained(sae_path, sae_id, device)
        self.d_in = self.chat_sae.cfg.d_in

        self.chat_scaling_factor = 1.0
        self.base_scaling_factor = 1.0

        self.chat_model.eval()
        self.base_model.eval()

        if self.chat_tokenizer.pad_token is None:
            self.chat_tokenizer.pad_token = self.chat_tokenizer.eos_token
        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
    
    def create_chat_prompt(self, conversation: List[ConversationTurn]) -> str:
        """
        Create chat-formatted prompt from conversation (single interaction).
        """
        if "gemma" in self.chat_model_name.lower():
            # Gemma chat format
            prompt = ""
            for turn in conversation:
                if turn.role == "user":
                    prompt += f"<start_of_turn>user\n{turn.content}<end_of_turn>\n"
                elif turn.role == "assistant":
                    prompt += f"<start_of_turn>model\n"
                    break  # Stop at assistant turn for generation
        elif "llama" in self.chat_model_name.lower():
            prompt = "<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|>"
            for turn in conversation:
                if turn.role == "user":
                    prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{turn.content}<|eot_id|>"
                elif turn.role == "assistant":
                    prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                    break  # Stop at assistant turn for generation
        else:
            prompt = ""
            for turn in conversation:
                if turn.role == "user":
                    prompt += f"<|user|>\n{turn.content}\n"
                elif turn.role == "assistant":
                    prompt += f"<|assistant|>\n"
                    break
        return prompt
    

    def create_base_prompt(self, conversation: List[ConversationTurn]) -> str:
        """
        Create base model prompt from conversation (single interaction).
        """
        if "llama" in self.base_model_name.lower():
            prompt = ""
            for turn in conversation:
                if turn.role == "user":
                    prompt += f"\n\n\nUser: ####\n\n{turn.content} ####"
                elif turn.role == "assistant":
                    prompt += "\n\n\nAssistant: ####\n\n"
        else: 
            prompt = ""
            for turn in conversation:
                if turn.role == "user":
                    prompt += f"User: {turn.content}\n"
                elif turn.role == "assistant":
                    prompt += "Assistant: "
                    break  # Stop at assistant turn for generation
        return prompt
    

    def conversation_to_prompt_pair(self, conversation: List[Dict[str, str]]) -> PromptPair:
        """
        Convert conversation format to PromptPair
        
        Args:
            conversation: List of dicts with 'content' and 'role' keys
            
        Returns:
            PromptPair object
        """
        turns = [ConversationTurn(content=turn["content"], role=turn["role"]) 
                for turn in conversation]
        
        assistant_response = None
        user_turns = []  # This will be 1-element list, actually
        
        for turn in turns:
            if turn.role == "user":
                user_turns.append(turn)
            elif turn.role == "assistant" and assistant_response is None:
                assistant_response = turn.content
                break
        
        if assistant_response is None:
            raise ValueError("No assistant response found in conversation")
        
        if not self.use_chat_format:
            # NOTE: This is a simple workaround to process continuous texts format.
            #       We do not use chat template and leave the `user` prompt empty.
            return PromptPair(
                chat_prompt='',
                base_prompt='',
                expected_response=assistant_response
            )

        # Create prompts (only include turns up to first assistant response)
        relevant_turns = user_turns + [ConversationTurn(content="", role="assistant")]
        
        chat_prompt = self.create_chat_prompt(relevant_turns)
        base_prompt = self.create_base_prompt(relevant_turns)
        
        return PromptPair(
            chat_prompt=chat_prompt,
            base_prompt=base_prompt,
            expected_response=assistant_response
        )
    
    def extract_response_tokens(self, 
                                full_text: str, 
                                response_text: str, 
                                tokenizer: AutoTokenizer) -> Tuple[List[int], int, int]:
        """
        Extract token indices for the assistant response part
        
        Args:
            full_text: Complete prompt + response text
            response_text: Just the assistant response part
            tokenizer: Tokenizer to use
            
        Returns:
            Tuple of (all_tokens, response_start_idx, response_end_idx)
        """
        full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
        response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
        
        response_start_idx = None
        for i in range(len(full_tokens) - len(response_tokens) + 1):
            if full_tokens[i:i + len(response_tokens)] == response_tokens:
                response_start_idx = i
                break
        
        if response_start_idx is None:
            response_start_idx = len(full_tokens) - len(response_tokens)
        
        response_end_idx = response_start_idx + len(response_tokens)
        
        return full_tokens, response_start_idx, response_end_idx
    
    def get_model_outputs(self, 
                          prompt: str, 
                          response: str, 
                          model: AutoModelForCausalLM, 
                          sae: SAE,
                          tokenizer: AutoTokenizer,
                          scaling_factor: float = 1.0) -> ModelOutputs:
        """
        Get model logits for a prompt + response pair
        
        Args:
            prompt: Input prompt
            response: Expected response
            model: Model to use for inference
            tokenizer: Corresponding tokenizer
            
        Returns:
            ModelOutputs containing logits and token information
        """
        def gather_target_act_hook(module, inputs, outputs):
            nonlocal activations, sae_reconstruction, sae_features

            acts = outputs[0].squeeze(0).clone() * scaling_factor
            activations = acts.clone()

            sae_acts = sae.encode(acts)
            sae_out = sae.decode(sae_acts)

            sae_reconstruction = sae_out
            sae_features = sae_acts

            return outputs

        full_text = prompt + response
        
        # Extract response token boundaries
        tokens, resp_start, resp_end = self.extract_response_tokens(
            full_text, response, tokenizer
        )
        
        # Tokenize for model input
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = inputs.input_ids.to(self.device)
        
        # Get logits
        with torch.no_grad():
            activations = None
            sae_reconstruction = None
            sae_features = None

            handle = model.base_model.get_submodule(f"layers.{self.act_layer}") \
                .register_forward_hook(gather_target_act_hook)
            try:
                outputs = model(input_ids)
            finally:
                handle.remove()

        logits = outputs.logits.squeeze(0)  # Remove batch dimension
        
        return ModelOutputs(
            logits=logits,
            activations=activations,
            sae_reconstruction=sae_reconstruction,
            sae_features=sae_features,
            tokens=tokens,
            response_start_idx=resp_start,
            response_end_idx=resp_end
        )

    def calculate_feature_metrics(self, 
                                  chat_features: torch.Tensor,
                                  base_features: torch.Tensor) -> Dict[str, float]:
        """Calculate feature-based metrics for a single token"""
        # Get top-k features
        chat_nonzero = chat_features.count_nonzero(dim=0).item()
        base_nonzero = base_features.count_nonzero(dim=0).item()
        k = int(min(chat_nonzero, base_nonzero, self.k))

        chat_indices = chat_features.topk(k=k, dim=0).indices.cpu().tolist()
        base_indices = base_features.topk(k=k, dim=0).indices.cpu().tolist()

        # Feature intersection ratio
        intersection = len(set(chat_indices) & set(base_indices))
        union = len(set(chat_indices) | set(base_indices))
        feature_intersection_ratio = intersection / union if union > 0 else 0.0

        # Feature exact-match ratio
        feature_match_ratio = sum([c == b for c, b in zip(chat_indices, base_indices)]) / len(chat_indices)

        # Average precision @ k
        relevant_indices = set(chat_indices[:self.ap_at_k])
        num_relevant_found = 0
        precision_sum = 0.
        for i, pred_idx in enumerate(base_indices[:self.ap_at_k]):
            if pred_idx in relevant_indices:
                num_relevant_found += 1
                precision_at_i = num_relevant_found / (i + 1)
                precision_sum += precision_at_i
        average_precision = precision_sum / self.ap_at_k

        return {
            'feature_intersection_ratio': feature_intersection_ratio,
            'feature_match_ratio': feature_match_ratio,
            'average_precision': average_precision,
            'chat_l0': chat_nonzero,
            'base_l0': base_nonzero
        }
    
    def process_prompt_pair(self, prompt_pair: PromptPair) -> Tuple[Dict[str, List[float]], FeaturePair]:
        """Process a single prompt pair and return metrics"""
        # Get outputs from both models
        chat_outputs = self.get_model_outputs(
            prompt_pair.chat_prompt, 
            prompt_pair.expected_response,
            self.chat_model, 
            self.chat_sae,
            self.chat_tokenizer,
            self.chat_scaling_factor
        )
        
        base_outputs = self.get_model_outputs(
            prompt_pair.base_prompt,
            prompt_pair.expected_response, 
            self.base_model,
            self.base_sae,
            self.base_tokenizer,
            self.base_scaling_factor
        )
        
        # Extract only response tokens
        chat_acts = chat_outputs.activations[chat_outputs.response_start_idx:chat_outputs.response_end_idx]
        chat_recon = chat_outputs.sae_reconstruction[chat_outputs.response_start_idx:chat_outputs.response_end_idx]
        chat_feats = chat_outputs.sae_features[chat_outputs.response_start_idx:chat_outputs.response_end_idx]
        
        base_acts = base_outputs.activations[base_outputs.response_start_idx:base_outputs.response_end_idx]
        base_recon = base_outputs.sae_reconstruction[base_outputs.response_start_idx:base_outputs.response_end_idx]
        base_feats = base_outputs.sae_features[base_outputs.response_start_idx:base_outputs.response_end_idx]
        
        # Ensure same length
        min_len = min(len(chat_acts), len(base_acts))
        
        # Calculate metrics for each token
        metrics = {
            'feature_intersection_ratio': [],
            'feature_match_ratio': [],
            'average_precision': [],
            'chat_fvu': [],
            'base_fvu': [],
            'mse': [],
            'chat_l0': [],
            'base_l0': []
        }
        
        for i in range(min_len):
            # Feature metrics
            feat_metrics = self.calculate_feature_metrics(chat_feats[i], base_feats[i])
            metrics['feature_intersection_ratio'].append(feat_metrics['feature_intersection_ratio'])
            metrics['feature_match_ratio'].append(feat_metrics['feature_match_ratio'])
            metrics['average_precision'].append(feat_metrics['average_precision'])
            
            # FVU
            metrics['chat_fvu'].append(self._calculate_fvu(chat_acts[i], chat_recon[i]))
            metrics['base_fvu'].append(self._calculate_fvu(base_acts[i], base_recon[i]))

            # MSE between chat and base activations
            metrics['mse'].append(self.calculate_mse(chat_acts[i], base_acts[i]))

            # L0
            metrics['chat_l0'].append(feat_metrics['chat_l0'])
            metrics['base_l0'].append(feat_metrics['base_l0'])
        
        feature_pair = FeaturePair(
            chat_features=chat_feats[:min_len].cpu(),
            base_features=base_feats[:min_len].cpu()
        )
        
        return metrics, feature_pair
    
    def estimate_scaling_factor(
        self, 
        conversations: List[List[Dict[str, str]]],
        n_samples_to_estimate: int = int(1e3),
        model_type: str = 'base'
    ) -> None:
        setattr(self, f"{model_type}_scaling_factor", 1.0)

        model = getattr(self, f"{model_type}_model")
        sae = getattr(self, f"{model_type}_sae")
        tokenizer = getattr(self, f"{model_type}_tokenizer")

        norms = []
        for conversation in tqdm(conversations[:n_samples_to_estimate], desc="Estimating norm scaling factor"):
            prompt_pair = self.conversation_to_prompt_pair(conversation)
            outputs = self.get_model_outputs(
                getattr(prompt_pair, f"{model_type}_prompt"),
                prompt_pair.expected_response,
                model,
                sae,
                tokenizer,
                scaling_factor=1.0
            )
            activations = outputs.activations[outputs.response_start_idx:outputs.response_end_idx]
            norms.append(activations.norm(dim=1).mean().item())
        mean_norm = mean(norms)

        scaling_factor = (self.d_in**0.5) / mean_norm
        setattr(self, f"{model_type}_scaling_factor", scaling_factor)

    def analyze_conversations(self, conversations: List[List[Dict[str, str]]]) -> Dict[str, Any]:
        """
        Analyze conversations and compute metrics
        
        Returns:
            Dictionary containing:
            - feature_diff: diff-in-means for base and chat model activations
            - feature_intersection_ratio: intersection ratio for each token
            - feature_match_ratio: exact matching ratio for each token
            - average_precision: ap @ k for each token
            - chat_fvu: FVU for each activation of the chat model
            - base_fvu: FVU for each activation of the base model
            - mse: MSE between chat and base activations
            - chat_l0: l0 for each reconstruction of the chat activation
            - base_l0: l0 for each reconstruction of the base activation
        """
        # Convert conversations to prompt pairs
        prompt_pairs = []
        for conversation in conversations:
            try:
                prompt_pair = self.conversation_to_prompt_pair(conversation)
                prompt_pairs.append(prompt_pair)
            except Exception as e:
                continue
        
        # Initialize results
        feature_diff = []
        num_updates = []
        all_metrics = {
            'feature_intersection_ratio': [],
            'feature_match_ratio': [],
            'average_precision': [],
            'chat_fvu': [],
            'base_fvu': [],
            'mse': [],
            'chat_l0': [],
            'base_l0': []
        }
        
        # Process each prompt pair
        for prompt_pair in tqdm(prompt_pairs, desc="Processing pairs"):
            try:
                metrics, feature_pair = self.process_prompt_pair(prompt_pair)
                
                # Update feature diff (per-position)
                self._update_diff_in_means(feature_diff, num_updates, feature_pair)
                
                # Append metrics (flat lists)
                for key in all_metrics:
                    all_metrics[key].extend(metrics[key])
                    
            except Exception as e:
                print(f"Error processing pair: {e}")
                continue
        
        # Prepare final results
        results = {
            'feature_diff': feature_diff,  # List of tensors, one per position
            **all_metrics  # Flat lists of metrics
        }
        
        return results

    @staticmethod
    def _calculate_fvu(activations: torch.Tensor, reconstruction: torch.Tensor) -> float:
        """Calculate FVU for a single token activation"""
        return (
            (activations - reconstruction).pow(2).sum()
            /
            (activations - activations.mean()).pow(2).sum()
        ).item()
    
    def calculate_mse(self, chat_activations: torch.Tensor, base_activations: torch.Tensor) -> float:
        """Calculate MSE between chat and base activations for a single token"""
        return F.mse_loss(chat_activations, base_activations).item()
    
    @staticmethod
    def _update_diff_in_means(feature_diff: List[torch.Tensor], 
                              num_updates: List[int],
                              feature_pair: FeaturePair) -> None:
        """Update running mean of feature differences"""
        diff = feature_pair.chat_features - feature_pair.base_features
        seq_len = diff.shape[0]
        
        # Extend lists if needed
        while len(feature_diff) < seq_len:
            feature_diff.append(torch.zeros_like(diff[0]))
            num_updates.append(0)
        
        # Update running mean for each position
        for i in range(seq_len):
            num_updates[i] += 1
            alpha = 1 - 1.0 / num_updates[i]
            feature_diff[i] = alpha * feature_diff[i] + (1 - alpha) * diff[i]
