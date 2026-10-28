"""LLM loader for Llama-3.2-3B-Instruct via Hugging Face Transformers.

Loads the model and tokenizer, and provides text generation capabilities.
"""

from __future__ import annotations

import os
import torch
from typing import List, Optional, Union

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers.utils import logging as hf_logging

# Reduce Transformers verbosity
hf_logging.set_verbosity_error()


class LLMLoader:
    """Loader for Llama-3.2-3B-Instruct model."""

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.2-3B-Instruct",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        trust_remote_code: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
    ):
        """Initialize the LLM loader.

        Args:
            model_id: Hugging Face model identifier.
            device: Device to use ('cuda', 'cpu', or None for auto).
            torch_dtype: Torch dtype for model weights.
            load_in_8bit: Whether to load model in 8-bit quantization.
            load_in_4bit: Whether to load model in 4-bit quantization.
            trust_remote_code: Whether to trust remote code.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling threshold.
            do_sample: Whether to use sampling.
            repetition_penalty: Penalty for repetition.
        """
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.repetition_penalty = repetition_penalty

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Set torch dtype
        if torch_dtype is None:
            if self.device == "cuda":
                self.torch_dtype = torch.float16
            else:
                self.torch_dtype = torch.float32
        else:
            self.torch_dtype = torch_dtype

        # Validate quantization arguments
        if load_in_8bit and load_in_4bit:
            raise ValueError("Cannot specify both load_in_8bit and load_in_4bit")

        # Load model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
                use_fast=True,
            )
            # Llama tokenizer may not have a pad token by default
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=trust_remote_code,
            )
            if self.device == "cpu":
                self.model.to(self.device)

        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_id}: {e}") from e

        # Set up generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=self.do_sample,
            repetition_penalty=self.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
        repetition_penalty: Optional[float] = None,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input prompt string.
            max_new_tokens: Override default max new tokens.
            temperature: Override default temperature.
            top_p: Override default top_p.
            do_sample: Override default do_sample.
            repetition_penalty: Override default repetition_penalty.

        Returns:
            Generated text string.
        """
        # Use provided values or defaults
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        do_sample = do_sample if do_sample is not None else self.do_sample
        repetition_penalty = (
            repetition_penalty if repetition_penalty is not None else self.repetition_penalty
        )

        # Benchmark adapters explicitly opt into full-prompt validation. Keep
        # the legacy policy for callers that have not selected this protocol.
        strict_limit = getattr(self, "strict_max_input_tokens", None)
        if strict_limit is None:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False)
            count = inputs["input_ids"].shape[1]
            context = self.model.config.max_position_embeddings
            if count > strict_limit or count + max_new_tokens > context:
                raise ValueError(
                    f"full prompt has {count} tokens; input limit={strict_limit}, "
                    f"max_new_tokens={max_new_tokens}, model context={context}; no truncation allowed")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode output (skip input tokens)
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )

        return generated_text.strip()

    def batch_generate(
        self,
        prompts: List[str],
        **kwargs,
    ) -> List[str]:
        """Generate text for multiple prompts.

        Args:
            prompts: List of input prompts.
            **kwargs: Generation parameters passed to generate.

        Returns:
            List of generated text strings.
        """
        return [self.generate(prompt, **kwargs) for prompt in prompts]


if __name__ == "__main__":
    # Simple test (will fail if model not available, but we can check instantiation)
    try:
        llm = LLMLoader()
        print(f"LLM loader initialized for {llm.model_id} on {llm.device}")
        print(f"Model dtype: {llm.model.dtype}")
    except Exception as e:
        print(f"Failed to initialize LLM loader: {e}")
