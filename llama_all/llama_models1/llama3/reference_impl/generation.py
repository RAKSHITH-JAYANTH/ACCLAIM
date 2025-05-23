# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from termcolor import cprint

from ..api.args import ModelArgs
from ..api.chat_format import ChatFormat, LLMInput
from ..api.datatypes import RawContent, RawMessage, StopReason, ToolPromptFormat
from ..api.tokenizer import Tokenizer
from .model import Transformer

#import intel_extension_for_pytorch as ipex

# import openvino.torch

#torch.set_num_threads(16)

@dataclass
class CompletionPrediction:
    generation: str
    decoded_tokens: Optional[List[str]] = None
    logprobs: Optional[List[List[float]]] = None

@dataclass
class ChatPrediction:
    generation: RawMessage
    decoded_tokens: Optional[List[str]] = None
    logprobs: Optional[List[List[float]]] = None

@dataclass
class TokenResult:
    token: int
    text: str
    logprobs: Optional[List[float]] = None

class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
        tokenizer_path: Optional[str] = None,
        seed: int = 1,
    ):

        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = 1

        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        torch.manual_seed(seed)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()

        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        ckpt_path = checkpoints[0] # [get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        if tokenizer_path:
            tokenizer = Tokenizer(model_path=tokenizer_path)
        else:
            tokenizer = Tokenizer.get_instance()

        assert model_args.vocab_size == tokenizer.n_words
        if model_args.vision_chunk_size > 0:
            from .multimodal.model import CrossAttentionTransformer

            model = CrossAttentionTransformer(model_args)
            model.setup_cache(model_args.max_batch_size, torch.bfloat16)
        else:
            model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=True)
        #model = model.to(torch.float16)
        #model = model.to('cpu')
        #model = ipex.optimize(model)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return Llama(model, tokenizer, model_args)

    def __init__(self, model: Transformer, tokenizer: Tokenizer, args: ModelArgs):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.formatter = ChatFormat(tokenizer)

    

    
    @torch.inference_mode()
    def generate(
        self,
        model_input: LLMInput,
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
        print_model_input: bool = False,
    ) -> Generator:
        params = self.model.params

        # import pdb
        # pdb.set_trace()
        start_time = time.time()
        #print(f"Loaded in {time.time() - start_time:.2f} seconds")
        if print_model_input:
            tokens_to_print = [
                self.formatter.vision_token if t == 128256 else t
                for t in model_input.tokens
            ]
            cprint(
                "Input to model:\n" + self.tokenizer.decode(tokens_to_print) + "\n",
                "red",
            )
        prompt_tokens = [model_input.tokens]

        bsz = 1
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        #print(min_prompt_len)

        if max_prompt_len >= params.max_seq_len:
            # cprint(
            #     f"Out of token budget {max_prompt_len} vs {params.max_seq_len}", "red"
            # )
            # return
            prompt_tokens[0] = prompt_tokens[0][: params.max_seq_len-10]


        total_len = min(max_gen_len + max_prompt_len, params.max_seq_len)
        print("total_len", total_len)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        print(min_prompt_len)

        is_vision = not isinstance(self.model, Transformer)
        if is_vision:
            images = model_input.vision.images if model_input.vision is not None else []
            mask = model_input.vision.mask if model_input.vision is not None else []

            xattn_caches, cross_attention_masks, full_text_row_masked_out_mask = (
                self.model.compute_vision_tokens_masks(
                    batch_images=[images],
                    batch_masks=[mask],
                    total_len=total_len,
                )
            )

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cpu")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cpu")
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float) # XXXXXXXXXXX

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cpu")
        input_text_mask = tokens != pad_id

        if echo:
            for i, t in enumerate(model_input.tokens):
                yield TokenResult(
                    token=t,
                    text=self.tokenizer.decode([t]),
                    logprobs=(
                        token_logprobs[0, i : i + 1].tolist() if logprobs else None
                    ),
                )

        stop_tokens = torch.tensor(self.tokenizer.stop_tokens)

        if torch.isnan(tokens).any():
            print("NaN detected in input data!")

        chunk_size = 2000
        is_prefill = True
        #self.model = torch.compile(self.model, backend="openvino", options={"device" : "CPU"})
        for cur_pos in range(min_prompt_len, total_len):

                       
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos, is_prefill, 0, 0, 0, chunk_size=chunk_size)
            

            if temperature > 0:
                if is_prefill:
                    print("printing logits final before softmax : ", str(logits[:, -1]))
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                if is_prefill:
                    print("printing probs : ", str(probs) )
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            
            next_token = next_token.reshape(-1)
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token

            target = tokens[:, prev_pos + 1 : cur_pos + 1]

            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=target,
                    reduction="none",
                    ignore_index=pad_id,
                )
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                torch.isin(next_token, stop_tokens)
            )
            
            if is_prefill: 
                print(f"Prefill time - TTFT {time.time() - start_time:.2f} seconds")

            
            is_prefill = False
            yield TokenResult(
                token=next_token[0].item(),
                text=self.tokenizer.decode(next_token.tolist()),
                logprobs=(
                    token_logprobs[:, cur_pos : cur_pos + 1][0].tolist()
                    if logprobs
                    else None
                ),
            )

            prev_pos = cur_pos
            break
            if all(eos_reached):
                break







    def text_completion(
        self,
        content: RawContent,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
        chunk_size: int = 5,
    ) -> CompletionPrediction:
        if (
            max_gen_len is None
            or max_gen_len == 0
            or max_gen_len >= self.model.params.max_seq_len
        ):
            max_gen_len = self.model.params.max_seq_len - 1

        model_input = self.formatter.encode_content(content)

        tokens = []
        token_logprobs = []
        decoded_tokens = []
        for result in self.generate(
            model_input=model_input,
            max_gen_len=max_gen_len,
            #chunk_size=chunk_size,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        ):
            tokens.append(result.token)
            if logprobs:
                decoded_tokens.append(result.text)
                token_logprobs.append(result.logprobs)

        generation = self.tokenizer.decode(tokens)
        if logprobs:
            return CompletionPrediction(
                generation=generation,
                logprobs=token_logprobs,
                decoded_tokens=decoded_tokens,
            )

        return CompletionPrediction(generation=generation)

    def chat_completion(
        self,
        messages: List[RawMessage],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        tool_prompt_format: ToolPromptFormat = ToolPromptFormat.json,
        echo: bool = False,
    ) -> ChatPrediction:
        if (
            max_gen_len is None
            or max_gen_len == 0
            or max_gen_len >= self.model.params.max_seq_len
        ):
            max_gen_len = self.model.params.max_seq_len - 1

        tokens = []
        token_logprobs = []
        decoded_tokens = []

        stop_reason = None
        for result in self.generate(
            model_input=self.formatter.encode_dialog_prompt(
                messages, tool_prompt_format
            ),
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        ):
            tokens.append(result.token)
            if result.text == "<|eot_id|>":
                stop_reason = StopReason.end_of_turn
            elif result.text == "<|eom_id|>":
                stop_reason = StopReason.end_of_message

            if logprobs:
                decoded_tokens.append(result.text)
                token_logprobs.append(result.logprobs)

        if stop_reason is None:
            stop_reason = StopReason.out_of_tokens

        message = self.formatter.decode_assistant_message(tokens, stop_reason)

        if logprobs:
            return ChatPrediction(
                generation=message,
                logprobs=token_logprobs,
                decoded_tokens=decoded_tokens,
            )

        return ChatPrediction(generation=message)

    def chat_completion_raw(
        self,
        messages: List[RawMessage],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        tool_prompt_format: ToolPromptFormat = ToolPromptFormat.json,
    ) -> List[int]:
        if (
            max_gen_len is None
            or max_gen_len == 0
            or max_gen_len >= self.model.params.max_seq_len
        ):
            max_gen_len = self.model.params.max_seq_len - 1

        output_tokens = []
        model_input = self.formatter.encode_dialog_prompt(messages, tool_prompt_format)
        input_tokens = model_input.tokens
        for result in self.generate(
            model_input=model_input,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=False,
        ):
            output_tokens.append(result.token)

        return input_tokens, output_tokens

    def text_completion_raw(
        self,
        content: RawContent,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
    ):
        if (
            max_gen_len is None
            or max_gen_len == 0
            or max_gen_len >= self.model.params.max_seq_len
        ):
            max_gen_len = self.model.params.max_seq_len - 1

        model_input = self.formatter.encode_content(content)
        input_tokens = model_input.tokens

        output_tokens = []
        for result in self.generate(
            model_input=model_input,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=False,
        ):
            output_tokens.append(result.token)

        return input_tokens, output_tokens

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
