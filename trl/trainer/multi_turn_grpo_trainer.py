from ..data_utils import apply_chat_template, is_conversational
from ..models import unwrap_model_for_generation
from typing import Union, Any, List, Dict, Optional
import torch
import torch.nn as nn
from transformers import GenerationConfig
from ..trainer.grpo_trainer import GRPOTrainer
import torch.nn.functional as F


class MultiTurnGRPOTrainer(GRPOTrainer):
    # ====== config for memory logging ======
    _mem_log_every: int = 1  # set to 0 to disable, or e.g. 10 to log every 10 steps

    # ---------- utilities ----------
    def _pad_and_stack_tensors(
        self,
        tensor_list: List[torch.Tensor],
        pad_value: int,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        squeezed = [t.squeeze(0) for t in tensor_list]
        if dtype is None:
            dtype = squeezed[0].dtype
        padded = nn.utils.rnn.pad_sequence(
            squeezed, batch_first=True, padding_value=pad_value
        )
        return padded.to(dtype=dtype, device="cpu")

    def _log_gpu_mem(self, tag: str) -> None:
        """Print allocated/reserved/peak (MB) on the current CUDA device."""
        if not torch.cuda.is_available():
            return
        if hasattr(self, "accelerator") and not self.accelerator.is_main_process:
            return
        dev = torch.cuda.current_device()
        alloc = torch.cuda.memory_allocated(dev) / 1e6
        reserv = torch.cuda.memory_reserved(dev) / 1e6
        peak = torch.cuda.max_memory_allocated(dev) / 1e6
        print(f"[MEM] {tag:<18} | alloc={alloc:8.1f} MB  reserved={reserv:8.1f} MB  peak={peak:8.1f} MB")

    # ---------- training ----------
    def training_step(self, model, inputs, num_items_in_batch):
        step = getattr(self.state, "global_step", 0)
        if self._mem_log_every and (step % self._mem_log_every == 0):
            torch.cuda.reset_peak_memory_stats()
            self._log_gpu_mem("before_step")

        loss = super().training_step(model, inputs, num_items_in_batch)

        if self._mem_log_every and (step % self._mem_log_every == 0):
            self._log_gpu_mem("after_step")
        return loss

    # ---------- rollout ----------
    @torch.no_grad()
    def _get_prompt_completion(self, example: Dict[str, Any]):
        device = self.accelerator.device
        env = self.args.env_class(**self.args.env_init_kwargs)
        history = env.reset(example["question"])
        turns = 0

        while not env.ended():
            ctx_text = self.processing_class.apply_chat_template(
                history, tokenize=False, add_generation_prompt=True
            )
            prompt_inputs = self.processing_class(
                ctx_text,
                return_tensors="pt",
                max_length=self.max_prompt_length,
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
            prompt_ids_cpu = prompt_inputs["input_ids"]
            prompt_mask_cpu = prompt_inputs["attention_mask"]

            if self.max_prompt_length is not None:
                prompt_ids_cpu = prompt_ids_cpu[:, -self.max_prompt_length :]
                prompt_mask_cpu = prompt_mask_cpu[:, -self.max_prompt_length :]

            prompt_ids = prompt_ids_cpu.to(device, non_blocking=True)
            prompt_mask = prompt_mask_cpu.to(device, non_blocking=True)

            gen_config = GenerationConfig(
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                temperature=0.7,
                max_new_tokens=self.max_completion_length,  # bounded
                use_cache=True,
            )

            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped:
                unwrapped.eval()
                prompt_completion_ids_dev = unwrapped.generate(
                    prompt_ids,
                    attention_mask=prompt_mask,
                    generation_config=gen_config,
                )

            if self._mem_log_every:
                self._log_gpu_mem("after_generate")

            self.model.train()

            prompt_len = prompt_ids.size(1)
            completion_ids_dev = prompt_completion_ids_dev[:, prompt_len:]
            completion_text = self.processing_class.decode(
                completion_ids_dev[0], skip_special_tokens=True
            )
            print("completion", completion_text)

            history = env.move(completion_text)
            turns += 1
            if turns >= self.args.max_turns:
                break

        prompt_completion_ids = prompt_completion_ids_dev.to("cpu")
        prompt_ids = prompt_ids_cpu.to("cpu")
        prompt_mask = prompt_mask_cpu.to("cpu")
        return prompt_completion_ids, prompt_ids, prompt_mask, history, env

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        """
        Memory-flat scoring of the last `logits_to_keep` tokens with logs enabled by default.
        Streams BOTH the prefix (to build KV cache) and the completion (to score) so we never
        hold (B, L, V) or (B, K, V) in memory.

        Defaults:
        • Logs are always printed (text + memory).
        • Set self.args.logits_prefix_chunk to >1 (e.g., 8/16) to speed up prefix streaming.
        """
        # ---- tiny logging helpers (always on) ----
        def dlog(msg: str):
            # only main process prints if accelerator exists
            if hasattr(self, "accelerator") and not self.accelerator.is_main_process:
                return
            print(f"[logps] {msg}")

        def memlog(tag: str):
            # prefer trainer's mem logger if present
            if hasattr(self, "_log_gpu_mem"):
                try:
                    self._log_gpu_mem(tag)
                    return
                except Exception:
                    pass
            if not torch.cuda.is_available():
                return
            if hasattr(self, "accelerator") and not self.accelerator.is_main_process:
                return
            dev = torch.cuda.current_device()
            alloc = torch.cuda.memory_allocated(dev) / 1e6
            reserv = torch.cuda.memory_reserved(dev) / 1e6
            peak  = torch.cuda.max_memory_allocated(dev) / 1e6
            print(f"[MEM] {tag:<18} | alloc={alloc:8.1f} MB  reserved={reserv:8.1f} MB  peak={peak:8.1f} MB")

        try:
            assert logits_to_keep > 0, "logits_to_keep must be > 0"
            B, L = input_ids.shape
            K = int(logits_to_keep)
            assert K <= L, f"logits_to_keep={K} cannot exceed sequence length L={L}"
            device = input_ids.device
            pref = L - K  # prefix length

            # stream prefix in small chunks (1 = flattest memory; raise for speed)
            prefix_chunk = int(getattr(self.args, "logits_prefix_chunk", 1))
            prefix_chunk = max(1, prefix_chunk)

            # header
            try:
                model_dev = next(model.parameters()).device
            except StopIteration:
                model_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dlog(f"start: B={B}, L={L}, K={K}, pref={pref}, chunk={prefix_chunk}, "
                f"ids_dev={device}, model_dev={model_dev}")
            memlog("logps_start")

            @torch.inference_mode()
            def log_softmax_gather_last(logits_last, tgt_ids):
                # logits_last: (b, V), tgt_ids: (b,)
                out = F.log_softmax(logits_last.float(), dim=-1).gather(1, tgt_ids.unsqueeze(1)).squeeze(1)
                if torch.isnan(out).any() or torch.isinf(out).any():
                    dlog("WARNING: NaN/Inf detected in gathered log probs")
                return out

            past = None
            last_logits = None

            # ---------- 1) Stream PREFIX to build KV cache ----------
            if pref > 0:
                j = 0
                chunk_idx = 0
                while j < pref:
                    end = min(pref, j + prefix_chunk)
                    dlog(f"prefix chunk {chunk_idx}: [{j}:{end})")
                    try:
                        out = model(
                            input_ids=input_ids[:, j:end],
                            attention_mask=(attention_mask[:, j:end] if attention_mask is not None else None),
                            use_cache=True,
                            return_dict=True,
                            past_key_values=past,
                        )
                    except Exception as e:
                        dlog(f"ERROR in prefix forward at chunk [{j}:{end}): {e}")
                        dlog(f"input_ids chunk shape={(input_ids[:, j:end]).shape}, "
                            f"attn chunk shape={None if attention_mask is None else (attention_mask[:, j:end]).shape}")
                        memlog("error_prefix_chunk")
                        raise
                    past = out.past_key_values
                    last_logits = out.logits[:, -1, :]  # (b, V)
                    if torch.isnan(last_logits).any() or torch.isinf(last_logits).any():
                        dlog("WARNING: NaN/Inf in last_logits (prefix)")
                    memlog(f"prefix_chunk_{chunk_idx}")
                    j = end
                    chunk_idx += 1

                # score token at position `pref`
                tgt0 = input_ids[:, pref]
                logp0 = log_softmax_gather_last(last_logits, tgt0)
            else:
                # No prefix: do a 1-token warmup
                dlog("no prefix; warmup 1-token forward")
                try:
                    out = model(
                        input_ids=input_ids[:, 0:1],
                        attention_mask=(attention_mask[:, 0:1] if attention_mask is not None else None),
                        use_cache=True,
                        return_dict=True,
                    )
                except Exception as e:
                    dlog(f"ERROR in warmup forward (no prefix): {e}")
                    memlog("error_warmup")
                    raise
                past = out.past_key_values
                tgt0 = input_ids[:, 0]
                logp0 = log_softmax_gather_last(out.logits[:, -1, :], tgt0)
                memlog("warmup_done")

            # output buffer
            logps = torch.empty((B, K), device=device, dtype=torch.float32)
            logps[:, 0] = logp0

            # minimal mask for single-token continuation
            one_mask = None
            if attention_mask is not None:
                one_mask = torch.ones((B, 1), dtype=attention_mask.dtype, device=device)

            # ---------- 2) Stream COMPLETION ----------
            cur = tgt0
            for i in range(1, K):
                # occasional progress + mem log
                if i <= 3 or i == K - 1 or (i % 64 == 0):
                    dlog(f"completion step i={i}/{K-1}")
                    memlog(f"completion_i={i}")
                try:
                    out = model(
                        input_ids=cur.unsqueeze(1),    # (b, 1)
                        attention_mask=one_mask,
                        use_cache=True,
                        past_key_values=past,
                        return_dict=True,
                    )
                except Exception as e:
                    dlog(f"ERROR in completion forward at i={i}: {e}")
                    memlog("error_completion_step")
                    raise
                past = out.past_key_values
                nxt_logits = out.logits[:, -1, :]  # (b, V)
                if torch.isnan(nxt_logits).any() or torch.isinf(nxt_logits).any():
                    dlog(f"WARNING: NaN/Inf in nxt_logits at i={i}")

                tgt = input_ids[:, (pref + i) if pref > 0 else i]
                logps[:, i] = log_softmax_gather_last(nxt_logits, tgt)
                cur = tgt

            dlog(f"done: logps shape={tuple(logps.shape)}, dtype={logps.dtype}, device={logps.device}")
            memlog("logps_done")
            if torch.isnan(logps).any() or torch.isinf(logps).any():
                dlog("WARNING: NaN/Inf detected in final logps")

            return logps

        except Exception as exc:
            # Final catch-all with context; re-raise to fail loudly in training
            dlog(f"FATAL in _get_per_token_logps: {type(exc).__name__}: {exc}")
            memlog("fatal_logps")
            raise



    # ---------- collation for training ----------
    def _prepare_inputs(
        self, inputs: List[Dict[str, Union[torch.Tensor, Any]]]
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device

        pc_list, p_list, pm_list, histories, envs = [], [], [], [], []
        for example in inputs:
            for _ in range(self.num_generations):
                pc_ids, p_ids, p_mask, h, env = self._get_prompt_completion(example)
                pc_list.append(pc_ids)
                p_list.append(p_ids)
                pm_list.append(p_mask)
                histories.append(h)
                envs.append(env)

        prompt_completion_ids = self._pad_and_stack_tensors(
            pc_list, pad_value=self.processing_class.pad_token_id
        )
        prompt_ids = self._pad_and_stack_tensors(
            p_list, pad_value=self.processing_class.pad_token_id
        )
        prompt_mask = self._pad_and_stack_tensors(
            pm_list, pad_value=0
        )

        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]  # CPU

        eos = (completion_ids == self.processing_class.eos_token_id)
        seq_len = completion_ids.size(1)
        eos_any = eos.any(dim=1)
        first_eos = torch.full((eos.size(0),), seq_len, dtype=torch.long, device="cpu")
        if eos_any.any():
            first_eos[eos_any] = eos[eos_any].int().argmax(dim=1)
        arange = torch.arange(seq_len, device="cpu").expand_as(eos)
        completion_mask = (arange <= first_eos.unsqueeze(1)).to(dtype=torch.long)

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # CPU
        logits_to_keep = completion_ids.size(1)

        prompt_completion_ids = prompt_completion_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        if self._mem_log_every:
            self._log_gpu_mem("after_ref_logps")

        prompts = [h[:-1] for h in histories]
        completions = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": c}] for c in completions]

        rewards_per_func = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device
        )
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]

                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
                del reward_inputs
            else:
                reward_kwargs = {k: [] for k in inputs[0].keys() if k not in ["prompt", "completion"]}
                for k in reward_kwargs:
                    for ex in inputs:
                        reward_kwargs[k].extend([ex[k]] * self.num_generations)
                output_reward = reward_func(prompts=prompts, completions=completions, envs=envs, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward, dtype=torch.float32, device=device)

        if self._mem_log_every:
            self._log_gpu_mem("after_rewards")

        rewards = rewards_per_func.sum(dim=1)
        G = self.num_generations
        mean_grouped = rewards.view(-1, G).mean(dim=1)
        std_grouped = rewards.view(-1, G).std(dim=1)
        mean_grouped = mean_grouped.repeat_interleave(G, dim=0)
        std_grouped = std_grouped.repeat_interleave(G, dim=0)
        advantages = (rewards - mean_grouped) / (std_grouped + 1e-4)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped).mean().item())

        return {
            "prompt_ids": prompt_ids.to(device, non_blocking=True),
            "prompt_mask": prompt_mask.to(device, non_blocking=True),
            "completion_ids": completion_ids.to(device, non_blocking=True),
            "completion_mask": completion_mask.to(device, non_blocking=True),
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }
