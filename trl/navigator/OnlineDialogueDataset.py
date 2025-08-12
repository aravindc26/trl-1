import torch, random, re
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from typing import List, Dict, Any, Callable
from transformers import StoppingCriteria, StoppingCriteriaList

_NAV = re.compile(r'^navigate\(([A-Za-z0-9_\-\/.#]+)\)$')
_STOP = re.compile(r"stop\s*\(\s*\)", re.I)

class StopAtTurn(StoppingCriteria):
    def __init__(self, stop_tokens):
        self.stop_tokens = stop_tokens

    def __call__(self, input_ids, scores):
        last_token = input_ids[0, -1].item()
        return last_token in self.stop_tokens

# In generate:

def parse_cmd(text: str) -> Dict[str, Any]:
    if _STOP.match(text.strip()):
        return {"action": "stop"}
    m = _NAV.match(text.strip())
    if m:
        return {"action": "navigate", "id": m.group(1).strip()}
    return {"action": "invalid"}

def collect_episode(
    model,
    tokenizer,
    env,
    init_prompt: str,
    reward_fn: Callable[[str, List[str], List[Dict[str, str]], Dict[str, str]], float],
    max_turns: int = 10,
    max_tokens: int = 128,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    print("collect_episode")

    if getattr(model, "config", None) is not None and getattr(model.config, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    context_budget = min(getattr(model.config, "max_position_embeddings", 4096) - 64, 2048)
    tokenizer.truncation_side = "left"  # keep the most recent turns

    history = env.reset(init_prompt)

    stop_tokens = tokenizer.convert_tokens_to_ids(["<|assistant|>", "<|user|>", "\n"])  # Adjust based on your tokenizer
    stopping_criteria = StoppingCriteriaList([StopAtTurn(stop_tokens)])


    for _ in range(max_turns):
        # 2·1  flatten dialogue ↦ input_ids
        ctx_txt = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in history)
        ctx_txt += "<|im_start|>assistant\n"
        ctx_ids = tokenizer(ctx_txt, return_tensors="pt",
                            truncation=True, max_length=context_budget, padding=False)
        ctx_ids = {k: v.to(device) for k, v in ctx_ids.items()}

        was_gc = getattr(model, "is_gradient_checkpointing", False)
        if was_gc:
            model.gradient_checkpointing_disable()
            old_use_cache = getattr(model.config, "use_cache", True)
            model.config.use_cache = True

        # 2·2  model generates next assistant turn
        with torch.no_grad():
            out = model.generate(
                input_ids=ctx_ids["input_ids"],
                attention_mask=ctx_ids["attention_mask"],   # ← explicit mask
                max_new_tokens=max_tokens,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,    # ← explicit pad id
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=False,
                output_scores=False,
                use_cache=True,                         # ← avoids checkpointing warn
                stopping_criteria=stopping_criteria,
            )

        if was_gc:
            model.gradient_checkpointing_enable()
            model.config.use_cache = old_use_cache

        prompt_lens = ctx_ids["attention_mask"].sum(dim=1)
        # slice only newly generated tokens
        reply_ids = out[0][prompt_lens[0]:]
        assistant_msg = tokenizer.decode(reply_ids, skip_special_tokens=True)
        print("assistant_msg", assistant_msg)
        print("full", tokenizer.decode(out[0], skip_special_tokens=True))
        print("prompt len", prompt_lens[0])

        # 2·3  drive the environment
        cmd = parse_cmd(assistant_msg)
        if cmd["action"] == "navigate":
            obs, done = env.navigate(cmd["id"])
        elif cmd["action"] == "stop":
            obs, done = env.stop()
        else:
            env.trajectory.append("__FAILED__")
            obs, done = "Invalid command", True
            history.extend([
                {"role": "assistant", "content": assistant_msg},
                {"role": "user",      "content": obs},
            ])
        if done:
            break

    # 2·4  external reward & tensors
    reward = reward_fn(init_prompt, env.trajectory, env.history, env.cache)

    prompt_txt = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                         for m in env.history[:-2]) + "<|im_start|>assistant\n"         # up to last user msg
    completion_txt = env.history[-2]["content"]              # last assistant msg

    prompt_ids     = tokenizer(prompt_txt,
                               truncation=True, max_length=10000,
                               add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0)
    completion_ids = tokenizer(completion_txt,
                               truncation=True, max_length=10000,
                               add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0)

    return {
        "prompt_ids":     prompt_ids,
        "completion_ids": completion_ids,
        "rewards":        torch.tensor(reward, dtype=torch.float),
    }

class OnlineDialogueDataset(IterableDataset):
    def __init__(
        self,
        model,
        tokenizer,
        env,
        init_prompts: List[str],
        reward_fn: Callable[[str, List[str], List[Dict[str, str]], Dict[str, str]], float],
        *,
        max_turns: int = 10,
        max_tokens: int = 128,
        device: str = "cuda",
    ):
        self.model       = model
        self.tokenizer   = tokenizer
        self.env         = env
        self.init_prompts= init_prompts
        self.reward_fn   = reward_fn
        self.kwargs      = dict(max_turns=max_turns,
                                max_tokens=max_tokens,
                                device=device)

    def __iter__(self):
        indices = list(range(len(self.init_prompts)))
        for idx in indices:
            env  = self.env
            init = self.init_prompts[idx]
            yield collect_episode(
                self.model, self.tokenizer, env, init,
                reward_fn=self.reward_fn, **self.kwargs
            )

    def __len__(self):
        return len(self.init_prompts)

    def __getitem__(self, idx):
        init_prompt = self.init_prompts[idx]
        env = self.env                    # fresh environment each call
        return collect_episode(
            self.model, self.tokenizer, env, init_prompt,
            reward_fn=self.reward_fn,
            **self.kwargs
        )

def make_collate_fn(tokenizer):
    pad_id = tokenizer.pad_token_id
    def collate(batch):
        p_max = max(len(b["prompt_ids"])     for b in batch)
        c_max = max(len(b["completion_ids"]) for b in batch)
        for b in batch:
            b["prompt_ids"]     = torch.nn.functional.pad(b["prompt_ids"][:p_max],     (p_max - len(b["prompt_ids"]), 0), value=pad_id)
            b["completion_ids"] = torch.nn.functional.pad(b["completion_ids"][:c_max], (0, c_max - len(b["completion_ids"])), value=pad_id)
        out = {k: torch.stack([b[k].cpu() if torch.is_tensor(b[k]) else b[k] for b in batch]) for k in batch[0]}
        out["prompt_mask"]     = (out["prompt_ids"]     != pad_id).long()
        out["completion_mask"] = (out["completion_ids"] != pad_id).long()
        return out
    return collate


def make_dataloader(
    model,
    tokenizer,
    env,
    init_prompts,
    reward_fn,
    *,
    batch_size: int = 4,
    num_workers: int = 0,
    max_turns: int = 10,
    max_tokens: int = 128,
    device: str = "cuda",
):
    dataset = OnlineDialogueDataset(
        model, tokenizer, env, init_prompts, reward_fn,
        max_turns=max_turns, max_tokens=max_tokens, device=device
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=make_collate_fn(tokenizer),
        num_workers=num_workers,
        pin_memory=True,
    )