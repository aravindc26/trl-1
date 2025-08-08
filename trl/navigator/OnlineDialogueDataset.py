import torch, random, re
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from typing import List, Dict, Any, Callable

_NAV = re.compile(r'^navigate\(([A-Za-z0-9_\-\/.#]+)\)$')
_STOP = re.compile(r"stop\s*\(\s*\)", re.I)

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

    for _ in range(max_turns):
        # 2·1  flatten dialogue ↦ input_ids
        ctx_txt = "".join(f"<|{m['role']}|>{m['content']}" for m in history)
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
                max_new_tokens=min(max_tokens, 200),
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,    # ← explicit pad id
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=False,
                output_scores=False,
                use_cache=True,                         # ← avoids checkpointing warn
            )

        if was_gc:
            model.gradient_checkpointing_enable()
            model.config.use_cache = old_use_cache

        # slice only newly generated tokens
        reply_ids = out[0][ctx_ids["input_ids"].shape[1]:]
        assistant_msg = tokenizer.decode(reply_ids, skip_special_tokens=True)

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

    prompt_txt = "".join(f"<|{m['role']}|>{m['content']}"
                         for m in env.history[:-2])          # up to last user msg
    completion_txt = env.history[-2]["content"]              # last assistant msg

    prompt_ids     = tokenizer(prompt_txt,
                               truncation=True, max_length=10000,
                               add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0).to(device)
    completion_ids = tokenizer(completion_txt,
                               truncation=True, max_length=10000,
                               add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0).to(device)

    return {
        "prompt_ids":     prompt_ids,
        "completion_ids": completion_ids,
        "rewards":        torch.tensor(reward, dtype=torch.float, device=device),
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
            b["prompt_ids"]     = F.pad(
                b["prompt_ids"], (p_max - len(b["prompt_ids"]), 0), value=pad_id)
            b["completion_ids"] = F.pad(
                b["completion_ids"], (0, c_max - len(b["completion_ids"])), value=pad_id)
        out = {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
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