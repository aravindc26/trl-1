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
    reward_fn: Callable[[str, List[str], List[Dict[str, str]]], float],
    max_turns: int = 10,
    max_tokens: int = 128,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:

    history = env.reset(init_prompt)

    for _ in range(max_turns):
        # 2·1  flatten dialogue ↦ input_ids
        ctx_txt = "".join(f"<|{m['role']}|>{m['content']}" for m in history)
        ctx_ids = tokenizer(ctx_txt, return_tensors="pt",
                            truncation=True, max_length=2048).input_ids.to(device)

        # 2·2  model generates next assistant turn
        with torch.no_grad():
            gen_ids = model.generate(
                ctx_ids, max_new_tokens=max_tokens, do_sample=True, top_p=0.9
            )[0][ctx_ids.shape[1]:]
        assistant_msg = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # 2·3  drive the environment
        cmd = parse_cmd(assistant_msg)
        if cmd["action"] == "navigate":
            obs, done = env.navigate(cmd["id"])
        elif cmd["action"] == "stop":
            obs, done = env.stop()
        else:
            obs, done = "Invalid command", False
            history.extend([
                {"role": "assistant", "content": assistant_msg},
                {"role": "user",      "content": obs},
            ])
        if done:
            break

    # 2·4  external reward & tensors
    reward = reward_fn(init_prompt, env.trajectory, env.history)

    prompt_txt = "".join(f"<|{m['role']}|>{m['content']}"
                         for m in env.history[:-2])          # up to last user msg
    completion_txt = env.history[-2]["content"]              # last assistant msg

    prompt_ids     = tokenizer(prompt_txt,
                               truncation=True, max_length=2048,
                               add_special_tokens=False).input_ids
    completion_ids = tokenizer(completion_txt,
                               truncation=True, max_length=512,
                               add_special_tokens=False).input_ids

    return {
        "prompt_ids":     torch.tensor(prompt_ids,     dtype=torch.long),
        "completion_ids": torch.tensor(completion_ids, dtype=torch.long),
        "rewards":        torch.tensor(reward,         dtype=torch.float),
    }

class OnlineDialogueDataset(IterableDataset):
    def __init__(
        self,
        model,
        tokenizer,
        env,
        init_prompts: List[str],
        reward_fn: Callable[[str, List[str], List[Dict[str, str]]], float],
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
        while True:
            env  = self.env
            init = random.choice(self.init_prompts)
            yield collect_episode(
                self.model, self.tokenizer, env, init,
                reward_fn=self.reward_fn, **self.kwargs
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