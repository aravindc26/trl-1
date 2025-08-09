from trl import GRPOTrainer
import torch
from typing_extensions import Dict, Any, Optional, Union, List

class OfflineMultiTurnGRPOTrainer(GRPOTrainer):
    # inside OfflineMultiTurnGRPOTrainer
    def _per_token_logps(
        self,
        model,
        input_ids: torch.Tensor,          # [B, T]
        attention_mask: torch.Tensor,     # [B, T]
        logits_to_keep: int,              # C (completion length)
    ) -> torch.Tensor:                    # returns [B, C]
        # forward
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits  # [B, T, V]

        # causal shift: logits[t] predicts token at labels[t]
        logits = logits[:, :-1, :]                         # [B, T-1, V]
        labels = input_ids[:, 1:]                          # [B, T-1]

        # keep only the tail corresponding to the completion tokens
        if logits_to_keep > 0:
            logits = logits[:, -logits_to_keep:, :]        # [B, C, V]
            labels = labels[:, -logits_to_keep:]           # [B, C]

        logps = torch.log_softmax(logits, dim=-1)          # [B, C, V]
        tok_logps = logps.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [B, C]
        return tok_logps

    def training_step(
        self,
        model,
        inputs: Union[Dict[str, Any], List[Dict[str, Any]]],
        num_items_in_batch: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        if isinstance(inputs, list):
            # normalize num_items_in_batch
            if isinstance(num_items_in_batch, torch.Tensor):
                nib_list = [num_items_in_batch] * len(inputs)
            elif isinstance(num_items_in_batch, list):
                nib_list = num_items_in_batch
            else:
                nib_list = [None] * len(inputs)

            losses = []
            for mb, nib in zip(inputs, nib_list):
                losses.append(self._training_step_one(model, mb, nib))
            return sum(losses) / len(losses)

        return self._training_step_one(model, inputs, num_items_in_batch)

    def _training_step_one(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        print("OfflineMultiTurnGRPOTrainer.training_step")
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.train()

        # move tensors to device
        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(self.args.device)

        # build full sequence + attention
        prompt_ids, completion_ids = inputs["prompt_ids"], inputs["completion_ids"]
        # 👇 Ensure [B, L] even when batch size == 1 or uncollated input sneaks in
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        if completion_ids.dim() == 1:
            completion_ids = completion_ids.unsqueeze(0)
        
        prompt_mask = inputs.get("prompt_mask")
        completion_mask = inputs.get("completion_mask")

        if prompt_mask is None:
            prompt_mask = (prompt_ids != self.tokenizer.pad_token_id).long()
        elif prompt_mask.dim() == 1:
            prompt_mask = prompt_mask.unsqueeze(0)

        if completion_mask is None:
            completion_mask = (completion_ids != self.tokenizer.pad_token_id).long()
        elif completion_mask.dim() == 1:
            completion_mask = completion_mask.unsqueeze(0)

        input_ids      = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # reference per-token logps (handles reference-free LoRA too)
        if getattr(self, "ref_model", None) is not None:
            with torch.no_grad():
                ref_logps = self._per_token_logps(self.ref_model, input_ids, attention_mask, logits_to_keep)
        else:
            with self.accelerator.unwrap_model(model).disable_adapter(), torch.no_grad():
                ref_logps = self._per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        # advantages from your external rewards
        rewards = inputs.pop("rewards").view(-1).to(input_ids)
        advantages = rewards - rewards.mean()

        # hand off to TRL's GRPO loss
        inputs.update({
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_logps,
            "advantages": advantages,
        })

        # compatible context manager name across versions
        ctx = getattr(self, "compute_loss_context_manager",
                      getattr(self, "compute_loss_context"))

        with ctx():
            loss = super().compute_loss(
                model,
                inputs,
                return_outputs=False,
                num_items_in_batch=num_items_in_batch,   # just pass it through
            )

        # compute these however your training_step has them
        mean_reward = self.accelerator.gather_for_metrics(inputs["rewards"]).float().mean().item()
        mean_len    = self.accelerator.gather_for_metrics(inputs["completion_mask"].sum(1)).float().mean().item()

        # If you computed per-token KL (or you rely on compute_loss to do it and stash it),
        # you can log it too. Otherwise skip.
        maybe_kl = None
        if "_metrics" in self.__dict__ and self._metrics.get("kl"):
            maybe_kl = self._metrics["kl"][-1]   # last computed by compute_loss()

        logs = {
            "train/loss":    (loss.detach() * self.args.gradient_accumulation_steps).item(),
            "train/reward":  mean_reward,
            "train/length":  mean_len,
        }
        if maybe_kl is not None:
            logs["train/kl"] = maybe_kl

        self.log(logs)           # <-- this is what ends up in W&B

        if self.args.n_gpu > 1:
            loss = loss.mean()
        self.accelerator.backward(loss)
        return loss.detach() / self.args.gradient_accumulation_steps
