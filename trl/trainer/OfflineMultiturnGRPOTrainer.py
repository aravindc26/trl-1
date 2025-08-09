from typing import Any, Dict, List, Optional, Union
import torch
from trl import GRPOTrainer

class OfflineMultiTurnGRPOTrainer(GRPOTrainer):
    def training_step(
        self,
        model: torch.nn.Module,
        inputs: Union[Dict[str, Any], List[Dict[str, Any]]],
        num_items_in_batch: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        # Handle list-of-microbatches or single dict
        if isinstance(inputs, list):
            if isinstance(num_items_in_batch, torch.Tensor):
                nibl = [num_items_in_batch] * len(inputs)
            elif isinstance(num_items_in_batch, list):
                nibl = num_items_in_batch
            else:
                nibl = [None] * len(inputs)
            losses = [self._training_step_one(model, mb, nib) for mb, nib in zip(inputs, nibl)]
            return sum(losses) / len(losses)
        return self._training_step_one(model, inputs, num_items_in_batch)

    def _training_step_one(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Any],
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        model.train()

        # move to device
        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(self.args.device)

        # fetch & ensure [B, L] tensors even when batch=1
        prompt_ids     = inputs["prompt_ids"];      completion_ids  = inputs["completion_ids"]
        if prompt_ids.dim() == 1:     prompt_ids = prompt_ids.unsqueeze(0)
        if completion_ids.dim() == 1: completion_ids = completion_ids.unsqueeze(0)

        prompt_mask     = inputs.get("prompt_mask")
        completion_mask = inputs.get("completion_mask")
        if prompt_mask is None:                prompt_mask = (prompt_ids != self.tokenizer.pad_token_id).long()
        elif prompt_mask.dim() == 1:           prompt_mask = prompt_mask.unsqueeze(0)
        if completion_mask is None:            completion_mask = (completion_ids != self.tokenizer.pad_token_id).long()
        elif completion_mask.dim() == 1:       completion_mask = completion_mask.unsqueeze(0)

        input_ids      = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # reference per-token logps (matches TRL behavior)
        if getattr(self, "ref_model", None) is not None:
            with torch.no_grad():
                ref_logps = self._get_per_token_logps(self.ref_model, input_ids, attention_mask, logits_to_keep)
        else:
            with self.accelerator.unwrap_model(model).disable_adapter(), torch.no_grad():
                ref_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        # advantages from external reward
        rewards = inputs.pop("rewards").view(-1).to(input_ids)
        advantages = rewards - rewards.mean()

        # hand off to TRL's compute_loss (keeps KL/masking right)
        inputs.update({
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_logps,
            "advantages": advantages,
        })

        ctx = getattr(self, "compute_loss_context_manager",
                      getattr(self, "compute_loss_context"))
        with ctx():
            loss = super().compute_loss(
                model, inputs, return_outputs=False, num_items_in_batch=num_items_in_batch
            )

        if self.args.n_gpu > 1:
            loss = loss.mean()
        self.accelerator.backward(loss)
        return loss.detach() / self.args.gradient_accumulation_steps
