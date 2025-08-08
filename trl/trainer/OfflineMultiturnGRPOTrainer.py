from trl import GRPOTrainer
import torch
from typing_extensions import Dict, Any, Optional, Union

class OfflineMultiTurnGRPOTrainer(GRPOTrainer):
    def training_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        model.train()

        # move tensors to device
        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(self.args.device)

        # build full sequence + attention
        prompt_ids, completion_ids = inputs["prompt_ids"], inputs["completion_ids"]
        prompt_mask = inputs.get("prompt_mask") or (prompt_ids != self.tokenizer.pad_token_id).long()
        completion_mask = inputs.get("completion_mask") or (completion_ids != self.tokenizer.pad_token_id).long()

        input_ids      = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # reference per-token logps (handles reference-free LoRA too)
        if getattr(self, "ref_model", None) is not None:
            with torch.no_grad():
                ref_logps = self._get_per_token_logps(self.ref_model, input_ids, attention_mask, logits_to_keep)
        else:
            with self.accelerator.unwrap_model(model).disable_adapter(), torch.no_grad():
                ref_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

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

        if self.args.n_gpu > 1:
            loss = loss.mean()
        self.accelerator.backward(loss)
        return loss.detach() / self.args.gradient_accumulation_steps
