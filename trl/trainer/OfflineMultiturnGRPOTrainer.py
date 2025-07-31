from typing import override
from trl import GRPOTrainer
import torch

class OfflineMultiTurnGRPOTrainer(GRPOTrainer):
    @override
    def training_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:

        input_ids     = torch.cat([inputs["prompt_ids"],
                                   inputs["completion_ids"]], dim=-1)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()

        with self.compute_loss_context():
            policy_logits = model(input_ids, attention_mask=attention_mask,
                                  use_cache=False).logits
            ref_logits    = self.ref_model(input_ids, attention_mask=attention_mask,
                                           use_cache=False).logits

        resp_start = inputs["prompt_ids"].shape[1]
        logp_pol   = policy_logits[:, resp_start:].log_softmax(-1)
        logp_ref   = ref_logits[:,  resp_start:].log_softmax(-1)
        rewards    = inputs["rewards"].unsqueeze(-1).to(logp_pol)
        advantages = rewards - rewards.mean()

        loss = -(advantages * (logp_pol - logp_ref).detach()).mean()

        if num_items_in_batch:
            loss = loss / num_items_in_batch

        return loss
