from ..data_utils import apply_chat_template, is_conversational
from ..models import unwrap_model_for_generation
from typing import Union, Any, List
import torch
from ..trainer.grpo_trainer import GRPOTrainer
import torch.nn as nn
from transformers import GenerationConfig

class MultiTurnGRPOTrainer(GRPOTrainer):
    def _pad_and_stack_tensors(self, tensor_list: List[torch.Tensor], pad_value: int) -> torch.Tensor:
        """Pads a list of 1D tensors to the same length and stacks them."""
        # Note: This assumes tensors are 1D. For 2D tensors like prompt_completion_ids,
        # you might need to handle padding differently or ensure they have a consistent shape.
        # This implementation handles the expected 1D/2D shapes from _get_prompt_completion.
        
        # Squeeze tensors from [1, L] to [L] if necessary
        squeezed_tensors = [t.squeeze(0) for t in tensor_list]

        # Pad to the length of the longest tensor in the list
        padded_tensors = nn.utils.rnn.pad_sequence(
            squeezed_tensors, batch_first=True, padding_value=pad_value
        )
        return padded_tensors.to(self.accelerator.device)

    def _get_prompt_completion(self, input):
        env = self.args.env_class(**self.args.env_init_kwargs)
        history = env.reset(input["question"])
        turns = 0
        while not env.ended():
            print("history", history)
            ctx_text = self.processing_class.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
            prompt_inputs = self.processing_class(ctx_text, return_tensors="pt", max_length=self.max_prompt_length, 
                padding=True, padding_side="left", add_special_tokens=False)
            prompt_ids, prompt_mask = prompt_inputs["input_ids"].to(self.accelerator.device), prompt_inputs["attention_mask"].to(self.accelerator.device)

            if self.max_prompt_length is not None:
                prompt_ids = prompt_ids[:, -self.max_prompt_length :]
                prompt_mask = prompt_mask[:, -self.max_prompt_length :]

            gen_config = GenerationConfig(do_sample=True, top_p=0.9, repetition_penalty=1.1, temperature=0.7, max_length=self.max_completion_length)

            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                print("------------------------------------------")
                print(f"Before eval(), is model in training mode? {unwrapped_model.training}")
                unwrapped_model.eval()
                print(f"After eval(), is model in training mode? {unwrapped_model.training}")
                print("------------------------------------------")
                prompt_completion_ids = unwrapped_model.generate(prompt_ids, attention_mask=prompt_mask, generation_config=gen_config) 

            print(f"3. AFTER the 'with' block, self.model.training is: {self.model.training}")

            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            
            completion = self.processing_class.decode(completion_ids[0], skip_special_tokens=True)
            print("completion", completion)
            history = env.move(completion)
            turns += 1
            if turns >= self.args.max_turns:
                break
        return prompt_completion_ids, prompt_ids, prompt_mask, history, env

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompt_completion_ids, prompt_ids, prompt_mask, histories, envs = [], [], [], [], []
        for input in inputs:
            for _ in range(self.num_generations):
                pc_ids, p_ids, p_mask, h, env = self._get_prompt_completion(input)
                prompt_completion_ids.append(pc_ids)
                prompt_ids.append(p_ids)
                prompt_mask.append(p_mask)
                histories.append(h)
                envs.append(env)
        
        prompt_completion_ids = self._pad_and_stack_tensors(prompt_completion_ids, pad_value=self.processing_class.pad_token_id)
        prompt_ids = self._pad_and_stack_tensors(prompt_ids, pad_value=self.processing_class.pad_token_id)
        prompt_mask = self._pad_and_stack_tensors(prompt_mask, pad_value=0)

        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

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
        
        prompts = [history[:-1] for history in histories]
        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
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
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        # Repeat each value in the column for `num_generations` times
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                output_reward_func = reward_func(prompts=prompts, completions=completions, envs=envs, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Log the metrics
        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }    