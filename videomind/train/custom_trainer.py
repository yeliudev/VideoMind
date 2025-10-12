# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import warnings

import nncore
import torch
from deepspeed import zero
from safetensors.torch import load_model, save_file
from torch.utils.data import Sampler
from transformers import Trainer, TrainerCallback
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import CHAT_TEMPLATE_NAME


def gather(param):
    if hasattr(param, 'ds_id'):
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def gather_lora_params(model, bias):
    assert bias in ('lora_only', 'all', 'none')

    if bias == 'lora_only':
        state_dict, maybe_lora_bias, lora_bias_names = dict(), dict(), set()
        for n, p in model.named_parameters():
            if 'modules_to_save' in n:
                state_dict[n] = p
            elif 'lora_' in n:
                state_dict[n] = p
                bias_name = n.split('lora_')[0] + 'bias'
                lora_bias_names.add(bias_name)
            elif 'bias' in n:
                maybe_lora_bias[n] = p
        for n, p in maybe_lora_bias.items():
            if n in lora_bias_names:
                state_dict[n] = p
    else:
        keys = ['lora_', 'modules_to_save', 'bias'] if bias == 'all' else ['lora_', 'modules_to_save']
        state_dict = {n: p for n, p in model.named_parameters() if any(k in n for k in keys)}

    state_dict = {n: gather(p) for n, p in state_dict.items()}
    return state_dict


def gather_key_params(model, keys):
    state_dict = {n: p for n, p in model.named_parameters() if p.requires_grad and any(k in n for k in keys)}
    state_dict = {n: gather(p) for n, p in state_dict.items()}
    return state_dict


class GroupSampler(Sampler):

    def __init__(self, group_size, data_types, seed):
        self.group_size = group_size
        self.data_types = data_types
        self.seed = seed

    def __len__(self):
        return len(self.data_types)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # avoid using dict or set here as they are not deterministic
        unique_types, groups = [], []
        for i, t in enumerate(self.data_types):
            if t not in unique_types:
                unique_types.append(t)
                groups.append([])
            groups[unique_types.index(t)].append(i)

        group_batches = []
        for group in groups:
            inds = [group[i] for i in torch.randperm(len(group), generator=g)]
            batches = [inds[i:i + self.group_size] for i in range(0, len(inds), self.group_size)]

            if len(batches[-1]) < self.group_size:
                batches = batches[:-1]

            group_batches += batches

        perm_group_batches = [group_batches[i] for i in torch.randperm(len(group_batches), generator=g)]
        inds = [i for batch in perm_group_batches for i in batch]

        return iter(inds)

    def set_epoch(self, epoch):
        self.epoch = epoch


class SetEpochCallback(TrainerCallback):

    # partially fixed in https://github.com/huggingface/accelerate/pull/3246
    # but not for the case of batch_sampler.batch_sampler.sampler
    def on_epoch_begin(self, args, state, control, **kwargs):
        shard_sampler = kwargs['train_dataloader'].batch_sampler
        batch_sampler = getattr(shard_sampler, 'batch_sampler', shard_sampler)
        batch_sampler.sampler.set_epoch(int(state.epoch))


class CustomTrainer(Trainer):

    def __init__(self, *args, processor=None, head_keys=None, **kwargs):
        super().__init__(*args, tokenizer=processor, **kwargs)
        self.add_callback(SetEpochCallback())
        self.processor = processor
        self.head_keys = head_keys

    def _get_train_sampler(self):
        if self.args.group_by_data_type:
            return GroupSampler(self.args.train_batch_size * self.args.world_size, self.train_dataset.data_types,
                                self.args.seed)
        else:
            return super()._get_train_sampler()

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if model is None:
            model = self.model

        super()._load_from_checkpoint(resume_from_checkpoint, model=model)

        partial_path = nncore.join(resume_from_checkpoint, 'pytorch_model.safetensors')
        if nncore.is_file(partial_path):
            load_model(model, partial_path, strict=False, device=model.device)

    def create_optimizer(self):
        if self.optimizer is None:
            grad_ps = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]

            decay_ps = get_parameter_names(self.model, ALL_LAYERNORM_LAYERS)
            decay_ps = [n for n in decay_ps if 'bias' not in n]

            if self.args.lora_lr is None:
                self.args.lora_lr = self.args.learning_rate

            if self.args.head_lr is None:
                self.args.head_lr = self.args.learning_rate

            lora_ps = [n for n, _ in grad_ps if 'lora' in n]
            head_ps = [n for n, _ in grad_ps if any(k in n for k in self.head_keys)]
            assert all(n not in lora_ps for n in head_ps) and all(n not in head_ps for n in lora_ps)

            groups = [{
                'params': [p for n, p in grad_ps if (n in decay_ps and n not in lora_ps and n not in head_ps)],
                'weight_decay': self.args.weight_decay
            }, {
                'params': [p for n, p in grad_ps if (n not in decay_ps and n not in lora_ps and n not in head_ps)],
                'weight_decay': 0.0
            }, {
                'params': [p for n, p in grad_ps if (n in decay_ps and n in lora_ps)],
                'weight_decay': self.args.weight_decay,
                'lr': self.args.lora_lr
            }, {
                'params': [p for n, p in grad_ps if (n not in decay_ps and n in lora_ps)],
                'weight_decay': 0.0,
                'lr': self.args.lora_lr
            }, {
                'params': [p for n, p in grad_ps if (n in decay_ps and n in head_ps)],
                'weight_decay': self.args.weight_decay,
                'lr': self.args.head_lr
            }, {
                'params': [p for n, p in grad_ps if (n not in decay_ps and n in head_ps)],
                'weight_decay': 0.0,
                'lr': self.args.head_lr
            }]

            optim_cls, kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optim_cls(groups, **kwargs)

        return self.optimizer

    def gather_and_save_model(self):
        deepspeed_zero3 = self.accelerator.deepspeed_config['zero_optimization']['stage'] == 3
        output_dir = self.args.output_dir

        if self.args.should_save:
            print(f'Saving final model to {nncore.abs_path(output_dir)}...')

        if self.processor is not None and self.args.should_save:
            self.processor.save_pretrained(output_dir)

            # https://github.com/huggingface/transformers/pull/33462
            if self.processor.chat_template is not None:
                chat_template = {'chat_template': self.processor.chat_template}
                nncore.dump(chat_template, nncore.join(output_dir, CHAT_TEMPLATE_NAME), indent=2)

        if self.args.save_full_model and self.args.lora_enable and deepspeed_zero3:
            warnings.warn('LoRA models cannot be saved in full mode under zero3, saving adapters instead')
            self.args.save_full_model = False

        if self.args.save_full_model:
            if self.args.lora_enable:
                self.model = self.model.merge_and_unload()

            if deepspeed_zero3 and not self.model_wrapped.zero_gather_16bit_weights_on_model_save():
                warnings.warn('Saving zero checkpoint, use zero_to_fp32.py to recover weights')
                self.model_wrapped.save_checkpoint(output_dir)
                return

            if deepspeed_zero3:
                state_dict = self.model_wrapped._zero3_consolidated_16bit_state_dict()
            else:
                state_dict = self.model.state_dict()

            if self.args.should_save:
                state_dict = {k[17:] if k.startswith('base_model.model.') else k: v for k, v in state_dict.items()}
                self._save(output_dir, state_dict=state_dict)
        else:
            if self.args.lora_enable:
                state_dict = gather_lora_params(self.model, self.args.lora_bias)
                if self.args.should_save:
                    self.model.save_pretrained(output_dir, state_dict=state_dict)

            if self.args.should_save:
                self.model.config.save_pretrained(output_dir)
                self.model.generation_config.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)

            state_dict = gather_key_params(self.model, self.head_keys)
            if self.args.should_save and state_dict:
                save_file(state_dict, nncore.join(output_dir, 'pytorch_model.safetensors'))

    def _save_checkpoint(self, model, trial, **kwargs):
        output_dir = self._get_output_dir(trial)
        output_dir = nncore.join(output_dir, f'{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}')

        if self.args.should_save:
            print(f'Saving checkpoint to {nncore.abs_path(output_dir)}...')

        super()._save_checkpoint(model, trial, **kwargs)

        if self.processor is not None and self.args.should_save:
            self.processor.save_pretrained(output_dir)

            # https://github.com/huggingface/transformers/pull/33462
            if self.processor.chat_template is not None:
                chat_template = {'chat_template': self.processor.chat_template}
                nncore.dump(chat_template, nncore.join(output_dir, CHAT_TEMPLATE_NAME), indent=2)

        if self.args.lora_enable:
            state_dict = gather_key_params(self.model, self.head_keys)
            if self.args.should_save:
                self.model.config.save_pretrained(output_dir)
                save_file(state_dict, nncore.join(output_dir, 'pytorch_model.safetensors'))
