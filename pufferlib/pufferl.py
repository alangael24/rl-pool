## puffer [train | eval | sweep] [env_name] [optional args] -- See https://puffer.ai for full detail0
# This is the same as python -m pufferlib.pufferl [train | eval | sweep] [env_name] [optional args]
# Distributed example: torchrun --standalone --nnodes=1 --nproc-per-node=6 -m pufferlib.pufferl train puffer_nmmo3

import contextlib
import warnings
warnings.filterwarnings('error', category=RuntimeWarning)

import os
import sys
import glob
import ast
import time
import random
import shutil
import argparse
import importlib
import configparser
from threading import Thread
from collections import defaultdict, deque

import numpy as np
import psutil

import torch
import torch.distributed
from torch.distributed.elastic.multiprocessing.errors import record
import torch.utils.cpp_extension

import pufferlib
import pufferlib.sweep
import pufferlib.vector
import pufferlib.pytorch
try:
    from pufferlib import _C
except ImportError:
    raise ImportError('Failed to import C/CUDA advantage kernel. If you have non-default PyTorch, try installing with --no-build-isolation')

import rich
import rich.traceback
from rich.table import Table
from rich.console import Console
from rich_argparse import RichHelpFormatter
rich.traceback.install(show_locals=False)

import signal # Aggressively exit on ctrl+c
signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

from torch.utils.cpp_extension import (
    CUDA_HOME,
    ROCM_HOME
)
# Assume advantage kernel has been built if torch has been compiled with CUDA or HIP support
# and can find CUDA or HIP in the system
ADVANTAGE_CUDA = bool(CUDA_HOME or ROCM_HOME)

class PuffeRL:
    def __init__(self, config, vecenv, policy, logger=None):
        # Backend perf optimization
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.deterministic = config['torch_deterministic']
        torch.backends.cudnn.benchmark = True

        # Reproducibility
        seed = config['seed']
        #random.seed(seed)
        #np.random.seed(seed)
        #torch.manual_seed(seed)

        # Vecenv info
        vecenv.async_reset(seed)
        obs_space = vecenv.single_observation_space
        atn_space = vecenv.single_action_space
        total_agents = vecenv.num_agents
        self.total_agents = total_agents

        # Experience
        if config['batch_size'] == 'auto' and config['bptt_horizon'] == 'auto':
            raise pufferlib.APIUsageError('Must specify batch_size or bptt_horizon')
        elif config['batch_size'] == 'auto':
            config['batch_size'] = total_agents * config['bptt_horizon']
        elif config['bptt_horizon'] == 'auto':
            config['bptt_horizon'] = config['batch_size'] // total_agents

        batch_size = config['batch_size']
        horizon = config['bptt_horizon']
        segments = batch_size // horizon
        self.segments = segments
        if total_agents > segments:
            raise pufferlib.APIUsageError(
                f'Total agents {total_agents} <= segments {segments}'
            )

        device = config['device']
        self.observations = torch.zeros(segments, horizon, *obs_space.shape,
            dtype=pufferlib.pytorch.numpy_to_torch_dtype_dict[obs_space.dtype],
            pin_memory=device == 'cuda' and config['cpu_offload'],
            device='cpu' if config['cpu_offload'] else device)
        self.actions = torch.zeros(segments, horizon, *atn_space.shape, device=device,
            dtype=pufferlib.pytorch.numpy_to_torch_dtype_dict[atn_space.dtype])
        self.values = torch.zeros(segments, horizon, device=device)
        self.logprobs = torch.zeros(segments, horizon, device=device)
        self.rewards = torch.zeros(segments, horizon, device=device)
        self.terminals = torch.zeros(segments, horizon, device=device)
        self.truncations = torch.zeros(segments, horizon, device=device)
        self.ratio = torch.ones(segments, horizon, device=device)
        self.importance = torch.ones(segments, horizon, device=device)
        self.ep_lengths = torch.zeros(total_agents, device=device, dtype=torch.int32)
        self.ep_indices = torch.arange(total_agents, device=device, dtype=torch.int32)
        self.free_idx = total_agents

        # LSTM
        if config['use_rnn']:
            n = vecenv.agents_per_batch
            h = policy.hidden_size
            self.lstm_h = {i*n: torch.zeros(n, h, device=device) for i in range(total_agents//n)}
            self.lstm_c = {i*n: torch.zeros(n, h, device=device) for i in range(total_agents//n)}

        # Minibatching & gradient accumulation
        minibatch_size = config['minibatch_size']
        max_minibatch_size = config['max_minibatch_size']
        self.minibatch_size = min(minibatch_size, max_minibatch_size)
        if minibatch_size > max_minibatch_size and minibatch_size % max_minibatch_size != 0:
            raise pufferlib.APIUsageError(
                f'minibatch_size {minibatch_size} > max_minibatch_size {max_minibatch_size} must divide evenly')

        if batch_size < minibatch_size:
            raise pufferlib.APIUsageError(
                f'batch_size {batch_size} must be >= minibatch_size {minibatch_size}'
            )

        self.accumulate_minibatches = max(1, minibatch_size // max_minibatch_size)
        self.total_minibatches = int(config['update_epochs'] * batch_size / self.minibatch_size)
        self.minibatch_segments = self.minibatch_size // horizon 
        if self.minibatch_segments * horizon != self.minibatch_size:
            raise pufferlib.APIUsageError(
                f'minibatch_size {self.minibatch_size} must be divisible by bptt_horizon {horizon}'
            )

        # Torch compile
        self.uncompiled_policy = policy
        self.policy = policy
        if config['compile']:
            self.policy = torch.compile(policy, mode=config['compile_mode'])
            self.policy.forward_eval = torch.compile(policy, mode=config['compile_mode'])
            pufferlib.pytorch.sample_logits = torch.compile(pufferlib.pytorch.sample_logits, mode=config['compile_mode'])

        # Optimizer
        if config['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(
                self.policy.parameters(),
                lr=config['learning_rate'],
                betas=(config['adam_beta1'], config['adam_beta2']),
                eps=config['adam_eps'],
            )
        elif config['optimizer'] == 'muon':
            import heavyball
            from heavyball import ForeachMuon
            warnings.filterwarnings(action='ignore', category=UserWarning, module=r'heavyball.*')
            heavyball.utils.compile_mode = "default"

            # # optionally a little bit better/faster alternative to newtonschulz iteration
            # import heavyball.utils
            # heavyball.utils.zeroth_power_mode = 'thinky_polar_express'

            # heavyball_momentum=True introduced in heavyball 2.1.1
            # recovers heavyball-1.7.2 behaviour - previously swept hyperparameters work well
            optimizer = ForeachMuon(
                self.policy.parameters(),
                lr=config['learning_rate'],
                betas=(config['adam_beta1'], config['adam_beta2']),
                eps=config['adam_eps'],
                heavyball_momentum=True,
            )
        else:
            raise ValueError(f'Unknown optimizer: {config["optimizer"]}')

        self.optimizer = optimizer

        # Logging
        self.logger = logger
        if logger is None:
            self.logger = NoLogger(config)

        # Learning rate scheduler
        epochs = config['total_timesteps'] // config['batch_size']
        eta_min = config['learning_rate'] * config['min_lr_ratio']
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=eta_min)
        self.total_epochs = epochs

        # Automatic mixed precision
        precision = config['precision']
        self.amp_context = contextlib.nullcontext()
        if config.get('amp', True) and config['device'] == 'cuda':
            self.amp_context = torch.amp.autocast(device_type='cuda', dtype=getattr(torch, precision))
        if precision not in ('float32', 'bfloat16'):
            raise pufferlib.APIUsageError(f'Invalid precision: {precision}: use float32 or bfloat16')

        # Initializations
        self.config = config
        self.vecenv = vecenv
        self.epoch = 0
        self.global_step = 0
        self.last_log_step = 0
        self.last_log_time = time.time()
        self.start_time = time.time()
        self.utilization = Utilization()
        self.profile = Profile()
        self.stats = defaultdict(list)
        self.last_stats = defaultdict(list)
        self.losses = {}

        # Dashboard
        self.model_size = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        self.print_dashboard(clear=True)

    @property
    def uptime(self):
        return time.time() - self.start_time

    @property
    def sps(self):
        if self.global_step == self.last_log_step:
            return 0

        return (self.global_step - self.last_log_step) / (time.time() - self.last_log_time)

    def evaluate(self):
        profile = self.profile
        epoch = self.epoch
        profile('eval', epoch)
        profile('eval_misc', epoch, nest=True)

        config = self.config
        device = config['device']

        if config['use_rnn']:
            for k in self.lstm_h:
                self.lstm_h[k].zero_()
                self.lstm_c[k].zero_()

        eval_sync_traj = bool(getattr(self.vecenv, 'sync_traj', True))
        if not eval_sync_traj and self.segments > self.total_agents:
            raise pufferlib.APIUsageError(
                'Async trajectory mode currently requires batch_size <= total_agents*bptt_horizon.'
            )
        active_envs = None
        if not eval_sync_traj:
            active_envs = torch.ones(self.total_agents, dtype=torch.bool, device=device)

        self.full_rows = 0
        while self.full_rows < self.segments:
            profile('env', epoch)
            o, r, d, t, info, env_id, mask = self.vecenv.recv()

            profile('eval_misc', epoch)
            env_id = np.asarray(env_id, dtype=np.int64)
            env_slice = None
            if not eval_sync_traj and config['use_rnn']:
                raise pufferlib.APIUsageError(
                    'RNN eval requires sync trajectory mode. '
                    'Use --vec.sync-traj true.'
                )
            if eval_sync_traj:
                is_contiguous = len(env_id) < 2 or np.all(np.diff(env_id) == 1)
                if is_contiguous:
                    env_slice = slice(int(env_id[0]), int(env_id[-1]) + 1)
                elif config['use_rnn']:
                    raise pufferlib.APIUsageError(
                        'RNN eval requires contiguous env_id batches. '
                        'Use --vec.sync-traj true or --vec.zero-copy true.'
                    )

            mask_arr = np.asarray(mask)

            profile('eval_copy', epoch)
            o = torch.as_tensor(o)
            o_device = o.to(device)#, non_blocking=True)
            r = torch.as_tensor(r).to(device)#, non_blocking=True)
            d = torch.as_tensor(d).to(device)#, non_blocking=True)
            t = torch.as_tensor(t).to(device)#, non_blocking=True)
            done_mask = torch.logical_or(d.bool(), t.bool())

            profile('eval_forward', epoch)
            with torch.no_grad(), self.amp_context:
                state = dict(
                    reward=r,
                    done=done_mask,
                    env_id=env_slice if env_slice is not None else env_id,
                    mask=mask,
                )

                if config['use_rnn']:
                    state['lstm_h'] = self.lstm_h[env_slice.start]
                    state['lstm_c'] = self.lstm_c[env_slice.start]

                logits, value = self.policy.forward_eval(o_device, state)
                action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
                r = torch.clamp(r, -1, 1)

            profile('eval_copy', epoch)
            with torch.no_grad():
                if config['use_rnn']:
                    self.lstm_h[env_slice.start] = state['lstm_h']
                    self.lstm_c[env_slice.start] = state['lstm_c']

                # Fast path for contiguous agent batches
                if eval_sync_traj and env_slice is not None:
                    self.global_step += int(mask_arr.sum())
                    l = self.ep_lengths[env_slice.start].item()
                    batch_rows = slice(self.ep_indices[env_slice.start].item(),
                        1 + self.ep_indices[env_slice.stop - 1].item())

                    if config['cpu_offload']:
                        self.observations[batch_rows, l] = o
                    else:
                        self.observations[batch_rows, l] = o_device

                    self.actions[batch_rows, l] = action
                    self.logprobs[batch_rows, l] = logprob
                    self.rewards[batch_rows, l] = r
                    self.terminals[batch_rows, l] = d.float()
                    self.truncations[batch_rows, l] = t.float()
                    self.values[batch_rows, l] = value.flatten()

                    # Note: We are not yet handling masks in this version
                    self.ep_lengths[env_slice] += 1
                    if l+1 >= config['bptt_horizon']:
                        num_full = env_slice.stop - env_slice.start
                        self.ep_indices[env_slice] = self.free_idx + torch.arange(
                            num_full, device=config['device']).int()
                        self.ep_lengths[env_slice] = 0
                        self.free_idx += num_full
                        self.full_rows += num_full
                else:
                    env_id_t = torch.as_tensor(env_id, device=device, dtype=torch.long)
                    keep = active_envs[env_id_t]
                    if keep.any():
                        keep_idx = torch.nonzero(keep, as_tuple=False).flatten()
                        keep_idx_cpu = keep_idx.detach().cpu().numpy()
                        self.global_step += int(mask_arr[keep_idx_cpu].sum())
                        keep_envs = env_id_t[keep_idx]
                        rows = self.ep_indices[keep_envs].long()
                        cols = self.ep_lengths[keep_envs].long()
                        values = value.flatten()

                        if config['cpu_offload']:
                            self.observations[rows.cpu(), cols.cpu()] = o[keep_idx.cpu()]
                        else:
                            self.observations[rows, cols] = o_device[keep_idx]

                        self.actions[rows, cols] = action[keep_idx]
                        self.logprobs[rows, cols] = logprob[keep_idx]
                        self.rewards[rows, cols] = r[keep_idx]
                        self.terminals[rows, cols] = d.float()[keep_idx]
                        self.truncations[rows, cols] = t.float()[keep_idx]
                        self.values[rows, cols] = values[keep_idx]

                        next_cols = cols + 1
                        done = next_cols >= config['bptt_horizon']
                        if done.any():
                            done_envs = keep_envs[done]
                            active_envs[done_envs] = False
                            self.ep_lengths[done_envs] = 0
                            self.full_rows += int(done.sum().item())

                        if (~done).any():
                            ongoing_envs = keep_envs[~done]
                            self.ep_lengths[ongoing_envs] = next_cols[~done].to(self.ep_lengths.dtype)

                action = action.cpu().numpy()
                if isinstance(logits, torch.distributions.Normal):
                    action = np.clip(action, self.vecenv.action_space.low, self.vecenv.action_space.high)

            profile('eval_misc', epoch)
            for i in info:
                for k, v in pufferlib.unroll_nested_dict(i):
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    elif isinstance(v, (list, tuple)):
                        self.stats[k].extend(v)
                    else:
                        self.stats[k].append(v)

            profile('env', epoch)
            self.vecenv.send(action)

        profile('eval_misc', epoch)
        self.free_idx = self.total_agents
        self.ep_indices = torch.arange(self.total_agents, device=device, dtype=torch.int32)
        self.ep_lengths.zero_()
        profile.end()
        return self.stats

    @record
    def train(self):
        profile = self.profile
        epoch = self.epoch
        profile('train', epoch)
        profile('train_misc', epoch, nest=True)
        losses = defaultdict(float)
        config = self.config
        device = config['device']

        b0 = config['prio_beta0']
        a = config['prio_alpha']
        clip_coef = config['clip_coef']
        vf_clip = config['vf_clip_coef']
        anneal_beta = b0 + (1 - b0)*a*self.epoch/self.total_epochs
        self.ratio[:] = 1
        recompute_advantages = config.get('recompute_advantage_each_minibatch', True)
        advantages = torch.zeros(self.values.shape, device=device)

        for mb in range(self.total_minibatches):
            profile('train_misc', epoch)
            self.amp_context.__enter__()

            if mb == 0 or recompute_advantages:
                advantages.zero_()
                advantages = compute_puff_advantage(self.values, self.rewards,
                    self.terminals, self.ratio, advantages, config['gamma'],
                    config['gae_lambda'], config['vtrace_rho_clip'], config['vtrace_c_clip'])

            # Prioritize experience by advantage magnitude
            adv = advantages.abs().sum(axis=1)
            prio_weights = torch.nan_to_num(adv**a, 0, 0, 0)
            prio_probs = (prio_weights + 1e-6)/(prio_weights.sum() + 1e-6)
            idx = torch.multinomial(prio_probs, self.minibatch_segments)
            mb_prio = (self.segments*prio_probs[idx, None])**-anneal_beta

            profile('train_copy', epoch)
            mb_obs = self.observations[idx]
            mb_actions = self.actions[idx]
            mb_logprobs = self.logprobs[idx]
            mb_rewards = self.rewards[idx]
            mb_terminals = self.terminals[idx]
            mb_truncations = self.truncations[idx]
            mb_ratio = self.ratio[idx]
            mb_values = self.values[idx]
            mb_returns = advantages[idx] + mb_values
            mb_advantages = advantages[idx]

            profile('train_forward', epoch)
            if not config['use_rnn']:
                mb_obs = mb_obs.reshape(-1, *self.vecenv.single_observation_space.shape)

            state = dict(
                action=mb_actions,
                lstm_h=None,
                lstm_c=None,
            )

            logits, newvalue = self.policy(mb_obs, state)
            actions, newlogprob, entropy = pufferlib.pytorch.sample_logits(logits, action=mb_actions)

            profile('train_misc', epoch)
            newlogprob = newlogprob.reshape(mb_logprobs.shape)
            logratio = newlogprob - mb_logprobs
            ratio = logratio.exp()
            self.ratio[idx] = ratio.detach()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > config['clip_coef']).float().mean()

            # NOTE: Commenting this out since adv is replaced below
            # adv = advantages[idx]
            # adv = compute_puff_advantage(mb_values, mb_rewards, mb_terminals,
            #     ratio, adv, config['gamma'], config['gae_lambda'],
            #     config['vtrace_rho_clip'], config['vtrace_c_clip'])

            # Weight advantages by priority and normalize
            adv = mb_advantages
            adv = mb_prio * (adv - adv.mean()) / (adv.std() + 1e-8)

            # Losses
            pg_loss1 = -adv * ratio
            pg_loss2 = -adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            newvalue = newvalue.view(mb_returns.shape)
            v_clipped = mb_values + torch.clamp(newvalue - mb_values, -vf_clip, vf_clip)
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss = 0.5*torch.max(v_loss_unclipped, v_loss_clipped).mean()

            entropy_loss = entropy.mean()

            loss = pg_loss + config['vf_coef']*v_loss - config['ent_coef']*entropy_loss
            self.amp_context.__enter__() # TODO: AMP needs some debugging

            # This breaks vloss clipping?
            self.values[idx] = newvalue.detach().float()

            # Logging
            profile('train_misc', epoch)
            losses['policy_loss'] += pg_loss.item() / self.total_minibatches
            losses['value_loss'] += v_loss.item() / self.total_minibatches
            losses['entropy'] += entropy_loss.item() / self.total_minibatches
            losses['old_approx_kl'] += old_approx_kl.item() / self.total_minibatches
            losses['approx_kl'] += approx_kl.item() / self.total_minibatches
            losses['clipfrac'] += clipfrac.item() / self.total_minibatches
            losses['importance'] += ratio.mean().item() / self.total_minibatches

            # Learn on accumulated minibatches
            profile('learn', epoch)
            loss.backward()
            if (mb + 1) % self.accumulate_minibatches == 0:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), config['max_grad_norm'])
                self.optimizer.step()
                self.optimizer.zero_grad()

        # Reprioritize experience
        profile('train_misc', epoch)
        if config['anneal_lr']:
            self.scheduler.step()

        y_pred = self.values.flatten()
        y_true = advantages.flatten() + self.values.flatten()
        var_y = y_true.var()
        explained_var = torch.nan if var_y == 0 else (1 - (y_true - y_pred).var() / var_y).item()
        losses['explained_variance'] = explained_var

        profile.end()
        logs = None
        self.epoch += 1
        done_training = self.global_step >= config['total_timesteps']
        if done_training or self.global_step == 0 or time.time() > self.last_log_time + 0.25:
            logs = self.mean_and_log()
            self.losses = losses
            self.print_dashboard()
            self.stats = defaultdict(list)
            self.last_log_time = time.time()
            self.last_log_step = self.global_step
            profile.clear()

        if self.epoch % config['checkpoint_interval'] == 0 or done_training:
            self.save_checkpoint()
            self.msg = f'Checkpoint saved at update {self.epoch}'

        return logs

    def mean_and_log(self):
        config = self.config
        for k in list(self.stats.keys()):
            v = self.stats[k]
            try:
                v = np.mean(v)
            except:
                del self.stats[k]

            self.stats[k] = v

        device = config['device']
        agent_steps = int(dist_sum(self.global_step, device))
        logs = {
            'SPS': dist_sum(self.sps, device),
            'agent_steps': agent_steps,
            'uptime': time.time() - self.start_time,
            'epoch': int(dist_sum(self.epoch, device)),
            'learning_rate': self.optimizer.param_groups[0]["lr"],
            **{f'environment/{k}': v for k, v in self.stats.items()},
            **{f'losses/{k}': v for k, v in self.losses.items()},
            **{f'performance/{k}': v['elapsed'] for k, v in self.profile},
            #**{f'environment/{k}': dist_mean(v, device) for k, v in self.stats.items()},
            #**{f'losses/{k}': dist_mean(v, device) for k, v in self.losses.items()},
            #**{f'performance/{k}': dist_sum(v['elapsed'], device) for k, v in self.profile},
        }

        if torch.distributed.is_initialized():
           if torch.distributed.get_rank() != 0:
               self.logger.log(logs, agent_steps)
               return logs
           else:
               return None

        self.logger.log(logs, agent_steps)
        return logs

    def close(self):
        self.vecenv.close()
        self.utilization.stop()
        model_path = self.save_checkpoint()
        run_id = self.logger.run_id
        path = os.path.join(self.config['data_dir'], f'{self.config["env"]}_{run_id}.pt')
        shutil.copy(model_path, path)
        return path

    def save_checkpoint(self):
        if torch.distributed.is_initialized():
           if torch.distributed.get_rank() != 0:
               return
 
        run_id = self.logger.run_id
        path = os.path.join(self.config['data_dir'], f'{self.config["env"]}_{run_id}')
        if not os.path.exists(path):
            os.makedirs(path)

        model_name = f'model_{self.config["env"]}_{self.epoch:06d}.pt'
        model_path = os.path.join(path, model_name)
        if os.path.exists(model_path):
            return model_path

        save_policy = self.uncompiled_policy
        if hasattr(save_policy, 'module'):
            save_policy = save_policy.module
        if hasattr(save_policy, '_orig_mod'):
            save_policy = save_policy._orig_mod

        try:
            state_dict = save_policy.state_dict()
        except RecursionError:
            # Fallback for rare module graph cycles introduced by wrappers/compilers.
            state_dict = {}
            for name, param in save_policy.named_parameters(recurse=True):
                state_dict[name] = param.detach()
            for name, buf in save_policy.named_buffers(recurse=True):
                state_dict[name] = buf.detach()

        torch.save(state_dict, model_path)

        state = {
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'agent_step': self.global_step,
            'update': self.epoch,
            'model_name': model_name,
            'run_id': run_id,
        }
        state_path = os.path.join(path, 'trainer_state.pt')
        torch.save(state, state_path + '.tmp')
        os.replace(state_path + '.tmp', state_path)
        return model_path

    def print_dashboard(self, clear=False, idx=[0],
            c1='[cyan]', c2='[dim default]', b1='[bright_cyan]', b2='[default]'):
        config = self.config
        sps = dist_sum(self.sps, config['device'])
        agent_steps = dist_sum(self.global_step, config['device'])
        if torch.distributed.is_initialized():
           if torch.distributed.get_rank() != 0:
               return
 
        profile = self.profile
        console = Console()
        dashboard = Table(box=rich.box.ROUNDED, expand=True,
            show_header=False, border_style='bright_cyan')
        table = Table(box=None, expand=True, show_header=False)
        dashboard.add_row(table)

        table.add_column(justify="left", width=30)
        table.add_column(justify="center", width=12)
        table.add_column(justify="center", width=12)
        table.add_column(justify="center", width=13)
        table.add_column(justify="right", width=13)

        table.add_row(
            f'{b1}PufferLib {b2}3.0 {idx[0]*" "}:blowfish:',
            f'{c1}CPU: {b2}{np.mean(self.utilization.cpu_util):.1f}{c2}%',
            f'{c1}GPU: {b2}{np.mean(self.utilization.gpu_util):.1f}{c2}%',
            f'{c1}DRAM: {b2}{np.mean(self.utilization.cpu_mem):.1f}{c2}%',
            f'{c1}VRAM: {b2}{np.mean(self.utilization.gpu_mem):.1f}{c2}%',
        )
        idx[0] = (idx[0] - 1) % 10
            
        s = Table(box=None, expand=True)
        remaining = f'{b2}A hair past a freckle{c2}'
        if sps != 0:
            remaining = duration((config['total_timesteps'] - agent_steps)/sps, b2, c2)

        s.add_column(f"{c1}Summary", justify='left', vertical='top', width=10)
        s.add_column(f"{c1}Value", justify='right', vertical='top', width=14)
        s.add_row(f'{b2}Env', f'{b2}{config["env"]}')
        s.add_row(f'{b2}Params', abbreviate(self.model_size, b2, c2))
        s.add_row(f'{b2}Steps', abbreviate(agent_steps, b2, c2))
        s.add_row(f'{b2}SPS', abbreviate(sps, b2, c2))
        s.add_row(f'{b2}Epoch', f'{b2}{self.epoch}')
        s.add_row(f'{b2}Uptime', duration(self.uptime, b2, c2))
        s.add_row(f'{b2}Remaining', remaining)

        delta = profile.eval['buffer'] + profile.train['buffer']
        p = Table(box=None, expand=True, show_header=False)
        p.add_column(f"{c1}Performance", justify="left", width=10)
        p.add_column(f"{c1}Time", justify="right", width=8)
        p.add_column(f"{c1}%", justify="right", width=4)
        p.add_row(*fmt_perf('Evaluate', b1, delta, profile.eval, b2, c2))
        p.add_row(*fmt_perf('  Forward', b2, delta, profile.eval_forward, b2, c2))
        p.add_row(*fmt_perf('  Env', b2, delta, profile.env, b2, c2))
        p.add_row(*fmt_perf('  Copy', b2, delta, profile.eval_copy, b2, c2))
        p.add_row(*fmt_perf('  Misc', b2, delta, profile.eval_misc, b2, c2))
        p.add_row(*fmt_perf('Train', b1, delta, profile.train, b2, c2))
        p.add_row(*fmt_perf('  Forward', b2, delta, profile.train_forward, b2, c2))
        p.add_row(*fmt_perf('  Learn', b2, delta, profile.learn, b2, c2))
        p.add_row(*fmt_perf('  Copy', b2, delta, profile.train_copy, b2, c2))
        p.add_row(*fmt_perf('  Misc', b2, delta, profile.train_misc, b2, c2))

        l = Table(box=None, expand=True, )
        l.add_column(f'{c1}Losses', justify="left", width=16)
        l.add_column(f'{c1}Value', justify="right", width=8)
        for metric, value in self.losses.items():
            l.add_row(f'{b2}{metric}', f'{b2}{value:.3f}')

        monitor = Table(box=None, expand=True, pad_edge=False)
        monitor.add_row(s, p, l)
        dashboard.add_row(monitor)

        table = Table(box=None, expand=True, pad_edge=False)
        dashboard.add_row(table)
        left = Table(box=None, expand=True)
        right = Table(box=None, expand=True)
        table.add_row(left, right)
        left.add_column(f"{c1}User Stats", justify="left", width=20)
        left.add_column(f"{c1}Value", justify="right", width=10)
        right.add_column(f"{c1}User Stats", justify="left", width=20)
        right.add_column(f"{c1}Value", justify="right", width=10)
        i = 0

        if self.stats:
            self.last_stats = self.stats

        for metric, value in (self.stats or self.last_stats).items():
            try: # Discard non-numeric values
                int(value)
            except:
                continue

            u = left if i % 2 == 0 else right
            u.add_row(f'{b2}{metric}', f'{b2}{value:.3f}')
            i += 1
            if i == 30:
                break

        if clear:
            console.clear()

        with console.capture() as capture:
            console.print(dashboard)

        print('\033[0;0H' + capture.get())

def compute_puff_advantage(values, rewards, terminals,
        ratio, advantages, gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip):
    '''CUDA kernel for puffer advantage with automatic CPU fallback. You need
    nvcc (in cuda-dev-tools or in a cuda-dev docker base) for PufferLib to
    compile the fast version.'''

    device = values.device
    if not ADVANTAGE_CUDA:
        values = values.cpu()
        rewards = rewards.cpu()
        terminals = terminals.cpu()
        ratio = ratio.cpu()
        advantages = advantages.cpu()

    torch.ops.pufferlib.compute_puff_advantage(values, rewards, terminals,
        ratio, advantages, gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip)

    if not ADVANTAGE_CUDA:
        return advantages.to(device)

    return advantages


def abbreviate(num, b2, c2):
    if num < 1e3:
        return f'{b2}{num}{c2}'
    elif num < 1e6:
        return f'{b2}{num/1e3:.1f}{c2}K'
    elif num < 1e9:
        return f'{b2}{num/1e6:.1f}{c2}M'
    elif num < 1e12:
        return f'{b2}{num/1e9:.1f}{c2}B'
    else:
        return f'{b2}{num/1e12:.2f}{c2}T'

def duration(seconds, b2, c2):
    if seconds < 0:
        return f"{b2}0{c2}s"
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{b2}{h}{c2}h {b2}{m}{c2}m {b2}{s}{c2}s" if h else f"{b2}{m}{c2}m {b2}{s}{c2}s" if m else f"{b2}{s}{c2}s"

def fmt_perf(name, color, delta_ref, prof, b2, c2):
    percent = 0 if delta_ref == 0 else int(100*prof['buffer']/delta_ref - 1e-5)
    return f'{color}{name}', duration(prof['elapsed'], b2, c2), f'{b2}{percent:2d}{c2}%'

def fmt_perf2(name, color, delta_ref, elapsed, b2, c2):
    percent = 0 if delta_ref == 0 else int(100*elapsed/delta_ref - 1e-5)
    return f'{color}{name}', duration(elapsed, b2, c2), f'{b2}{percent:2d}{c2}%'

def dist_sum(value, device):
    if not torch.distributed.is_initialized():
        return value

    tensor = torch.tensor(value, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor.item()

def dist_mean(value, device):
    if not torch.distributed.is_initialized():
        return value

    return dist_sum(value, device) / torch.distributed.get_world_size()

class Profile:
    def __init__(self, frequency=5):
        self.profiles = defaultdict(lambda: defaultdict(float))
        self.frequency = frequency
        self.stack = []

    def __iter__(self):
        return iter(self.profiles.items())

    def __getattr__(self, name):
        return self.profiles[name]

    def __call__(self, name, epoch, nest=False):
        # Skip profiling the first few epochs, which are noisy due to setup
        if (epoch + 1) % self.frequency != 0:
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        tick = time.time()
        if len(self.stack) != 0 and not nest:
            self.pop(tick)

        self.stack.append(name)
        self.profiles[name]['start'] = tick

    def pop(self, end):
        profile = self.profiles[self.stack.pop()]
        delta = end - profile['start']
        profile['delta'] += delta
        # Multiply delta by freq to account for skipped epochs
        profile['elapsed'] += delta * self.frequency

    def end(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.time()
        for i in range(len(self.stack)):
            self.pop(end)

    def clear(self):
        for prof in self.profiles.values():
            if prof['delta'] > 0:
                prof['buffer'] = prof['delta']
                prof['delta'] = 0

class Utilization(Thread):
    def __init__(self, delay=1, maxlen=20):
        super().__init__()
        self.cpu_mem = deque([0], maxlen=maxlen)
        self.cpu_util = deque([0], maxlen=maxlen)
        self.gpu_util = deque([0], maxlen=maxlen)
        self.gpu_mem = deque([0], maxlen=maxlen)
        self.stopped = False
        self.delay = delay
        self.start()

    def run(self):
        while not self.stopped:
            self.cpu_util.append(100*psutil.cpu_percent()/psutil.cpu_count())
            mem = psutil.virtual_memory()
            self.cpu_mem.append(100*mem.active/mem.total)
            if torch.cuda.is_available():
                # Monitoring in distributed crashes nvml
                if torch.distributed.is_initialized():
                   time.sleep(self.delay)
                   continue

                self.gpu_util.append(torch.cuda.utilization())
                free, total = torch.cuda.mem_get_info()
                self.gpu_mem.append(100*(total-free)/total)
            else:
                self.gpu_util.append(0)
                self.gpu_mem.append(0)

            time.sleep(self.delay)

    def stop(self):
        self.stopped = True

def downsample(data_list, num_points):
    if not data_list or num_points <= 0:
        return []
    if num_points == 1:
        return [data_list[-1]]
    if len(data_list) <= num_points:
        return data_list

    last = data_list[-1]
    data_list = data_list[:-1]

    data_np = np.array(data_list)
    num_points -= 1  # one down for the last one

    n = (len(data_np) // num_points) * num_points
    data_np = data_np[-n:] if n > 0 else data_np
    downsampled = data_np.reshape(num_points, -1).mean(axis=1)

    return downsampled.tolist() + [last]

def _coerce_cpp_threads(train_cfg, vec_cfg):
    threads = train_cfg.get('cpp_num_threads', vec_cfg.get('num_workers', 'auto'))
    if isinstance(threads, str):
        if threads == 'auto':
            cores = psutil.cpu_count(logical=False) or psutil.cpu_count() or 8
            return max(1, int(cores))
        threads = int(threads)
    return max(1, int(threads))

def _infer_agents_per_env(env_name, args):
    short_name = env_name.replace('puffer_', '')
    if short_name == 'pool':
        num_tables = int(args.get('env', {}).get('num_envs', 1))
        return max(1, num_tables * 2)

    package = args['package']
    module_name = 'pufferlib.ocean' if package == 'ocean' else f'pufferlib.environments.{package}'
    env_module = importlib.import_module(module_name)
    make_env = env_module.env_creator(env_name)

    env_kwargs = dict(args['env'])

    try:
        probe = make_env(**env_kwargs)
    except TypeError:
        try:
            probe = make_env()
        except Exception:
            fallback = int(args.get('env', {}).get('num_agents', 1))
            return max(1, fallback)
    except Exception:
        fallback = int(args.get('env', {}).get('num_agents', 1))
        return max(1, fallback)

    try:
        return int(getattr(probe, 'num_agents', 1))
    finally:
        close_fn = getattr(probe, 'close', None)
        if callable(close_fn):
            close_fn()

def _map_cpp_configs(env_name, args):
    train = args['train']
    vec = args['vec']
    env_cfg = dict(args['env'])
    policy = dict(args.get('policy', {}))

    agents_per_env = _infer_agents_per_env(env_name, args)
    vec_num_envs = int(vec.get('num_envs', 1))
    total_agents = max(1, vec_num_envs * agents_per_env)

    raw_num_buffers = train.get('cpp_num_buffers', 'auto')
    if isinstance(raw_num_buffers, str) and raw_num_buffers == 'auto':
        num_buffers = vec_num_envs
    else:
        num_buffers = int(raw_num_buffers)
    num_buffers = max(1, min(num_buffers, total_agents))
    while total_agents % num_buffers != 0 and num_buffers > 1:
        num_buffers -= 1

    num_threads = _coerce_cpp_threads(train, vec)

    cpp_train = {
        'env': env_name,
        'env_name': env_name,
        'data_dir': train['data_dir'],
        'checkpoint_interval': int(train['checkpoint_interval']),
        'total_timesteps': int(train['total_timesteps']),
        'horizon': int(train.get('bptt_horizon', 64)),
        'learning_rate': float(train['learning_rate']),
        'min_lr_ratio': float(train.get('min_lr_ratio', 0.0)),
        'anneal_lr': 1 if train.get('anneal_lr', True) else 0,
        'beta1': float(train.get('adam_beta1', train.get('beta1', 0.95))),
        'beta2': float(train.get('adam_beta2', train.get('beta2', 0.999))),
        'eps': float(train.get('adam_eps', train.get('eps', 1e-12))),
        'minibatch_size': int(train['minibatch_size']),
        'replay_ratio': float(train.get('update_epochs', train.get('replay_ratio', 1.0))),
        'max_grad_norm': float(train['max_grad_norm']),
        'clip_coef': float(train['clip_coef']),
        'vf_clip_coef': float(train['vf_clip_coef']),
        'vf_coef': float(train['vf_coef']),
        'ent_coef': float(train['ent_coef']),
        'gamma': float(train['gamma']),
        'gae_lambda': float(train['gae_lambda']),
        'vtrace_rho_clip': float(train['vtrace_rho_clip']),
        'vtrace_c_clip': float(train['vtrace_c_clip']),
        'prio_alpha': float(train['prio_alpha']),
        'prio_beta0': float(train['prio_beta0']),
        'use_rnn': 1 if train.get('use_rnn', False) else 0,
        'cudagraphs': int(train.get('cudagraphs', -1)),
        'kernels': 1 if train.get('kernels', True) else 0,
        'profile': 1 if train.get('profile', False) else 0,
        'rank': 0,
        'world_size': 1,
        'nccl_id_path': '/tmp/puffer_nccl_id',
    }
    vec_cfg = {
        'total_agents': int(total_agents),
        'num_buffers': int(num_buffers),
        'num_threads': int(num_threads),
    }
    policy_cfg = {
        'hidden_size': int(policy.get('hidden_size', 128)),
        'num_layers': int(policy.get('num_layers', 1)),
    }
    return cpp_train, vec_cfg, env_cfg, policy_cfg

class CppPuffeRL:
    def __init__(self, config, vec_config, env_config, policy_config, logger=None):
        if not hasattr(_C, 'create_pufferl'):
            raise ImportError(
                'C++ runtime symbols are missing in pufferlib._C. '
                'Build with: python setup.py build_pool --inplace'
            )

        self.config = config
        self.logger = logger or NoLogger(config)
        self.pufferl_cpp = _C.create_pufferl(config, vec_config, env_config, policy_config)
        self.policy_fp32 = self.pufferl_cpp.policy_fp32

        self.global_step = 0
        self.epoch = 0
        self.last_log_step = 0
        self.last_log_time = time.time()
        self.start_time = time.time()
        self.batch_size = int(config['horizon'] * vec_config['total_agents'])

        self.stats = {}
        self.losses = {}
        self.profile = {}
        self.utilization = {}
        self.msg = ''
        self.model_size = sum(p.numel() for p in self.policy_fp32.parameters() if p.requires_grad)

    @property
    def uptime(self):
        return time.time() - self.start_time

    @property
    def sps(self):
        if self.global_step == self.last_log_step:
            return 0
        return (self.global_step - self.last_log_step) / max(1e-9, time.time() - self.last_log_time)

    def evaluate(self):
        _C.rollouts(self.pufferl_cpp)
        self.global_step += self.batch_size

    def write_logs(self, env_logs):
        agent_steps = int(self.global_step)
        logs = {
            'SPS': int(self.sps),
            'agent_steps': agent_steps,
            'uptime': self.uptime,
            'epoch': int(self.epoch),
            **{f'environment/{k}': v for k, v in env_logs.items()},
            **{f'losses/{k}': v for k, v in self.losses.items()},
            **{f'performance/{k}': v for k, v in self.profile.items()},
        }
        self.logger.log(logs, agent_steps)
        return logs

    def save_checkpoint(self):
        run_id = self.logger.run_id
        path = os.path.join(self.config['data_dir'], f'{self.config["env"]}_{run_id}')
        if not os.path.exists(path):
            os.makedirs(path)

        model_name = f'model_{self.config["env"]}_{self.epoch:06d}.pt'
        model_path = os.path.join(path, model_name)
        if not os.path.exists(model_path):
            torch.save(dict(self.policy_fp32.named_parameters()), model_path)

        state = {
            'global_step': self.global_step,
            'agent_step': self.global_step,
            'update': self.epoch,
            'model_name': model_name,
            'run_id': run_id,
        }
        state_path = os.path.join(path, 'trainer_state.pt')
        torch.save(state, state_path + '.tmp')
        os.replace(state_path + '.tmp', state_path)
        return model_path

    def print_dashboard(self):
        score = self.stats.get('score', 0.0)
        perf = self.stats.get('perf', 0.0)
        print(
            f'[CPP] step={self.global_step} epoch={self.epoch} '
            f'sps={self.sps/1e3:.1f}K score={score:.3f} perf={perf:.3f}'
        )

    def train(self):
        _C.train(self.pufferl_cpp)
        self.epoch += 1

        logs = None
        done_training = self.global_step >= self.config['total_timesteps']
        if done_training or self.global_step == 0 or time.time() > self.last_log_time + 0.6:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            self.stats = _C.log_environments(self.pufferl_cpp)
            self.losses = _C.log_losses(self.pufferl_cpp)
            self.profile = _C.log_profile(self.pufferl_cpp)
            self.utilization = _C.log_utilization(self.pufferl_cpp)

            logs = self.write_logs(self.stats)
            self.print_dashboard()
            self.last_log_time = time.time()
            self.last_log_step = self.global_step

        if self.epoch % self.config['checkpoint_interval'] == 0 or done_training:
            self.save_checkpoint()

        return logs

    def close(self):
        model_path = self.save_checkpoint()
        _C.close(self.pufferl_cpp)
        self.pufferl_cpp = None
        return model_path

def train_cpp(env_name, args=None, logger=None, early_stop_fn=None):
    args = args or load_config(env_name)
    cpp_train, vec_cfg, env_cfg, policy_cfg = _map_cpp_configs(env_name, args)

    if args['neptune']:
        logger = NeptuneLogger(args)
    elif args['wandb']:
        logger = WandbLogger(args)
    else:
        logger = logger or NoLogger(args)

    pufferl = CppPuffeRL(cpp_train, vec_cfg, env_cfg, policy_cfg, logger=logger)
    logging_threshold = min(0.20*cpp_train['total_timesteps'], 100_000_000)
    all_logs = []

    while pufferl.global_step < cpp_train['total_timesteps']:
        pufferl.evaluate()
        logs = pufferl.train()
        if logs is None:
            continue

        should_stop_early = False
        if early_stop_fn is not None:
            should_stop_early = early_stop_fn(logs)
            if 'early_stop_threshold' in logs:
                pufferl.logger.log({'environment/early_stop_threshold': logs['early_stop_threshold']}, logs['agent_steps'])

        if pufferl.global_step > logging_threshold:
            all_logs.append(logs)

        if should_stop_early:
            model_path = pufferl.close()
            pufferl.logger.close(model_path, early_stop=True)
            return all_logs

    model_path = pufferl.close()
    pufferl.logger.close(model_path, early_stop=False)
    return all_logs

class NoLogger:
    def __init__(self, args):
        self.run_id = str(int(100*time.time()))

    def log(self, logs, step):
        pass

    def close(self, model_path, early_stop):
        pass

class NeptuneLogger:
    def __init__(self, args, load_id=None, mode='async'):
        import neptune as nept
        neptune_name = args['neptune_name']
        neptune_project = args['neptune_project']
        neptune = nept.init_run(
            project=f"{neptune_name}/{neptune_project}",
            capture_hardware_metrics=False,
            capture_stdout=False,
            capture_stderr=False,
            capture_traceback=False,
            with_id=load_id,
            mode=mode,
            tags = [args['tag']] if args['tag'] is not None else [],
        )
        self.run_id = neptune._sys_id
        self.neptune = neptune
        for k, v in pufferlib.unroll_nested_dict(args):
            neptune[k].append(v)
        self.should_upload_model = not args['no_model_upload']

    def log(self, logs, step):
        for k, v in logs.items():
            self.neptune[k].append(v, step=step)

    def upload_model(self, model_path):
        self.neptune['model'].track_files(model_path)

    def close(self, model_path, early_stop):
        self.neptune['early_stop'] = early_stop
        if self.should_upload_model:
            self.upload_model(model_path)
        self.neptune.stop()

    def download(self):
        self.neptune["model"].download(destination='artifacts')
        return f'artifacts/{self.run_id}.pt'
 
class WandbLogger:
    def __init__(self, args, load_id=None, resume='allow'):
        import wandb
        wandb.init(
            id=load_id or wandb.util.generate_id(),
            project=args['wandb_project'],
            group=args['wandb_group'],
            allow_val_change=True,
            save_code=False,
            resume=resume,
            config=args,
            tags = [args['tag']] if args['tag'] is not None else [],
            settings=wandb.Settings(console="off"),  # stop sending dashboard to wandb
        )
        self.wandb = wandb
        self.run_id = wandb.run.id
        self.should_upload_model = not args['no_model_upload']

    def log(self, logs, step):
        self.wandb.log(logs, step=step)

    def upload_model(self, model_path):
        artifact = self.wandb.Artifact(self.run_id, type='model')
        artifact.add_file(model_path)
        self.wandb.run.log_artifact(artifact)

    def close(self, model_path, early_stop):
        self.wandb.run.summary['early_stop'] = early_stop
        if self.should_upload_model:
            self.upload_model(model_path)
        self.wandb.finish()

    def download(self):
        artifact = self.wandb.use_artifact(f'{self.run_id}:latest')
        data_dir = artifact.download()
        model_file = max(os.listdir(data_dir))
        return f'{data_dir}/{model_file}'

def train(env_name, args=None, vecenv=None, policy=None, logger=None, early_stop_fn=None):
    args = args or load_config(env_name)
    if str(args['train'].get('runtime', 'python')).lower() == 'cpp':
        return train_cpp(env_name, args=args, logger=logger, early_stop_fn=early_stop_fn)

    # Assume TorchRun DDP is used if LOCAL_RANK is set
    if 'LOCAL_RANK' in os.environ:
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        print("World size", world_size)
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        master_port = os.environ.get('MASTER_PORT', '29500')
        local_rank = int(os.environ["LOCAL_RANK"])
        print(f"rank: {local_rank}, MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")
        torch.cuda.set_device(local_rank)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

    vecenv = vecenv or load_env(env_name, args)
    policy = policy or load_policy(args, vecenv, env_name)

    if 'LOCAL_RANK' in os.environ:
        args['train']['device'] = torch.cuda.current_device()
        torch.distributed.init_process_group(backend='nccl', world_size=world_size)
        policy = policy.to(local_rank)
        model = torch.nn.parallel.DistributedDataParallel(
            policy, device_ids=[local_rank], output_device=local_rank
        )
        if hasattr(policy, 'lstm'):
            #model.lstm = policy.lstm
            model.hidden_size = policy.hidden_size

        model.forward_eval = policy.forward_eval
        policy = model.to(local_rank)

    if args['neptune']:
        logger = NeptuneLogger(args)
    elif args['wandb']:
        logger = WandbLogger(args)

    train_config = { **args['train'], 'env': env_name }
    pufferl = PuffeRL(train_config, vecenv, policy, logger)

    # Sweep needs data for early stopped runs, so send data when steps > 100M
    logging_threshold = min(0.20*train_config['total_timesteps'], 100_000_000)
    all_logs = []

    while pufferl.global_step < train_config['total_timesteps']:
        if train_config['device'] == 'cuda':
            torch.compiler.cudagraph_mark_step_begin()
        pufferl.evaluate()
        if train_config['device'] == 'cuda':
            torch.compiler.cudagraph_mark_step_begin()
        logs = pufferl.train()

        if logs is not None:
            should_stop_early = False
            if early_stop_fn is not None:
                should_stop_early = early_stop_fn(logs)
                # This is hacky, but need to see if threshold looks reasonable
                if 'early_stop_threshold' in logs:
                    pufferl.logger.log({'environment/early_stop_threshold': logs['early_stop_threshold']}, logs['agent_steps'])

            if pufferl.global_step > logging_threshold:
                all_logs.append(logs)

            if should_stop_early:
                model_path = pufferl.close()
                pufferl.logger.close(model_path, early_stop=True)
                return all_logs

    # Final eval. You can reset the env here, but depending on
    # your env, this can skew data (i.e. you only collect the shortest
    # rollouts within a fixed number of epochs)
    for i in range(128):  # Run eval for at least 32, but put a hard stop at 128.
        stats = pufferl.evaluate()
        if i >= 32 and stats:
            break

    logs = pufferl.mean_and_log()
    if logs is not None:
        all_logs.append(logs)

    pufferl.print_dashboard()
    model_path = pufferl.close()
    pufferl.logger.close(model_path, early_stop=False)
    return all_logs

def eval(env_name, args=None, vecenv=None, policy=None):
    args = args or load_config(env_name)
    backend = args['vec']['backend']
    if backend != 'PufferEnv':
        backend = 'Serial'

    args['vec'] = dict(backend=backend, num_envs=1)
    vecenv = vecenv or load_env(env_name, args)

    policy = policy or load_policy(args, vecenv, env_name)
    ob, info = vecenv.reset()
    driver = vecenv.driver_env
    num_agents = vecenv.observation_space.shape[0]
    device = args['train']['device']

    state = {}
    if args['train']['use_rnn']:
        state = dict(
            lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
            lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
        )

    frames = []
    while True:
        render = driver.render()
        if len(frames) < args['save_frames']:
            frames.append(render)

        # Screenshot Ocean envs with F12, gifs with control + F12
        if driver.render_mode == 'ansi':
            print('\033[0;0H' + render + '\n')
            time.sleep(1/args['fps'])
        elif driver.render_mode == 'rgb_array':
            pass
            #import cv2
            #render = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)
            #cv2.imshow('frame', render)
            #cv2.waitKey(1)
            #time.sleep(1/args['fps'])

        with torch.no_grad():
            ob = torch.as_tensor(ob).to(device)
            logits, value = policy.forward_eval(ob, state)
            action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
            action = action.cpu().numpy().reshape(vecenv.action_space.shape)

        if isinstance(logits, torch.distributions.Normal):
            action = np.clip(action, vecenv.action_space.low, vecenv.action_space.high)

        ob = vecenv.step(action)[0]

        if len(frames) > 0 and len(frames) == args['save_frames']:
            import imageio
            imageio.mimsave(args['gif_path'], frames, fps=args['fps'], loop=0)
            print(f'Saved {len(frames)} frames to {args["gif_path"]}')

def stop_if_loss_nan(logs):
    return any("losses/" in k and np.isnan(v) for k, v in logs.items())

def sweep(args=None, env_name=None):
    args = args or load_config(env_name)
    if not args['wandb'] and not args['neptune']:
        raise pufferlib.APIUsageError('Sweeps require either wandb or neptune')
    args['no_model_upload'] = True  # Uploading trained model during sweep crashed wandb

    method = args['sweep'].pop('method')
    try:
        sweep_cls = getattr(pufferlib.sweep, method)
    except:
        raise pufferlib.APIUsageError(f'Invalid sweep method {method}. See pufferlib.sweep')

    sweep = sweep_cls(args['sweep'])
    points_per_run = args['sweep']['downsample']
    target_key = f'environment/{args["sweep"]["metric"]}'
    running_target_buffer = deque(maxlen=30)

    def stop_if_perf_below(logs):
        if stop_if_loss_nan(logs):
            logs['is_loss_nan'] = True
            return True

        if method != 'Protein':
            return False

        if ('uptime' in logs and target_key in logs):
            metric_val, cost = logs[target_key], logs['uptime']
            running_target_buffer.append(metric_val)
            target_running_mean = np.mean(running_target_buffer)
            
            # If metric distribution is percentile, threshold is also logit transformed
            threshold = sweep.get_early_stop_threshold(cost)
            logs['early_stop_threshold'] = max(threshold, -5)  # clipping for visualization

            if sweep.should_stop(max(target_running_mean, metric_val), cost):
                logs['is_loss_nan'] = False
                return True
        return False

    for i in range(args['max_runs']):
        seed = time.time_ns() & 0xFFFFFFFF
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # In the first run, skip sweep and use the train args specified in the config
        if i > 0:
            sweep.suggest(args)

        all_logs = train(env_name, args=args, early_stop_fn=stop_if_perf_below)
        all_logs = [e for e in all_logs if target_key in e]

        if not all_logs:
            sweep.observe(args, 0, 0, is_failure=True)
            continue

        total_timesteps = args['train']['total_timesteps']

        scores = downsample([log[target_key] for log in all_logs], points_per_run)
        costs = downsample([log['uptime'] for log in all_logs], points_per_run)
        timesteps = downsample([log['agent_steps'] for log in all_logs], points_per_run)

        is_final_loss_nan = all_logs[-1].get('is_loss_nan', False)
        if is_final_loss_nan:
            s = scores.pop()
            c = costs.pop()
            args['train']['total_timesteps'] = timesteps.pop()
            sweep.observe(args, s, c, is_failure=True)

        for score, cost, timestep in zip(scores, costs, timesteps):
            args['train']['total_timesteps'] = timestep
            sweep.observe(args, score, cost)

        # Prevent logging final eval steps as training steps
        args['train']['total_timesteps'] = total_timesteps

def profile(args=None, env_name=None, vecenv=None, policy=None):
    args = load_config()
    vecenv = vecenv or load_env(env_name, args)
    policy = policy or load_policy(args, vecenv)

    train_config = dict(**args['train'], env=args['env_name'], tag=args['tag'])
    pufferl = PuffeRL(train_config, vecenv, policy, neptune=args['neptune'], wandb=args['wandb'])

    import torchvision.models as models
    from torch.profiler import profile, record_function, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            for _ in range(10):
                stats = pufferl.evaluate()
                pufferl.train()

    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
    prof.export_chrome_trace("trace.json")

def export(args=None, env_name=None, vecenv=None, policy=None):
    args = args or load_config(env_name)
    args['vec'] = dict(backend='Serial', num_envs=1)
    vecenv = vecenv or load_env(env_name, args)
    policy = policy or load_policy(args, vecenv)

    weights = []
    for name, param in policy.named_parameters():
        weights.append(param.data.cpu().numpy().flatten())
        print(name, param.shape, param.data.cpu().numpy().ravel()[0])
    
    path = f'{args["env_name"]}_weights.bin'
    weights = np.concatenate(weights)
    weights.tofile(path)
    print(f'Saved {len(weights)} weights to {path}')

def autotune(args=None, env_name=None, vecenv=None, policy=None):
    package = args['package']
    module_name = 'pufferlib.ocean' if package == 'ocean' else f'pufferlib.environments.{package}'
    env_module = importlib.import_module(module_name)
    env_name = args['env_name']
    make_env = env_module.env_creator(env_name)
    pufferlib.vector.autotune(make_env, batch_size=args['train']['env_batch_size'])
 
def load_env(env_name, args):
    package = args['package']
    module_name = 'pufferlib.ocean' if package == 'ocean' else f'pufferlib.environments.{package}'
    env_module = importlib.import_module(module_name)
    make_env = env_module.env_creator(env_name)
    return pufferlib.vector.make(make_env, env_kwargs=args['env'], **args['vec'])

def load_policy(args, vecenv, env_name=''):
    package = args['package']
    module_name = 'pufferlib.ocean' if package == 'ocean' else f'pufferlib.environments.{package}'
    env_module = importlib.import_module(module_name)

    device = args['train']['device']
    policy_cls = getattr(env_module.torch, args['policy_name'])
    policy = policy_cls(vecenv.driver_env, **args['policy'])

    rnn_name = args['rnn_name']
    if rnn_name is not None:
        rnn_cls = getattr(env_module.torch, args['rnn_name'])
        policy = rnn_cls(vecenv.driver_env, policy, **args['rnn'])

    policy = policy.to(device)

    load_id = args['load_id']
    if load_id is not None:
        if args['neptune']:
            path = NeptuneLogger(args, load_id, mode='read-only').download()
        elif args['wandb']:
            path = WandbLogger(args, load_id).download()
        else:
            raise pufferlib.APIUsageError('No run id provided for eval')

        state_dict = torch.load(path, map_location=device)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        policy.load_state_dict(state_dict)

    load_path = args['load_model_path']
    if load_path == 'latest':
        load_path = max(glob.glob(f"experiments/{env_name}*.pt"), key=os.path.getctime)

    if load_path is not None:
        state_dict = torch.load(load_path, map_location=device)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        policy.load_state_dict(state_dict)
        #state_path = os.path.join(*load_path.split('/')[:-1], 'state.pt')
        #optim_state = torch.load(state_path)['optimizer_state_dict']
        #pufferl.optimizer.load_state_dict(optim_state)

    return policy

def load_config(env_name, parser=None):
    puffer_dir = os.path.dirname(os.path.realpath(__file__))
    puffer_config_dir = os.path.join(puffer_dir, 'config/**/*.ini')
    puffer_default_config = os.path.join(puffer_dir, 'config/default.ini')
    if env_name == 'default':
        p = configparser.ConfigParser()
        p.read(puffer_default_config)
    else:
        for path in glob.glob(puffer_config_dir, recursive=True):
            p = configparser.ConfigParser()
            p.read([puffer_default_config, path])
            if env_name in p['base']['env_name'].split(): break
        else:
            raise pufferlib.APIUsageError('No config for env_name {}'.format(env_name))

    return process_config(p, parser=parser)

def load_config_file(file_path, fill_in_default=True, parser=None):
    if not os.path.exists(file_path):
        raise pufferlib.APIUsageError('No config file found')

    config_paths = [file_path]

    if fill_in_default:
        puffer_dir = os.path.dirname(os.path.realpath(__file__))
        # Process the puffer defaults first
        config_paths.insert(0, os.path.join(puffer_dir, 'config/default.ini'))

    p = configparser.ConfigParser()
    p.read(config_paths)

    return process_config(p, parser=parser)

def make_parser():
    '''Creates the argument parser with default PufferLib arguments.'''
    parser = argparse.ArgumentParser(formatter_class=RichHelpFormatter, add_help=False)
    parser.add_argument('--load-model-path', type=str, default=None,
        help='Path to a pretrained checkpoint')
    parser.add_argument('--load-id', type=str,
        default=None, help='Kickstart/eval from from a finished Wandb/Neptune run')
    parser.add_argument('--render-mode', type=str, default='auto',
        choices=['auto', 'human', 'ansi', 'rgb_array', 'raylib', 'None'])
    parser.add_argument('--save-frames', type=int, default=0)
    parser.add_argument('--gif-path', type=str, default='eval.gif')
    parser.add_argument('--fps', type=float, default=15)
    parser.add_argument('--max-runs', type=int, default=200, help='Max number of sweep runs')
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--wandb-project', type=str, default='pufferlib')
    parser.add_argument('--wandb-group', type=str, default='debug')
    parser.add_argument('--neptune', action='store_true', help='Use neptune for logging')
    parser.add_argument('--neptune-name', type=str, default='pufferai')
    parser.add_argument('--neptune-project', type=str, default='ablations')
    parser.add_argument('--no-model-upload', action='store_true', help='Do not upload models to wandb or neptune')
    parser.add_argument('--local-rank', type=int, default=0, help='Used by torchrun for DDP')
    parser.add_argument('--tag', type=str, default=None, help='Tag for experiment')
    return parser

def process_config(config, parser=None):
    if parser is None:
        parser = make_parser()

    parser.description = f':blowfish: PufferLib [bright_cyan]{pufferlib.__version__}[/]' \
        ' demo options. Shows valid args for your env and policy'

    def parse_bool(value):
        """Robust bool parser for CLI/config overrides."""
        if isinstance(value, bool):
            return value
        s = str(value).strip().lower()
        if s in ('1', 'true', 't', 'yes', 'y', 'on'):
            return True
        if s in ('0', 'false', 'f', 'no', 'n', 'off'):
            return False
        raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')

    def auto_type(value):
        """Type inference for numeric args that use 'auto' as a default value"""
        if value == 'auto': return value
        if value.isnumeric(): return int(value)
        return float(value)

    for section in config.sections():
        for key in config[section]:
            try:
                value = ast.literal_eval(config[section][key])
            except:
                value = config[section][key]

            fmt = f'--{key}' if section == 'base' else f'--{section}.{key}'
            parser.add_argument(
                fmt.replace('_', '-'),
                default=value,
                type=parse_bool if isinstance(value, bool) else (auto_type if value == 'auto' else type(value))
            )

    parser.add_argument('-h', '--help', default=argparse.SUPPRESS,
        action='help', help='Show this help message and exit')

    # Unpack to nested dict
    parsed = vars(parser.parse_args())
    args = defaultdict(dict)
    for key, value in parsed.items():
        next = args
        for subkey in key.split('.'):
            prev = next
            next = next.setdefault(subkey, {})

        prev[subkey] = value

    args['train']['env'] = args['env_name'] or ''  # for trainer dashboard
    args['train']['use_rnn'] = args['rnn_name'] is not None
    return args

def main():
    err = 'Usage: puffer [train, eval, sweep, autotune, profile, export] [env_name] [optional args]. --help for more info'
    if len(sys.argv) < 3:
        raise pufferlib.APIUsageError(err)

    mode = sys.argv.pop(1)
    env_name = sys.argv.pop(1)
    if mode == 'train':
        train(env_name=env_name)
    elif mode == 'eval':
        eval(env_name=env_name)
    elif mode == 'sweep':
        sweep(env_name=env_name)
    elif mode == 'autotune':
        autotune(env_name=env_name)
    elif mode == 'profile':
        profile(env_name=env_name)
    elif mode == 'export':
        export(env_name=env_name)
    else:
        raise pufferlib.APIUsageError(err)

if __name__ == '__main__':
    main()
