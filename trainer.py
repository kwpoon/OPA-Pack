import os
import numpy as np
import torch
from tools import get_mask_from_state
from tqdm import trange
from collections import deque
from tensorboardX import SummaryWriter
import torch.multiprocessing as mp
np.set_printoptions(threshold=np.inf)
import time
from training_runtime import HierarchicalRuntime, write_json

# Distributed training for online packing policy
def learningPara(T, priority_weight_increase, model_save_path, dqn, mem, timeStr, args, counter, lock, sub_time_str):
    log_writer_path = './logs/runs/{}'.format('IR-' + timeStr + '-loss')
    if not os.path.exists(log_writer_path):
      os.makedirs(log_writer_path)
    writer = SummaryWriter(log_writer_path)
    targetCounter = T
    checkCounter = T
    logCounter = T
    timeStep = T
    if args.device.type.lower() != 'cpu':
        torch.cuda.set_device(args.device)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = args.enable_cudnn
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    print('Distributed Training Start')
    torch.set_num_threads(1)
    while True:
        if not lock.value:
            for i in range(len(mem)):
                mem[i].priority_weight = min(mem[i].priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1

            dqn.reset_noise()
            loss = dqn.learn(mem)  # Train with n-step distributional double-Q learning

            # Update target network
            if timeStep - targetCounter >= args.target_update:
                targetCounter = timeStep
                dqn.update_target_net()

            if timeStep % args.checkpoint_interval == 0:
                    sub_time_str = time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))

            # Checkpoint the network #
            if (args.checkpoint_interval != 0) and (timeStep - checkCounter >= args.save_interval):
                checkCounter = timeStep
                dqn.save(model_save_path, 'checkpoint{}.pt'.format(sub_time_str))

            if timeStep - logCounter >= args.print_log_interval:
                logCounter = timeStep
                writer.add_scalar("Training/Value loss", loss.mean().item(), logCounter)

            timeStep += 1
        else:
            time.sleep(0.5)

# Distributed training for buffered packing policy
def learningParaHierachical(T, priority_weight_increase, model_save_path, orderDQN, locDQN,
                 orderMem, locMem, timeStr, args, counter, lock, sub_time_str, stop_event):
    log_dir = os.path.join(args.output_dir, 'tensorboard', 'learner') if args.output_dir else './logs/runs/IR-' + timeStr + '-loss'
    writer = SummaryWriter(log_dir)
    torch.cuda.set_device(args.device)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    runtime = HierarchicalRuntime(args, (orderDQN, locDQN), model_save_path)
    updates = args.start_update
    print('Distributed hierarchical learner started', flush=True)
    try:
        while not stop_event.is_set():
            if mp.parent_process() is not None and not mp.parent_process().is_alive():
                raise RuntimeError('Training parent exited; stopping learner')
            if lock.value or any(not m.transitions.full.value and m.transitions.index.value <= m.n for m in orderMem + locMem):
                stop_event.wait(0.1)
                continue
            for memory in orderMem + locMem:
                memory.priority_weight = min(memory.priority_weight + priority_weight_increase, 1)
            orderDQN.reset_noise()
            orderLoss = orderDQN.learn(orderMem)
            locDQN.reset_noise()
            locLoss = locDQN.learn(locMem)
            updates += 1
            if updates % args.target_update == 0:
                orderDQN.update_target_net()
                locDQN.update_target_net()
            if updates % args.print_log_interval == 0 or updates == 1:
                writer.add_scalar('Training/Value loss', locLoss.mean().item(), updates)
                writer.add_scalar('Training/Order value loss', orderLoss.mean().item(), updates)
                writer.flush()
                print(f'Learner update {updates}: order_loss={orderLoss.mean().item():.6f}, location_loss={locLoss.mean().item():.6f}', flush=True)
            runtime.tick(updates, counter.value)
    finally:
        runtime.close()
        writer.close()


class trainer(object):
    def __init__(self, writer, timeStr, dqn, mem):
        self.writer = writer
        self.timeStr = timeStr
        self.dqn = dqn
        self.mem = mem

    def train_q_value(self, envs, args):
        priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)
        sub_time_str = time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))

        model_save_path = os.path.join(args.model_save_path, self.timeStr)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        if args.save_memory_path is not None:
            memory_save_path = os.path.join(model_save_path, args.save_memory_path)
            if not os.path.exists(memory_save_path):
                os.makedirs(memory_save_path)


        episode_rewards = deque(maxlen=10)
        episode_ratio = deque(maxlen=10)
        episode_counter = deque(maxlen=10)
        state = envs.reset()
        reward_clip = torch.ones((args.num_processes, 1)) * args.reward_clip
        R, loss = 0, 0
        if args.distributed:
            counter= mp.Value('i', 0)
            lock = mp.Value('b', False)
        # Training loop
        self.dqn.train()
        for T in trange(1, args.T_max + 1):

            if T % args.replay_frequency == 0 and not args.distributed:
                self.dqn.reset_noise()  # Draw a new set of noisy weights

            mask = get_mask_from_state(state, args, args.bufferSize)
            action = self.dqn.act(state, mask)  # Choose an action greedily (with noisy weights)


            next_state, reward, done, infos = envs.step(action.cpu().numpy())  # Step

            validSample = []
            for _ in range(len(infos)):
                validSample.append(infos[_]['Valid'])
                if done[_] and infos[_]['Valid']:
                    if 'reward' in infos[_].keys():
                        episode_rewards.append(infos[_]['reward'])
                    else:
                        episode_rewards.append(infos[_]['episode']['r'])
                    if 'ratio' in infos[_].keys():
                        episode_ratio.append(infos[_]['ratio'])
                    if 'counter' in infos[_].keys():
                        episode_counter.append(infos[_]['counter'])


            if args.reward_clip > 0:
                reward = torch.maximum(torch.minimum(reward, reward_clip), -reward_clip)  # Clip rewards

            for i in range(len(state)):
                if validSample[i]:
                    self.mem[i].append(state[i], action[i], reward[i], done[i])  # Append transition to memory

            if args.distributed:
                counter.value = T
                if T == args.learn_start:
                    learningProcess = mp.Process(target=learningPara, args=(T, priority_weight_increase, model_save_path, self.dqn, self.mem, self.timeStr, args, counter, lock, sub_time_str))
                    learningProcess.start()
            else:
                # Train and test
                if T >= args.learn_start:
                    for i in range(len(self.mem)):
                        self.mem[i].priority_weight = min(self.mem[i].priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1

                    if T % args.replay_frequency == 0:
                        loss = self.dqn.learn(self.mem)  # Train with n-step distributional double-Q learning
                    # Update target network
                    if T % args.target_update == 0:
                        self.dqn.update_target_net()

                    # Checkpoint the network #
                    if (args.checkpoint_interval != 0) and (T % args.save_interval == 0):
                        if T % args.checkpoint_interval == 0:
                            sub_time_str = time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))
                        self.dqn.save(model_save_path, 'checkpoint{}.pt'.format(sub_time_str))

                    if T % args.print_log_interval == 0:
                        self.writer.add_scalar("Training/Value loss",  loss.mean().item(), T)

            state = next_state
            if len(episode_rewards)!= 0:
                self.writer.add_scalar('Metric/Reward mean', np.mean(episode_rewards), T)
                self.writer.add_scalar('Metric/Reward max', np.max(episode_rewards), T)
                self.writer.add_scalar('Metric/Reward min', np.min(episode_rewards), T)
            if len(episode_ratio) != 0:
                self.writer.add_scalar('Metric/Ratio', np.mean(episode_ratio), T)
            if len(episode_counter) != 0:
                self.writer.add_scalar('Metric/Length', np.mean(episode_counter), T)

class trainer_hierarchical(object):
    def __init__(self, writer, timeStr, DQNs, MEMs):
        self.writer = writer
        self.timeStr = timeStr
        self.orderDQN, self.locDQN = DQNs
        self.orderMem, self.locMem = MEMs
        self.learningProcess = None
        self.stop_event = mp.Event()

    def train_q_value(self, envs, args):
        priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)
        actionNum = args.action_space
        sub_time_str = time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))

        model_save_path = os.path.join(args.output_dir, 'checkpoints') if args.output_dir else os.path.join(args.model_save_path, self.timeStr)
        args.training_started_at = time.time()
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        if args.save_memory_path is not None:
            memory_save_path = os.path.join(model_save_path, args.save_memory_path)
            if not os.path.exists(memory_save_path):
                os.makedirs(memory_save_path)

        episode_rewards = deque(maxlen=10)
        episode_ratio = deque(maxlen=10)
        episode_counter = deque(maxlen=10)
        orderState = envs.reset()
        reward_clip = torch.ones((args.num_processes, 1)) * args.reward_clip
        R, orderLoss, locLoss = 0, 0, 0
        if args.distributed:
            counter= mp.Value('i', 0)
            lock = mp.Value('b', False)

        self.orderDQN.eval()
        self.locDQN.eval()

        measured_step_seconds = deque(maxlen=32)
        valid_transitions = 0
        # Training loop
        self.orderDQN.train()
        self.locDQN.train()
        for T in trange(1, args.T_max + 1):
            if self.stop_event.is_set():
                break
            step_started = time.perf_counter()
            if self.learningProcess is not None and not self.learningProcess.is_alive():
                if self.stop_event.is_set():
                    break
                raise RuntimeError(f'Learner exited unexpectedly: {self.learningProcess.exitcode}')
            if args.training_hours and time.time() - args.training_started_at >= args.training_hours * 3600:
                break

            if T % args.replay_frequency == 0:
                self.orderDQN.reset_noise()  # Draw a new set of noisy weights
                self.locDQN.reset_noise()  # Draw a new set of noisy weights
            orderAction = self.orderDQN.act(orderState, None)
            locState = envs.get_action_candidates(orderAction.cpu().numpy())
            locState = torch.from_numpy(np.array(locState)).float().to(args.device)
            if not args.selectedAction:
                if args.heightMap and not args.physics:
                    locMask = locState[:, 0:args.action_space].reshape(-1, actionNum)
                else:
                    if args.elementWise:
                        locMask = locState[:, (args.packed_holder + 1) * args.objVecLen : (args.packed_holder + 1) * args.objVecLen + actionNum].reshape(-1, actionNum)
                    else:
                        locMask = locState[:, args.objVecLen : args.objVecLen + actionNum].reshape(-1, actionNum)
            else:
                locMask = locState[:, 0 : args.selectedAction * 5].reshape(args.num_processes, args.selectedAction, 5)[:,:,-1]

            locAction = self.locDQN.act(locState, locMask)  # Choose an action greedily (with noisy weights)
            next_order_state, reward, done, infos = envs.step(locAction.cpu().numpy())  # Step

            validSample = []
            for _ in range(len(infos)):
                validSample.append(infos[_]['Valid'])
                if done[_] and infos[_]['Valid']:
                    if 'reward' in infos[_].keys():
                        episode_rewards.append(infos[_]['reward'])
                    else:
                        episode_rewards.append(infos[_]['episode']['r'])
                    if 'ratio' in infos[_].keys():
                        episode_ratio.append(infos[_]['ratio'])
                    if 'counter' in infos[_].keys():
                        episode_counter.append(infos[_]['counter'])

            if args.reward_clip > 0:
                reward = torch.maximum(torch.minimum(reward, reward_clip), -reward_clip)  # Clip rewards

            for i in range(len(orderState)):
                if validSample[i]:
                    self.orderMem[i].append(orderState[i], orderAction[i], reward[i], done[i])  # Append transition to memory
                    self.locMem[i].append(locState[i], locAction[i], reward[i], done[i])  # Append transition to memory

            # todo: sample outside and update priorities uniformly, or maintain their memory seperately
            if args.distributed:
                counter.value = T
                if T == args.learn_start:
                    self.learningProcess = mp.Process(target=learningParaHierachical, args=(T, priority_weight_increase, model_save_path, self.orderDQN, self.locDQN,
                                                                            self.orderMem, self.locMem, self.timeStr, args, counter, lock, sub_time_str, self.stop_event))
                    self.learningProcess.start()
            else:
                if T >= args.learn_start:
                    for i in range(args.num_processes):
                        self.orderMem[i].priority_weight = min(self.orderMem[i].priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1
                        self.locMem[i].priority_weight = min(self.locMem[i].priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1

                    if T % args.replay_frequency == 0:
                        orderLoss = self.orderDQN.learn(self.orderMem)  # Train with n-step distributional double-Q learning
                        locLoss = self.locDQN.learn(self.locMem)  # Train with n-step distributional double-Q learning

                    # Update target network
                    if T % args.target_update == 0:
                        self.orderDQN.update_target_net()
                        self.locDQN.update_target_net()

                    # Checkpoint the network #
                    if (args.checkpoint_interval != 0) and (T % args.save_interval == 0):
                        if T % args.checkpoint_interval == 0:
                            sub_time_str = time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))
                        self.orderDQN.save(model_save_path, 'orderCheckpoint_{}.pt'.format(sub_time_str))
                        self.locDQN.save(model_save_path, 'locCheckpoint_{}.pt'.format(sub_time_str))

                    if T % args.print_log_interval == 0:
                        self.writer.add_scalar("Training/Value loss",  locLoss.mean().item(), T)
                        self.writer.add_scalar("Training/Order value loss",  orderLoss.mean().item(), T)

            orderState = next_order_state
            valid_transitions += sum(validSample)
            if T > args.learn_start + 8:
                measured_step_seconds.append(time.perf_counter() - step_started)
            if args.output_dir and (T % 8 == 0 or T == args.T_max):
                mean_seconds = float(np.mean(measured_step_seconds)) if measured_step_seconds else None
                write_json(os.path.join(args.output_dir, 'actor_progress.json'), {
                    'dataset': args.dataset, 'environments': args.num_processes, 'vector_steps': T,
                    'attempts': T * args.num_processes, 'valid_transitions': int(valid_transitions),
                    'elapsed_seconds': time.time() - args.training_started_at,
                    'measured_vector_steps': len(measured_step_seconds),
                    'mean_vector_step_seconds': mean_seconds,
                    'attempts_per_second': args.num_processes / mean_seconds if mean_seconds else None,
                    'hours_for_800000_attempts': 800000 * mean_seconds / args.num_processes / 3600 if mean_seconds else None,
                    'updated_at': time.time(),
                })

            if len(episode_rewards)!= 0:
                self.writer.add_scalar('Metric/Reward mean', np.mean(episode_rewards), T)
                self.writer.add_scalar('Metric/Reward max', np.max(episode_rewards), T)
                self.writer.add_scalar('Metric/Reward min', np.min(episode_rewards), T)
            if len(episode_ratio) != 0:
                self.writer.add_scalar('Metric/Ratio', np.mean(episode_ratio), T)
            if len(episode_counter) != 0:
                self.writer.add_scalar('Metric/Length', np.mean(episode_counter), T)


    def close(self):
        self.stop_event.set()
        if self.learningProcess is not None:
            self.learningProcess.join(timeout=960)
            if self.learningProcess.is_alive():
                self.learningProcess.terminate()
                self.learningProcess.join(timeout=30)
            if self.learningProcess.exitcode != 0:
                raise RuntimeError(f'Learner failed with exit code {self.learningProcess.exitcode}')
