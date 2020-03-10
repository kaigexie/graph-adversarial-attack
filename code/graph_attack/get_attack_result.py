from __future__ import print_function

import os
import sys
import numpy as np
import torch
import networkx as nx
import random
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from copy import deepcopy

from datetime import datetime
import cPickle as cp
sys.path.append('%s/../common' % os.path.dirname(os.path.realpath(__file__)))
from graph_embedding import S2VGraph
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from q_net import NStepQNet, QNet, greedy_actions
sys.path.append('%s/../common' % os.path.dirname(os.path.realpath(__file__)))
from cmd_args import cmd_args

from rl_common import GraphEdgeEnv, local_args, load_graphs, test_graphs, load_base_model, attackable, get_supervision
from nstep_replay_mem import NstepReplayMem

sys.path.append('%s/../graph_classification' % os.path.dirname(os.path.realpath(__file__)))
from graph_common import loop_dataset

class Agent(object):
    def __init__(self, g_list, test_g_list, env):
        self.g_list = g_list
        if test_g_list is None:
            self.test_g_list = g_list
        else:
            self.test_g_list = test_g_list
        self.mem_pool = NstepReplayMem(memory_size=50000, n_steps=2)
        self.env = env
        # self.net = QNet()
        self.net = NStepQNet(2)
        self.old_net = NStepQNet(2)
        if cmd_args.ctx == 'gpu':
            self.net = self.net.cuda()
            self.old_net = self.old_net.cuda()
        self.eps_start = 1.0
        self.eps_end = 1.0
        self.eps_step = 10000
        self.burn_in = 100        
        self.step = 0

        self.best_eval = None
        self.pos = 0
        self.sample_idxes = list(range(len(g_list)))
        random.shuffle(self.sample_idxes)
        self.take_snapshot()

    def take_snapshot(self):
        self.old_net.load_state_dict(self.net.state_dict())

    def make_actions(self, time_t, greedy=False):
        self.eps = self.eps_end + max(0., (self.eps_start - self.eps_end)
                * (self.eps_step - max(0., self.step)) / self.eps_step)

        if random.random() < self.eps and not greedy:
            actions = self.env.uniformRandActions()
        else:
            cur_state = self.env.getStateRef()
            actions, _, _ = self.net(time_t, cur_state, None, greedy_acts=True)
            actions = list(actions.cpu().numpy())
            
        return actions

    def run_simulation(self):
        if (self.pos + 1) * cmd_args.batch_size > len(self.sample_idxes):
            self.pos = 0
            random.shuffle(self.sample_idxes)

        selected_idx = self.sample_idxes[self.pos * cmd_args.batch_size : (self.pos + 1) * cmd_args.batch_size]
        self.pos += 1
        self.env.setup([self.g_list[idx] for idx in selected_idx])

        t = 0
        while not env.isTerminal():
            list_at = self.make_actions(t)
            list_st = self.env.cloneState()
            self.env.step(list_at)

            assert (env.rewards is not None) == env.isTerminal()
            if env.isTerminal():
                rewards = env.rewards
                s_prime = None
            else:
                rewards = np.zeros(len(list_at), dtype=np.float32)
                s_prime = self.env.cloneState()

            self.mem_pool.add_list(list_st, list_at, rewards, s_prime, [env.isTerminal()] * len(list_at), t)
            t += 1

    def eval(self):
        self.env.setup(deepcopy(self.test_g_list))
        t = 0
        while not self.env.isTerminal():
            list_at = self.make_actions(t, greedy=True)
            self.env.step(list_at)
            t += 1
        test_loss = loop_dataset(env.g_list, env.classifier, list(range(len(env.g_list))))
        print('\033[93m average test: loss %.5f acc %.5f\033[0m' % (test_loss[0], test_loss[1]))

        if cmd_args.phase == 'train' and self.best_eval is None or test_loss[1] < self.best_eval:
            print('----saving to best attacker since this is the best attack rate so far.----')
            torch.save(self.net.state_dict(), cmd_args.save_dir + '/epoch-best.model')
            with open(cmd_args.save_dir + '/epoch-best.txt', 'w') as f:
                f.write('%.4f\n' % test_loss[1])
            self.best_eval = test_loss[1]

        reward = np.mean(self.env.rewards)
        print(reward)
        return reward, test_loss[1]

    def train(self):
        log_out = open(cmd_args.logfile, 'w', 0)
        pbar = tqdm(range(self.burn_in), unit='batch')
        for p in pbar:
            self.run_simulation()
        pbar = tqdm(range(local_args.num_steps), unit='steps')
        optimizer = optim.Adam(self.net.parameters(), lr=cmd_args.learning_rate)
        for self.step in pbar:

            self.run_simulation()

            if self.step % 100 == 0:
                self.take_snapshot()
            if self.step % 100 == 0:
                r, acc = self.eval()
                log_out.write('%d %.6f %.6f\n' % (self.step, r, acc))

            cur_time, list_st, list_at, list_rt, list_s_primes, list_term = self.mem_pool.sample(batch_size=cmd_args.batch_size)

            list_target = torch.Tensor(list_rt)
            if cmd_args.ctx == 'gpu':
                list_target = list_target.cuda()

            cleaned_sp = []
            nonterms = []
            for i in range(len(list_st)):
                if not list_term[i]:
                    cleaned_sp.append(list_s_primes[i])
                    nonterms.append(i)

            if len(cleaned_sp):
                _, _, banned = zip(*cleaned_sp)
                _, q_t_plus_1, prefix_sum_prime = self.old_net(cur_time + 1, cleaned_sp, None)
                _, q_rhs = greedy_actions(q_t_plus_1, prefix_sum_prime, banned)
                list_target[nonterms] = q_rhs
            
            # list_target = get_supervision(self.env.classifier, list_st, list_at)
            list_target = Variable(list_target.view(-1, 1))

            _, q_sa, _ = self.net(cur_time, list_st, list_at)

            loss = F.mse_loss(q_sa, list_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description('exp: %.5f, loss: %0.5f' % (self.eps, loss) )

        log_out.close()

    def _eval_data(self):

        data_dir = 'data_dir'
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        dir_name = datetime.now().strftime("%m-%d-%H-%M-%S")
        cur_data_dir = os.path.join(data_dir, dir_name)
        os.mkdir(cur_data_dir)

        print('===== Saving original graphs ... =====')
        with open('%s/original_glist.pkl' % (cur_data_dir, ), 'wb') as f:
            original_graphs = [g.to_networkx() for g in self.test_g_list]
            cp.dump(original_graphs, f)
        print('===== Saved original graphs. =====')

        self.env.setup(deepcopy(self.test_g_list))
        t = 0
        while not self.env.isTerminal():
            list_at = self.make_actions(t, greedy=True)
            self.env.step(list_at)
            t += 1
        test_loss = loop_dataset(env.g_list, env.classifier, list(range(len(env.g_list))))
        print('\033[93m average test: loss %.5f acc %.5f\033[0m' % (test_loss[0], test_loss[1]))

        print('===== Saving new graphs ... =====')
        with open('%s/new_glist.pkl' % (cur_data_dir, ), 'wb') as f:
            new_graphs = [g.to_networkx() for g in env.g_list]
            cp.dump(new_graphs, f)
        print('===== Saved new graphs. =====')
        return original_graphs, new_graphs, cur_data_dir

    def _display(original_graphs, new_graphs, g_num=1):
        pass

def load_pkl(fname, num_graph):
    original_num_graph = num_graph
    if num_graph == 1:
        num_graph = 100

    g_list = []
    with open(fname, 'rb') as f:
        for i in range(num_graph):
            g = cp.load(f)
            g_list.append(g)

    if original_num_graph == 1:
        return [random.choice(g_list)]

    return g_list

def _load_graphs(graph_fname, graph_type, num_graph):
    cur_list = load_pkl('%s/%s' % (cmd_args.data_folder, graph_fname), num_graph)
    assert len(cur_list) == num_graph
    test_glist = [S2VGraph(g, graph_type) for g in cur_list]
    label_map = {i: i-1 for i in range(cmd_args.min_c, cmd_args.max_c+1)}
    print('# test:', len(test_glist))
    return label_map, test_glist

def _display(original_glist, new_glist, cur_data_dir):

    def _display_individual(idx, g1, g2):
        # print(len(g1.nodes()))
        # print(len(g2.nodes()))
        # print(len(g1.edges()))
        # print(len(g2.edges()))
        pos = nx.random_layout(g1)
        pos = nx.spring_layout(g1, dim=2, pos=pos)
        e_color_g1 = ['pink'] * len(g1.edges())
        plt.subplot(211)
        nx.draw(g1, node_size=3, node_color='green', edge_color=e_color_g1, pos=pos)
        #plt.show()

        e_color_g2 = ['pink'] * len(g2.edges())
        new_edges = set(g2.edges()) - set(g1.edges())
        for e in new_edges:
            e_color_g2[g2.edges().index(e)] = 'blue'
        plt.subplot(212)
        nx.draw(g2, node_size=3, node_color='green', edge_color=e_color_g2, pos=pos)
        #plt.show()
        plt.savefig('%s/comparison_%s.pdf' % (cur_data_dir, idx))
        plt.clf()

    # original_nlen = [len(g.nodes()) for g in original_glist]
    # new_nlen = [len(g.nodes()) for g in new_glist]
    # original_elen = [len(g.edges()) for g in original_glist]
    # new_elen = [len(g.edges()) for g in new_glist]
    # for x, y in zip(original_nlen, new_nlen):
    #     print(x, y)
    # res = 0
    # for x, y in zip(original_elen, new_elen):
    #     print(x, y)
    #     if x != y:
    #         res += 1
    # print(res)
    # for g1, g2 in zip(original_glist, new_glist):
    #     set1 = set(g1.edges())
    #     set2 = set(g2.edges())
    #     print('set1 - set2 =', set1-set2)
    #     print('set2 - set1 =', set2-set1)
    print('===== Comparison figure being saved ... =====')
    for idx, (g1, g2) in enumerate(zip(original_glist, new_glist)):
        if idx == 10:
            break
        _display_individual(idx, g1, g2)
    print('===== Comparison figure saved. =====')

    

if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    graph_fname = cmd_args.graph_fname
    graph_type = cmd_args.graph_type
    num_graph = cmd_args.num_graph
    label_map, g_list = _load_graphs(graph_fname, graph_type, num_graph)
    base_classifier = load_base_model(label_map, g_list)
    env = GraphEdgeEnv(base_classifier, n_edges = 1)
    agent = Agent(g_list, None, env)
    
    agent.net.load_state_dict(torch.load(cmd_args.save_dir + '/epoch-best.model'))
    original_graphs, new_graphs, cur_data_dir = agent._eval_data()

    if cmd_args.display_mode == 1:
        _display(original_graphs, new_graphs, cur_data_dir)
        # env.setup([g_list[idx] for idx in selected_idx])
        # t = 0
        # while not env.isTerminal():
        #     policy_net = net_list[t]
        #     t += 1            
        #     batch_graph, picked_nodes = env.getState()
        #     log_probs, prefix_sum = policy_net(batch_graph, picked_nodes)
        #     actions = env.sampleActions(torch.exp(log_probs).data.cpu().numpy(), prefix_sum.data.cpu().numpy(), greedy=True)
        #     env.step(actions)

        # test_loss = loop_dataset(env.g_list, base_classifier, list(range(len(env.g_list))))
        # print('\033[93maverage test: loss %.5f acc %.5f\033[0m' % (test_loss[0], test_loss[1]))
        
        # print(np.mean(avg_rewards), np.mean(env.rewards))
