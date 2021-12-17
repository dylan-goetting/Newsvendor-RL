from numpy.lib.index_tricks import AxisConcatenator
from stable_baselines3 import DQN
import os
from cogbidders.base_bidder import BaseBidAgent
from cogsimulators.base_exchange import BidResponse
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from bisect import bisect
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.manifold import TSNE
import pdb
import matplotlib.pyplot as plt
import json

class SARS(object):
    """
    Transition class that keeps track of timestamps as well as SARS
    """

    def __init__(self, s0, t0, a, r, s1, t1, d):
        self.s0 = s0
        self.t0 = t0
        self.a = a
        self.r = r
        self.s1 = s1
        self.t1 = t1
        self.d = d


class CriticNet(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, output_size)

        self.fc5 = nn.Linear(input_size, 64)
        self.fc6 = nn.Linear(64, 32)
        self.fc7 = nn.Linear(32, 16)
        self.fc8 = nn.Linear(16, output_size)

        self.optimizer = optim.Adam(self.parameters())
        self.criterion = nn.MSELoss()

    def forward(self, states, actions):
        x = torch.cat((states, actions), 1)
        print(x.size())
        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(q1))
        q1 = F.relu(self.fc3(q1))
        q1 = self.fc4(q1)

        q2 = F.relu(self.fc1(x))
        q2 = F.relu(self.fc2(q2))
        q2 = F.relu(self.fc3(q2))
        q2 = self.fc4(q2)

        return q1, q2

    def q1(self, states, actions):
        x = torch.cat((states, actions), 1)
        q = F.relu(self.fc1(x))
        q = F.relu(self.fc2(q))
        q = F.relu(self.fc3(q))

        return self.fc4(q)


class ActorNet(nn.Module):
    # TODO: limit the action range with a sigmoid func or something
    def __init__(self, input_size, output_size=1):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, output_size)

        self.optimizer = optim.Adam(self.parameters())
        # self.criterion = 

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        
        return x


class sb3DDPGBidAgent(BaseBidAgent):
    """
    as of now, params are the union of what the SB3 class takes in and what the original bidding agent takes it, with a
    few overlaps. Super method initialized all the SB3 attributes and data, which will probably be useful later. Rest of
    the __init__ code is taken from the original agent. There are a few overlaps that have not yet been cleaned
    """

    def __init__(self, experience_memory, experiment_name='default', total_budget_usd=1000, max_simulation_time=5000,
                 bid_options_microusd_range=[0, 5000], conv_reward_microusd=20000000, mini_batch_size=1024,
                 gradient_threshold=1, user_types=None, num_of_user_types = 2, gamma=0.99,
                 noise_mean = 0.0, noise_std = 0.05, buffer_priming_period = 1500, reward_scaler=100000, attribution_window=float('inf'),
                 clip=0.25, experiment_number=1, u2v_dimension=10, u2v_window=10, u2v_iterations=100, use_u2v=True, 
                 use_site_info=False, num_of_sites=7, site_to_index = None, use_s2v=False, use_time_of_day=False, tau=0.005,
                 use_multiplier_info=True, userIDMultiplierDict = None, multipliers = [3], multiplier_range=[1,3], budget_per_1000=100,
                 evaluate=False, exploration_stop = 4500):

        self.spend_start = 0
        self.multiplier_range = multiplier_range
        self.budget_per_1000 = budget_per_1000
        self.current_multiplier = sum(multiplier_range)/2
        self.bid_prices = defaultdict(lambda: [])
        self.plots = [True] * 4000

        self.evaluate = evaluate
        self.bid_times = []
        self.experiment_name = experiment_name
        self.experiment_number = experiment_number  # multiple number of experiments for the same parameters.
        self.attribution_window = attribution_window
        self.last_spend = 0
        self.missed_conversions = 0
        self.use_time_of_day = use_time_of_day
        self.state_size = 1
        self.use_s2v = use_s2v
        self.use_u2v = use_u2v
        self.exploration_stop = exploration_stop

        # Logs and statistics
        self._user_web_logs = defaultdict(list)
        self._user_stats = defaultdict(lambda: [[], [], [], []])  # for each userid key, a list of bidreqtimes, wintimes, convtimes, bidprices
        self._user_site_stats = defaultdict(lambda: [[] for _ in range(3*num_of_sites)]) # for each userid key, a list of bidreqtimes, wintimes, convtimes according to the website they are on.
        self._user_multiplier_stats = defaultdict(lambda: [[] for _ in range(3*len(self.multipliers))]) # for each userid key, a list of bidreqtimes, wintimes, convtimes according to the multiplier which they have.
        self._total_spend_usd = 0
        self._conversions = []

        # gpu or cpu.
        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")

        # campaign information
        self._max_simulation_time = max_simulation_time
        self._total_budget_usd = total_budget_usd
        self.num_of_user_types = num_of_user_types

        # u2v training information
        self.u2v_dims = u2v_dimension
        self.u2v_win = u2v_window  # window of urls for training u2v
        self.u2v_iterations = u2v_iterations  # iterations for training u2v
        if self.use_u2v: self.state_size += u2v_dimension  # number of u2v dimensions
        else: self.state_size += self.num_of_user_types

        # site information
        self.use_site_info = use_site_info
        self.num_of_sites = num_of_sites
        self.site_to_index = site_to_index
        if self.use_site_info:
            if self.use_s2v: self.state_size += u2v_dimension
            else: self.state_size += num_of_sites
        if self.use_time_of_day: self.state_size += 1

        # reward information
        self.reward_scaler = reward_scaler
        self.userIDMultiplierDict = userIDMultiplierDict
        self.use_multiplier_info = use_multiplier_info
        self.multipliers = multipliers
        if self.use_multiplier_info: self.state_size += 1
        self.conv_reward_microusd = conv_reward_microusd
        self._bid_range = bid_options_microusd_range  # list of possible bids (currently no creative selection) AKA action space

        # DRL information
        self.gamma = gamma
        self._user_last_sars = {}  # pointer to memory of user or updating.
        self._gradient_threshold = gradient_threshold
        self._mini_batch_size = mini_batch_size
        self._n_mini_batch = 0  # number of mini batches executed so far.
        self._replay_memory = experience_memory
        self.critic_net = self._create_critic_network(is_train=True)
        self.critic_net_target = self._create_critic_network(is_train=False)
        self.actor_net = self._create_actor_network(is_train=True)
        self.actor_net_target = self._create_actor_network(is_train=False)
        self._update_target_networks(tau=1)  # Updates target to the policy's params
        self._is_bidding = False  # wait to start bidding until user2vec has enough data to begin
        self.critic_net.to(self.device)
        self.critic_net_target.to(self.device)
        self.actor_net.to(self.device)
        self.actor_net_target.to(self.device)
        self.clip = clip  # gradient clipping.
        self.buffer_priming_period = buffer_priming_period #TODO do we use this?
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.tau = tau

        # Results
        self._report_interval = 20
        self._last_report_time = 0
        self._user_types = user_types  # held out from bidder is using u2v. Used EXCLUSIVELY for reporting results
        self._reports = list()
        self._site_reports = list()
        self._multiplier_reports = list()
        self.mini_batch_loss = []

        # make directories
        if not os.path.exists(f'logs/{self.experiment_name}'):
            os.mkdir(f'logs/{self.experiment_name}')
        if not os.path.exists(f'logs/{self.experiment_name}/exp_{self.experiment_number}'):
            os.mkdir(f'logs/{self.experiment_name}/exp_{self.experiment_number}')

        if not os.path.exists(f'models/{self.experiment_name}'):
            os.mkdir(f'models/{self.experiment_name}')
        if not os.path.exists(f'models/{self.experiment_name}/exp_{self.experiment_number}'):
            os.mkdir(f'models/{self.experiment_name}/exp_{self.experiment_number}')

    def process_bid_request(self, bid_request):
        """
        essentially the 'select_action' part. Updates user stats, constructs a state, pushes to memory, and
        returns a bid with the chosen price,
        """

        t = bid_request.simulation_time
        uid = bid_request.user_id
        self._user_web_logs[uid].append(bid_request.url)  # records that the given user made this bid request
        # bid = 0
        bid_actual = 0
        creative = 0
        is_done = False
        if not self._is_bidding:  # not bidding yet and we need to build u2v vectors, so pretty much does nothing/bids 0
            self._build_user2vec_model()

        elif not self.use_u2v or uid in self.u2v_vecs:  # self.u2v_ids should be synonymous
            # construct the current state based on current observation
            time_since_last5_views = np.ones(1)
            temp = (t - np.array(self._user_stats[uid][1][-1:])) / 10000  # as percent of 10000 minutes.
            time_since_last5_views[:len(temp)] = temp
            hour = (t % 40)*0.3
            
            #to use u2v or not to use u2v.
            if self.use_u2v:
                curr_state = np.hstack([self.u2v_vecs[uid], time_since_last5_views])
            else:
                one_hot_user_type = np.zeros(self.num_of_user_types)
                one_hot_user_type[self._user_types[uid] - 1] = 1
                curr_state = np.hstack([one_hot_user_type, time_since_last5_views])

            #site information
            if self.use_site_info:
                if self.use_s2v:
                    curr_state = np.hstack([curr_state, self.s2v_vecs[bid_request.url]])
                else:
                    one_hot_site_type = np.zeros(self.num_of_sites)
                    one_hot_site_type[self.site_to_index[bid_request.url]] = 1
                    curr_state = np.hstack([curr_state, one_hot_site_type])

            #time of the day information.
            if self.use_time_of_day:
                curr_state = np.hstack([curr_state, hour])

            # do we have a different multiplier for different users?
            # if yes, then add a multiplier representation to the state
            if self.use_multiplier_info:
                self.update_multiplier(t)
                one_hot_multiplier_type = np.zeros(len(self.multipliers))
                one_hot_multiplier_type[self.multipliers.index(self.userIDMultiplierDict[uid])] = 1
                curr_state = np.hstack([curr_state, self.current_multiplier])

            # complete the last SARS of this user and add to memory
            if uid in self._user_last_sars:
                self._user_last_sars[uid].s1 = curr_state  # changing the next state from none to s1
                self._user_last_sars[uid].t1 = t  # changing the next time from none to t1
                self._replay_memory.add(self._user_last_sars[uid])  # add completed SARS to memory

            noise = 0
            if t < self.buffer_priming_period:
                bid = np.random.uniform(-1, 1)
                bid_actual = self._scale_up(bid)
            else:
                # select a bid
                bid = self.actor_net(torch.tensor(curr_state, dtype=torch.float, device=self.device))
                bid = bid.detach().numpy()
                if not self.evaluate:
                    # noise = self.noise_std*np.random.randn() + self.noise_mean
                    noise = np.random.normal(self.noise_mean, self.noise_std)
                    bid += noise
                bid = np.clip(bid, -1, 1)
                bid = float(bid[0])
                bid_actual = self._scale_up(bid)
            # prepare memory (though don't add to replay_memory because not complete yet):
            # WARNING: this assumes that base exchange will post auction results before processing next internet event.

            if t % self._report_interval == 0 and t % 400 == 0:
                self.bid_prices[t].append(bid_actual)
            if t < self._max_simulation_time and t%400 == 1 and t>1 and self.plots[t-1]:
                plt.clf()
                plt.hist(self.bid_prices[t-1], bins=100)
                plt.title(f'Bidding prices at time: {t-1}')
                plt.show()
                self.plots[t-1] = False


            is_done = (self._total_spend_usd > self._total_budget_usd) or (
                    bid_request.simulation_time > self._max_simulation_time)
            self._user_last_sars[uid] = SARS(curr_state, t, bid, 0, None, None, is_done)

            if t == self.buffer_priming_period + 500:
                self.evaluate = True
                print("begin bidding")

            if not self.evaluate and t > self.buffer_priming_period:
                self._train_minibatch()

            if bid > 0:
                self.bid_times.append((t % 40)*0.3)

            # update user stats
            self._user_stats[uid][0].append(t)
            self._user_stats[uid][3].append(bid_actual)
            self._user_site_stats[uid][self.site_to_index[bid_request.url]].append(t)
            self._user_multiplier_stats[uid][self.multipliers.index(self.userIDMultiplierDict[uid])].append(t)

            # Output and record status update
            if (t % self._report_interval == 0) and (t != self._last_report_time):
                self._last_report_time = t
                type_bidreqs = defaultdict(lambda: 0.0)
                type_wins = defaultdict(lambda: 0.0)
                type_convs = defaultdict(lambda: 0.0)
                type_prices = defaultdict(lambda: [])
                type_av_prices = defaultdict(lambda: 0.0)
                for uid in self._user_stats:
                    type = self._user_types[uid]
                    type_bidreqs[type] += bisect(self._user_stats[uid][0], t) - bisect(self._user_stats[uid][0],
                                                                                       t - self._report_interval)
                    type_wins[type] += bisect(self._user_stats[uid][1], t) - bisect(self._user_stats[uid][1],
                                                                                    t - self._report_interval)
                    type_convs[type] += bisect(self._user_stats[uid][2], t) - bisect(self._user_stats[uid][2],
                                                                                     t - self._report_interval)
                    prices = self._user_stats[uid][3][bisect(self._user_stats[uid][0], t-self._report_interval):]
                    if len(prices) > 0:
                        type_prices[type].append(sum(prices)/len(prices))

                for type in type_prices.keys():
                    type_av_prices[type] = sum(type_prices[type])/len(type_prices[type])

                site_type_bidreqs = defaultdict(lambda: defaultdict(int))
                site_type_wins = defaultdict(lambda: defaultdict(int))
                for uid in self._user_site_stats:
                    type = self._user_types[uid]
                    for site in range(1,self.num_of_sites):
                        site_type_bidreqs[type][site] += bisect(self._user_site_stats[uid][site], t) - bisect(self._user_site_stats[uid][site], t - self._report_interval)
                        winIndx = self.num_of_sites + site
                        site_type_wins[type][site] += bisect(self._user_site_stats[uid][winIndx], t) - bisect(self._user_site_stats[uid][winIndx], t - self._report_interval)

                #logging multiplier specific agent performance.
                multiplier_type_bidreqs = defaultdict(lambda: defaultdict(int))
                multiplier_type_wins = defaultdict(lambda: defaultdict(int))
                for uid in self._user_multiplier_stats:
                    type = self._user_types[uid]
                    for multiplierIdx, multiplier in enumerate(self.multipliers):
                        multiplier_type_bidreqs[type][multiplierIdx] += bisect(self._user_multiplier_stats[uid][multiplierIdx], t) - bisect(self._user_multiplier_stats[uid][multiplierIdx], t - self._report_interval)
                        winIndx = len(self.multipliers) + multiplierIdx
                        multiplier_type_wins[type][multiplierIdx] += bisect(self._user_multiplier_stats[uid][winIndx], t) - bisect(self._user_multiplier_stats[uid][winIndx], t - self._report_interval)

                recent_spend = self._total_spend_usd - self.last_spend
                current_cpa = 0
                if sum([type_convs[type] for type in type_bidreqs]) > 0:
                    current_cpa = recent_spend / (sum([type_convs[type] for type in type_bidreqs]))

                s = f'Time:{t} Spent:{self._total_spend_usd:.2f}, Noise: {noise} Steps:{self._n_mini_batch} Abs_Time:{datetime.datetime.now().strftime("%I:%M%p")} Num_Memories:{self._replay_memory.size()} Multiplier:{self.current_multiplier:.2f}'
                if t > self.buffer_priming_period:
                    s += f' Pace = {(self._total_spend_usd - self.spend_start)*1000/(t-self.buffer_priming_period)}'
                print(s)
                self.last_spend = self._total_spend_usd

                # report = list()
                for type in type_bidreqs:
                    print(
                        f'DDPG Type {type}: frac won: {type_wins[type]/type_bidreqs[type]:.3f}, '
                        f'average bid price: {type_av_prices[type]:.2f}, convs: {int(type_convs[type])}')
                    # report.append([t, type, type_wins[type], type_bidreqs[type], type_convs[type]])

                self._reports.append([t, type_bidreqs, type_wins, type_convs, type_av_prices])
                self._site_reports.append([t, site_type_bidreqs, site_type_wins])
                self._multiplier_reports.append([t, multiplier_type_bidreqs, multiplier_type_wins])
                self.missed_conversions = 0

                #stop exploration after a certain time.
                if t > self.exploration_stop:
                    self.evaluate = True

                # saving logs and models.
                with open(f'logs/{self.experiment_name}/exp_{self.experiment_number}/insensitive_control.json', 'w') as outfile:
                    json.dump([self._reports, self._user_stats], outfile)
                with open(f'logs/{self.experiment_name}/exp_{self.experiment_number}/insensitive_control2.json', 'w') as outfile:
                    json.dump([self._site_reports, self._user_site_stats], outfile)
                with open(f'logs/{self.experiment_name}/exp_{self.experiment_number}/insensitive_control3.json', 'w') as outfile:
                    json.dump([self._multiplier_reports, self._user_multiplier_stats], outfile)
                torch.save(self.critic_net.state_dict(), f'models/{self.experiment_name}/exp_{self.experiment_number}/insensitive_control_critic_net_at_t{t}.net') # saving the net
                torch.save(self.actor_net.state_dict(), f'models/{self.experiment_name}/exp_{self.experiment_number}/insensitive_control_actor_net_at_t{t}.net')

        return BidResponse(bid_actual, creative, is_done)  # returning the bid while checking if the campaign is done

    def update_multiplier(self, t):
        """
        Calculate whether we are overpacing or underpacing our budget constraint, and then update the multiplier in the
        direction to match the specified pace. If the model hasn't started optimizing yet, we set the multiplier randomly
        to give the model experience of all different multipliers
        """
        if t == self.buffer_priming_period:
            self.spend_start = self._total_spend_usd
        if t <= self.buffer_priming_period:
            self.current_multiplier = np.random.uniform(self.multiplier_range[0], self.multiplier_range[1])
        else:
            pace = (self._total_spend_usd - self.spend_start)/(t-self.buffer_priming_period)
            loss = np.tanh((pace - self.budget_per_1000 / 1000)/2)
            if loss > 0:
                self.current_multiplier += loss * (self.multiplier_range[1] - self.current_multiplier)
            else:
                self.current_multiplier += loss * (self.current_multiplier - self.multiplier_range[0])

    def process_auction_result(self, bid_request, win_price_microusd):
        """
        lets the agent know what the result of the auction was
        """
        if win_price_microusd is not None:
            self._total_spend_usd += win_price_microusd / 1e6  # updating how much we spent
            uid = bid_request.user_id
            # update user stats, website stats and multiplier stats
            self._user_stats[uid][1].append(bid_request.simulation_time)
            win_site_index = self.num_of_sites + self.site_to_index[bid_request.url]
            self._user_site_stats[uid][win_site_index].append(bid_request.simulation_time)
            multiplier_index = len(self.multipliers) + self.multipliers.index(self.userIDMultiplierDict[uid])
            self._user_multiplier_stats[uid][multiplier_index].append(bid_request.simulation_time)
            # update last SARS with the negative reward
            self._user_last_sars[uid].r -= win_price_microusd * 10 ** self.current_multiplier / float(self.reward_scaler)

    def process_click(self, sim_time, bid_req_id):
        # feel like we should add something here
        pass

    def process_conversion(self, simulation_time, user_id):
        """
        lets the agent know when a given user converted
        """
        self._conversions.append((simulation_time, user_id))  # log of conversions
        if self._is_bidding:
            # update user stats
            self._user_stats[user_id][2].append(simulation_time)  # appending a conversion in the user stats
            # con_site_index = 2*self.num_of_sites + self.site_to_index[url] # the conversion object doesn't include website atm.
            # self._user_site_stats[user_id][win_site_index].append(simulation_time)
            # update last SARS with large rewards
            if user_id in self._user_last_sars:
                if simulation_time - self._user_last_sars[user_id].t0 <= self.attribution_window:
                    self._user_last_sars[user_id].r += self.conv_reward_microusd / float(self.reward_scaler)
                elif self._user_types[user_id] == 1:
                    # print(f'missed a conversion: type: {self._user_types[user_id]}')
                    self.missed_conversions += 1

    def get_user_web_logs(self):
        return self._user_web_logs

    def _build_user2vec_model(self):
        """
        builds and trains model to map uid to high dimensional vector
        """
        if not self.use_u2v and not self.use_s2v:
            self._is_bidding = True
            return

        # Check if we have enough data to compute user2vec model
        avg_len = np.mean([len(self._user_web_logs[uid]) for uid in self._user_web_logs])
        if avg_len > 50:
            print('DQN Agent has collected enough data to build user2vec model.')
            from gensim.models.doc2vec import Doc2Vec, TaggedDocument

            # convert logs into docs for doc2vec
            self.u2v_ids = self._user_web_logs.keys()
            self.u2v_docs = []
            for id in self.u2v_ids:
                self.u2v_docs.append(TaggedDocument(self._user_web_logs[id], [id]))

            # train doc2vec model
            self.u2v_model = Doc2Vec(self.u2v_docs, vector_size=self.u2v_dims, window=self.u2v_win,
                                     epochs=self.u2v_iterations)
            self.u2v_vecs = dict(zip(self.u2v_ids, list(self.u2v_model.docvecs.vectors)))
            self.s2v_vecs = dict(zip(self.u2v_model.wv.index_to_key, list(self.u2v_model.wv.vectors)))

            # save the user2vecs
            self.make_uservec_plots()

            # time to start bidding
            print('DQN Agent will now begin bidding.')
            self._is_bidding = True

        return

    def _create_critic_network(self, is_train=None):
        """
        returns a neural net object (defined at top) with
        """
        return CriticNet(input_size=self.state_size+1, output_size=1)

    def _create_actor_network(self, is_train=None):
        """
        returns a neural net object (defined at top) with
        """
        return ActorNet(input_size=self.state_size, output_size=1)

    def _update_target_networks(self, tau):
        for target_param, param in zip(self.critic_net_target.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data*tau)

        for target_param, param in zip(self.actor_net_target.parameters(), self.actor_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data*tau)

    def _generate_target_q_values(self, next_state_batch, reward_batch):
        """
        performs a forward pass on the states and adds the reward batch to calculate the target values. meant to be
        overridden for DDQN agent
        """
        
        # Get the actions and the state values to compute the targets
        noise = np.random.normal(self.noise_mean, self.noise_std/5)
        #noise = 0
        next_action_batch = (self.actor_net_target(next_state_batch) + noise).clamp(self._bid_range[0], self._bid_range[1])
        q1, q2 = self.critic_net_target(next_state_batch, next_action_batch.detach())
        next_state_action_values = torch.min(q1, q2)
        #next_state_action_values = self.critic_net.q1(next_state_batch, next_action_batch.detach())
        target_values = reward_batch.unsqueeze(1) + self.gamma * next_state_action_values

        return target_values

    def _scale_up(self, bid):
        # return (bid * (self._bid_range[1]-self._bid_range[0]) + self._bid_range[1] + self._bid_range[0])/2
        return self._bid_range[0] + (0.5 * (bid + 1.0) * (self._bid_range[1] - self._bid_range[0]))

    def _train_minibatch(self):
        """
        Samples a batch and updates the parameters/networks of the agent according to the sampled batch.
        This means we ...
            1. Compute the targets
            2. Update the Q-function/critic by one step of gradient descent
            3. Update the policy/actor by one step of gradient ascent
            4. Update the target networks through a soft update
        """

        if self._replay_memory.size() < self._mini_batch_size:
            return

        # get a mini batch from the replay memory.
        mini_batch = self._replay_memory.sample(batch_size=self._mini_batch_size)

        states = mini_batch.states
        next_states = mini_batch.next_states
        rewards = mini_batch.rewards
        actions = mini_batch.actions

        # loading all the tensors on device.
        rewards.to(self.device)
        actions.to(self.device)
        states.to(self.device)
        next_states.to(self.device)

        # train the critic network
        target_q_values = self._generate_target_q_values(next_state_batch=next_states, reward_batch=rewards)
        self.critic_net.optimizer.zero_grad()
        q1, q2 = self.critic_net(states, actions)
        value_loss = self.critic_net.criterion(q1, target_q_values) + self.critic_net.criterion(q2, target_q_values)
        value_loss.backward()
        self.critic_net.optimizer.step()

        # train the actor network
        if self._n_mini_batch % 2 == 0:

            self.actor_net.optimizer.zero_grad()
            actions_taken = self.actor_net(states)
            policy_loss = -self.critic_net.q1(states, actions_taken).mean()
            policy_loss.backward()
            self.actor_net.optimizer.step()
            self._update_target_networks(self.tau)
            del policy_loss

        del actions
        del rewards
        del states
        del next_states
        del value_loss

        self._n_mini_batch += 1

    def make_uservec_plots(self):
        # are the user vecs constructed representative enough?
        uservecs = np.array(self.u2v_model.dv.vectors)
        types = [self._user_types[uid] for uid in self.u2v_ids]
        tsne = TSNE(n_components=2, perplexity=40.0)
        reduced_vectors = tsne.fit_transform(uservecs)
        df = pd.DataFrame()
        df['type'] = types
        df['x'] = reduced_vectors[:, 0]
        df['y'] = reduced_vectors[:, 1]
        df.plot.scatter(x='x', y='y', c='type', colormap='jet')
        plt.savefig(f'logs/{self.experiment_name}/user2vec_plot.png')
        plt.close()