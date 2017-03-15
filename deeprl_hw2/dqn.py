
import numpy as np
import random
import os
import cPickle as pickle

"""Main DQN agent."""

class DQNAgent(object):
    """Class implementing DQN.

    This is a basic outline of the functions/parameters you will need
    in order to implement the DQNAgnet. This is just to get you
    started. You may need to tweak the parameters, add new ones, etc.

    Feel free to change the functions and funciton parameters that the
    class provides.

    We have provided docstrings to go along with our suggested API.

    Parameters
    ----------
    q_network: keras.models.Model
      Your Q-network model.
    preprocessor: deeprl_hw2.core.Preprocessor
      The preprocessor class. See the associated classes for more
      details.
    memory: deeprl_hw2.core.Memory
      Your replay memory.
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    num_burn_in: int
      Before you begin updating the Q-network your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: int
      How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your
      replay memory, for every Q-network update that you run.
    batch_size: int
      How many samples in each minibatch.
    """
    def __init__(self,
                 state_shape,
                 model_input_shape,
                 q_network,
                 preprocessor,
                 memory,
                 policy,
                 gamma,
                 target_reset_interval,
                 num_burn_in,
                 train_freq,
                 batch_size,
                 eval_interval,
                 eval_episodes,
                 double_net,
                 output,
                 do_render):
        self.state_shape = state_shape
        self.model_input_shape = model_input_shape
        self.q_network = q_network
        self.preprocessor = preprocessor
        self.memory = memory
        self.policy = policy
        self.gamma = gamma
        self.target_reset_interval = target_reset_interval
        self.num_burn_in = num_burn_in
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.double_net = double_net
        self.output = output
        self.do_render = do_render

    def compile(self, optimizer, loss_func):
        """Setup all of the TF graph variables/ops.

        This is inspired by the compile method on the
        keras.models.Model class.

        This is a good place to create the target network, setup your
        loss function and any placeholders you might need.
        
        You should use the mean_huber_loss function as your
        loss_function. You can also experiment with MSE and other
        losses.

        The optimizer can be whatever class you want. We used the
        keras.optimizers.Optimizer class. Specifically the Adam
        optimizer.
        """
        self.q_network['online'].compile(loss=loss_func, optimizer=optimizer)
        self.q_network['target'].compile(loss=loss_func, optimizer=optimizer)

    def calc_q_values(self, state):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        pass

    def select_action(self, state, **kwargs):
        """Select the action based on the current state.

        You will probably want to vary your behavior here based on
        which stage of training your in. For example, if you're still
        collecting random samples you might want to use a
        UniformRandomPolicy.

        If you're testing, you might want to use a GreedyEpsilonPolicy
        with a low epsilon.

        If you're training, you might want to use the
        LinearDecayGreedyEpsilonPolicy.

        This would also be a good place to call
        process_state_for_network in your preprocessor.

        Returns
        --------
        selected action
        """
        pass
        

    def update_policy(self, iter_num):
        """Update your policy.

        Behavior may differ based on what stage of training your
        in. If you're in training mode then you should check if you
        should update your network parameters based on the current
        step and the value you set for train_freq.

        Inside, you'll want to sample a minibatch, calculate the
        target values, update your network, and then update your
        target values.

        You might want to return the loss and other metrics as an
        output. They can help you monitor how training is going.
        """
        mini_batch = random.sample(self.memory, self.batch_size)
        input_b = []
        input_b_n = []
        for st_m, act, rew, st_m_n, done_b in mini_batch:
            st = self.preprocessor.process_state_for_network(st_m)
            input_b.append(st.reshape(self.model_input_shape))
            st_n = self.preprocessor.process_state_for_network(st_m_n)
            input_b_n.append(st_n.reshape(self.model_input_shape))
        input_b = np.stack(input_b)
        input_b_n = np.stack(input_b_n)
        q_target_b_n = self.q_network['target'].predict(input_b_n)
        
        target_b = np.zeros(q_target_b_n.shape, dtype=np.float32)
        for idx, (st_m, act, rew, _, done_b) in enumerate(mini_batch):
            if done_b:
                target_b[idx, act] = rew
            else:
                target_b[idx, act] = rew + self.gamma * np.max(q_target_b_n[idx])
        
        if self.double_net:
            if np.random.rand() < 0.5:
                loss_online = self.q_network['online'].train_on_batch(input_b, target_b)
                loss_target = self.q_network['target'].evaluate(input_b, target_b, verbose=0)
            else:
                loss_target = self.q_network['target'].train_on_batch(input_b, target_b)
                loss_online = self.q_network['online'].evaluate(input_b, target_b, verbose=0)
        else:
            loss_online = self.q_network['online'].train_on_batch(input_b, target_b)
            loss_target = self.q_network['target'].evaluate(input_b, target_b, verbose=0)
            if iter_num % self.target_reset_interval == 0:
                print 'update update update update update'
                self.q_network['target'].set_weights(self.q_network['online'].get_weights())
        return loss_online, loss_target

    def fit(self, env, num_iterations, max_episode_length=None):
        """Fit your model to the provided environment.

        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.

        You should probably also periodically save your network
        weights and any other useful info.

        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.

        Parameters
        ----------
        env: gym.Env
          This is your Atari environment. You should wrap the
          environment using the wrap_atari_env function in the
          utils.py
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """
        
        window = self.state_shape[0]
        save_pivots = [int(num_iterations / 3), int(2 * num_iterations / 3), num_iterations]
        
        # filling in self.batch_size states
        if self.memory is not None and len(self.memory) < self.batch_size:
            print '########## burning in some samples #############'
            env.reset()
            noop = 0
            state_mem = np.zeros(self.state_shape, dtype=np.uint8)
            while len(self.memory) < self.batch_size:
                state_mem_next = []
                reward = 0.0
                done = False
                for _ in range(window):
                    obs_next, obs_reward, obs_done, info = env.step(noop)
                    if self.do_render:
                        env.render()
                    obs_next_mem = self.preprocessor.process_state_for_memory(obs_next)
                    state_mem_next.append(obs_next_mem)
                    reward += obs_reward
                    done = done or obs_done
                state_mem_next = np.stack(state_mem_next)
                reward = self.preprocessor.process_reward(reward)
                
                # store transition into replay memory
                transition_mem = (state_mem, noop, reward, state_mem_next, done)
                self.memory.append(transition_mem)
                if done:
                    env.reset()
                    state_mem = np.zeros(self.state_shape, dtype=np.uint8)
        
        iter_num = 0
        episode = 0
        
        while iter_num < num_iterations:
            env.reset()
            state_mem = np.zeros(self.state_shape, dtype=np.uint8)
            state = self.preprocessor.process_state_for_network(state_mem)
            done = False
            
            episode_counter = 0
            print '########## begin new episode #############'
            while episode_counter < max_episode_length or max_episode_length is None:
                # get online q value and get action
                input_state = np.stack([state.reshape(self.model_input_shape)])
                q_online = self.q_network['online'].predict(input_state)
                action = self.policy['train'].select_action(q_online, iter_num)
                
                # do action to get the next state
                state_mem_next = []
                reward = 0.0
                done = False
                for _ in range(window):
                    obs_next, obs_reward, obs_done, info = env.step(action)
                    if self.do_render:
                        env.render()
                    obs_next_mem = self.preprocessor.process_state_for_memory(obs_next)
                    state_mem_next.append(obs_next_mem)
                    reward += obs_reward
                    done = done or obs_done
                state_mem_next = np.stack(state_mem_next)
                
                state_next = self.preprocessor.process_state_for_network(state_mem_next)
                reward = self.preprocessor.process_reward(reward)
                
                # store transition into replay memory
                transition_mem = (state_mem, action, reward, state_mem_next, done)
                if self.memory is not None: # self.memory should be a deque with a max length
                    self.memory.append(transition_mem)
                
                # update networks
                if self.memory is None:
                    input_state_next = np.stack([state_next.reshape(self.model_input_shape)])
                    
                    q_target_next = self.q_network['target'].predict(input_state_next)
                    target = np.zeros(q_target_next.shape, dtype=np.float32)
                    if done:
                        target[0, action] = reward
                    else:
                        target[0, action] = reward + self.gamma * np.max(q_target_next)
                    loss_online = loss_target = self.q_network['online'].train_on_batch(input_state, target)
                    self.q_network['target'].set_weights(self.q_network['online'].get_weights())
                else:
                    loss_online, loss_target = self.update_policy(iter_num)
                
                if iter_num % self.eval_interval == 0:
                    print '########## evaluation #############'
                    self.evaluate(env, num_episodes=self.eval_episodes)
                
                state_mem = state_mem_next
                state = state_next
                episode_counter += 1
                iter_num += 1
                
                if iter_num in save_pivots:
                    weight_save_name = os.path.join(self.output, 'online_weight_{:d}.save'.format(iter_num))
                    with open(weight_save_name, 'wb') as save:
                        weights = self.q_network['online'].get_weights()
                        pickle.dump((weights, self.memory), save, protocol=pickle.HIGHEST_PROTOCOL)
                    print 'weights & memory written to {:s}'.format(weight_save_name)
                
                if done:
                    break
            episode += 1
            print 'losses:', loss_online, loss_target
            print '{:d} out of {:d} iterations'.format(iter_num, num_iterations)
                    

    def evaluate(self, env, num_episodes, max_episode_length=None):
        """Test your agent with a provided environment.
        
        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """
        window = self.state_shape[0]
        total_reward = 0.0
        for episode in range(num_episodes):
            env.reset()
            state_mem = np.zeros(self.state_shape, dtype=np.uint8)
            state = self.preprocessor.process_state_for_network(state_mem)
            done = False
            episode_counter = 0
            cum_reward = 0.0
            while episode_counter < max_episode_length or max_episode_length is None:
                
                # get online q value and get action
                input_state = np.stack([state.reshape(self.model_input_shape)])
                q_online = self.q_network['online'].predict(input_state)
                action = self.policy['evaluation'].select_action(q_online)
                
                # do action to get the next state
                state_mem_next = []
                reward = 0.0
                done = False
                for _ in range(window):
                    obs_next, obs_reward, obs_done, info = env.step(action)
                    if self.do_render:
                        env.render()
                    obs_next_mem = self.preprocessor.process_state_for_memory(obs_next)
                    state_mem_next.append(obs_next_mem)
                    reward += obs_reward
                    done = done or obs_done
                state_mem_next = np.stack(state_mem_next)
                
                state_next = self.preprocessor.process_state_for_network(state_mem_next)
                reward = self.preprocessor.process_reward(reward)
                cum_reward += reward
                state_mem = state_mem_next
                state = state_next
                episode_counter += 1
                if done:
                    break
            print '  episode reward: {:f}'.format(cum_reward)
            total_reward += cum_reward
        print 'average episode reward: {:f}'.format(total_reward / num_episodes)
