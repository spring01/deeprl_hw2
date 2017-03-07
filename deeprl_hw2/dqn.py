
import numpy as np
import random

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
                 q_network,
                 preprocessor,
                 memory,
                 policy,
                 gamma,
                 target_reset_interval,
                 num_burn_in,
                 train_freq,
                 batch_size):
        self.q_network = q_network
        self.preprocessor = preprocessor
        self.memory = memory
        self.policy = policy
        self.gamma = gamma
        self.target_reset_interval = target_reset_interval
        self.num_burn_in = num_burn_in
        self.train_freq = train_freq
        self.batch_size = batch_size

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
        return self.q_network.predict(state)

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
        q_value = self.calc_q_values(state)
        

    def update_policy(self):
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
        pass

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
        
        input_shape = self.q_network['online'].get_config()[0]['config']['batch_input_shape']
        
        iter_num = 0
        episode = 0
        while iter_num < num_iterations:
            env.reset()
            state_mem = self.preprocessor['history'].reset()
            state = self.preprocessor['atari'].process_state_for_network(state_mem)
            done = False
            
            episode_counter = 0
            print '##########begin new episode#############'
            while episode_counter < max_episode_length or max_episode_length is None:
                print 'episode %d' % episode
                print iter_num
                env.render()
                # get online q value and get action
                
                state_arr = np.stack(state)
                state_arr = state_arr.reshape(input_shape[1:])
                input_state = np.zeros([1] + list(state_arr.shape), dtype=np.float32)
                input_state[0] = state_arr
                q_online = self.q_network['online'].predict(input_state)
                action = self.policy['train'].select_action(q_online, iter_num)
                
                # do action to get the next state
                obs_next, reward, done, info = env.step(action)
                reward = self.preprocessor['atari'].process_reward(reward)
                
                # process state
                obs_next_mem_grey = self.preprocessor['atari'].process_state_for_memory(obs_next)
                state_mem_next = self.preprocessor['history'].process_state_for_network(obs_next_mem_grey)
                state_next = self.preprocessor['atari'].process_state_for_network(state_mem_next)
                
                # store transition into replay memory
                transition_mem = (state_mem, action, reward, state_mem_next, done)
                if self.memory is not None: # self.memory should be a deque with a max length
                    self.memory.append(transition_mem)
                
                
                # update networks
                if self.memory is None:
                    state_arr_next = np.stack(state_next)
                    state_arr_next = state_arr_next.reshape(input_shape[1:])
                    input_state_next = np.zeros([1] + list(state_arr_next.shape), dtype=np.float32)
                    input_state_next[0] = state_arr_next
                    
                    q_target_next = self.q_network['target'].predict(input_state_next)
                    target = np.zeros(q_target_next.shape, dtype=np.float32)
                    if done:
                        target[0, action] = reward
                    else:
                        target[0, action] = reward + self.gamma * np.max(q_target_next)
                    #~ import pdb; pdb.set_trace()
                    self.q_network['online'].train_on_batch(input_state, target)
                    self.q_network['target'].set_weights(self.q_network['online'].get_weights())
                elif len(self.memory) >= self.batch_size:
                    mini_batch = random.sample(self.memory, self.batch_size)
                    input_b = []
                    input_b_n = []
                    for st_m, act, rew, st_m_n, done_b in mini_batch:
                        st = self.preprocessor['atari'].process_state_for_network(st_m)
                        st = np.stack(st).reshape(input_shape[1:])
                        input_b.append(st)
                        st_n = self.preprocessor['atari'].process_state_for_network(st_m_n)
                        st_n = np.stack(st_n).reshape(input_shape[1:])
                        input_b_n.append(st_n)
                    input_b = np.stack(input_b)
                    input_b_n = np.stack(input_b_n)
                    q_target_b_n = self.q_network['target'].predict(input_b_n)
                    #~ import pdb; pdb.set_trace()
                    target_b = np.zeros(q_target_b_n.shape, dtype=np.float32)
                    for idx, (st_m, act, rew, _, done_b) in enumerate(mini_batch):
                        if done_b:
                            target_b[idx, act] = rew
                        else:
                            target_b[idx, act] = rew + self.gamma * np.max(q_target_b_n[idx])
                    curr_value = self.q_network['online'].train_on_batch(input_b, target_b)
                    print curr_value
                    if iter_num % self.target_reset_interval == 0:
                        print 'update update update update update'
                        self.q_network['target'].set_weights(self.q_network['online'].get_weights())
                
                state_mem = state_mem_next
                state = state_next
                episode_counter += 1
                iter_num += 1
                if done:
                    break
            episode += 1
                    

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
        pass
