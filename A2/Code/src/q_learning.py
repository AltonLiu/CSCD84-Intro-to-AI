"""
CSCD84 - Artificial Intelligence, Winter 2025, Assignment 2
B. Chan
"""

import numpy as np


class QLearning:
    def __init__(self, num_actions, features, policy_rng, learner_rng, alpha, eps):
        self.num_actions = num_actions
        self.features = features
        self.policy_rng = policy_rng
        self.learner_rng = learner_rng
        self.alpha = alpha
        self.eps = eps

        # Randomly initialize parameters following standard normal
        self.parameters = learner_rng.randn(self.features.dim)

    def compute_qsa(self, state, action):
        feature = self.features(state, action)
        return self.parameters @ feature

    def get_action(self, curr_state, *args, **kwargs):
        """
        Samples action using epsilon-greedy strategy.
        """

        action = self.policy_rng.randint(self.num_actions)
        prob = self.policy_rng.uniform()

        # ========================================================
        
        if prob <= 1 - self.eps:
            # Select the best action when probability 1 - eps
            q_values = [self.compute_qsa(curr_state, a) for a in range(self.num_actions)]
            action = np.argmax(q_values)
        
        # ========================================================

        return int(action)

    def compute_update(self, curr_state, action, reward, done, next_state):
        """
        Computes the update.

        Recall that the update rule is:
            (r + gamma * max_b' Q_w(s', a') - Q_w(s, a)) * grad_w Q_w(s, a)

        NOTE: Upon reaching terminal, we no longer bootstrap from the next Q-value.
        """

        update = None

        # ========================================================
        
        gamma = 1.0
        
        # Compute Q_w(s, a)
        qsa = self.compute_qsa(curr_state, action)

        # Compute the target value
        if done:
            target = reward
        else:
            next_q_values = [self.compute_qsa(next_state, a) for a in range(self.num_actions)]
            target = reward + gamma * np.max(next_q_values)

        # Compute the gradient of Q_w(s, a)
        feature = self.features(curr_state, action)
        grad_qsa = feature

        # Compute the update
        update = (target - qsa) * grad_qsa

        # ========================================================

        return update

    def learn(self, curr_state, action, reward, done, next_state, *args, **kwargs):
        """
        Updates the parameters of the Q-function
        """

        update = self.compute_update(curr_state, action, reward, done, next_state)
        self.parameters += self.alpha * update
