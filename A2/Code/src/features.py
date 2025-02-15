"""
CSCD84 - Artificial Intelligence, Winter 2025, Assignment 2
B. Chan
"""

import numpy as np


class TabularFeature:
    """
    Tabular feature
    """

    @property
    def dim(self):
        return 64 * 5
    
    def __call__(self, state, action):
        q_table = np.zeros((64, 5))

        char_loc = np.where(state == b"C")
        char_loc_flattened = (char_loc[0] * 8 + char_loc[1]).item()
        
        q_table[char_loc_flattened, action] = 1

        return q_table.flatten()


class ReferenceLinearFeature:
    """
    A reference linear feature---it is by no means the best features.
    """

    @property
    def dim(self):
        return 64 * 2 + 1
    
    def __call__(self, state, action):
        char_loc = np.where(state == b"C")
        char_loc_flattened = (char_loc[0] * 8 + char_loc[1]).item()
        goal_loc = np.where(state == b"G")
        if len(goal_loc[0]) == 0:
            goal_loc_flattened = char_loc_flattened
        else:
            goal_loc_flattened = (goal_loc[0] * 8 + goal_loc[1]).item()

        next_loc = np.zeros(64)
        if action == 4:
            next_loc[char_loc_flattened] = 1
        elif action == 3:
            if char_loc_flattened - 8 < 0:
                next_loc[char_loc_flattened] = 1
            else:
                next_loc[char_loc_flattened - 8] = 1
        elif action == 2:
            if (char_loc_flattened + 1) % 8 == 0:
                next_loc[char_loc_flattened] = 1
            else:
                next_loc[char_loc_flattened + 1] = 1
        elif action == 1:
            if char_loc_flattened + 8 >= 64:
                next_loc[char_loc_flattened] = 1
            else:
                next_loc[char_loc_flattened + 8] = 1
        elif action == 0:
            if (char_loc_flattened - 1) % 8 == 0:
                next_loc[char_loc_flattened] = 1
            else:
                next_loc[char_loc_flattened - 1] = 1
        else:
            raise ValueError("Action should be [0, ..., 4], got: {}".format(action))

        feature = np.concatenate(
            (
                np.eye(64)[char_loc_flattened],
                next_loc,
                [goal_loc_flattened / 64],
            )
        ).astype(np.float32)

        return feature


class LinearFeature:
    """
    Converts state-action pair into a d-dimensional feature.

    TODO:
    Implement __call__ method which transforms a state-action pair into a d-dimensional vector.
    Here you get to choose your own feature dimensionality
    """

    @property
    def dim(self):
        # TODO: Modify this
        return 64
    
    def __call__(self, state, action):
        """
        Here, a state is representation using a 2d grid of characters.
        For example:
        [
            [b'F' b'F' b'F' b'F' b'H' b'H' b'F' b'F']
            [b'F' b'H' b'H' b'F' b'H' b'F' b'F' b'F']
            [b'H' b'F' b'F' b'F' b'F' b'F' b'F' b'F']
            [b'F' b'F' b'H' b'H' b'F' b'F' b'F' b'F']
            [b'F' b'F' b'F' b'F' b'F' b'H' b'H' b'F']
            [b'F' b'F' b'F' b'F' b'F' b'H' b'F' b'F']
            [b'F' b'H' b'F' b'F' b'H' b'F' b'F' b'F']
            [b'C' b'F' b'S' b'F' b'F' b'G' b'F' b'F']
        ]

        Description:
        - "C" for location of the agent
        - “S” for Start tile
        - “G” for Goal tile
        - “F” for frozen tile
        - “H” for a tile with a hole
        """
        feature = np.zeros(self.dim)

        # ========================================================
        # TODO: Implement linear feature

        # ========================================================

        return feature
