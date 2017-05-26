# Markov Decision Process
# ---
# @author Yiren Lu
# @email luyiren [at] seas [dot] upenn [dot] edu
#
# MIT License


class MDP:

  def get_states(self):
    """
    get a list of all states
    """
    abstract

  def get_actions(self, state):
    """
    get all the actions that can be takens on the current state
    """
    abstract

  def get_reward(self, state):
    """
    return the reward on current state
    """
    abstract

  def get_transition_states_and_probs(self, state, action):
    """
    get all the possible transition states and their probabilities with [action] on [state]
    """
    abstract

  def is_terminal(self, state):
    """
    return True is the [state] is terminal
    """
    abstract
