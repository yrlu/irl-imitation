# Environment Abstract Class
# ---
# @author Yiren Lu
# @email luyiren [at] seas [dot] upenn [dot] edu
#
# MIT License


class Env:
  
  def reset(self, start_state):
    """
    Reset the gridworld for model-free learning. It assumes only 1 agent in the gridworld.
    """
    abstract


  def get_current_state(self):
    abstract


  def step(self, action):
    abstract