import unittest
import sys
# if "../" not in sys.path:
# sys.path.append("../")
# from envs import gridworld
import gridworld
import value_iteration


class ValueIterationAgentTest(unittest.TestCase):
  """
  Unit test for value iteration agent
  """

  def setUp(self):
    grid = [['0', '0', '0', '1'],
            ['0', 'x', '0', '-1'],
            ['0', '0', '0', '0']]

    self.grid = grid
    self.gw_non_deterministic = gridworld.GridWorld(
        grid, {(0, 3), (1, 3)}, 0.8)

    self.agent = value_iteration.ValueIterationAgent(
        self.gw_non_deterministic, 0.9, 100)

  def test_eval_policy(self):
    print 'Show evaluation of the optimal policy:'
    self.gw_non_deterministic.display_value_grid(
        self.agent.eval_policy_dist(self.agent.get_policy_dist()))

  def test_show_policy(self):
    print 'Show policy learned by value iteration:'
    self.gw_non_deterministic.display_policy_grid(
        self.agent.get_optimal_policy())

  def test_values(self):
    print 'Show value iteration results:'
    self.gw_non_deterministic.display_value_grid(self.agent.values)


if __name__ == '__main__':
  unittest.main()
