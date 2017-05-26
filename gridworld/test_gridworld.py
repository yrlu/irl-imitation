import unittest
import gridworld


class GridWorldTest(unittest.TestCase):
  """
  Unit test for grid world
  """

  def setUp(self):
    grid = [['0', '0', '0', '0', '10'],
            ['0', 'x', '0', '0', '-10'],
            ['0', '0', '0', '0', '0']]

    self.grid = grid
    self.gw_deterministic = gridworld.GridWorld(grid, {(0, 4), (1, 4)}, 1)
    self.gw_non_deterministic = gridworld.GridWorld(
        grid, {(0, 4), (1, 4)}, 0.8)

  def test_grid_dims(self):
    self.assertEqual(len(self.gw_deterministic.get_grid()), 3)
    self.assertEqual(len(self.gw_deterministic.get_grid()[0]), 5)

  def test_grid_values(self):
    grid_tmp = self.gw_deterministic.get_grid()
    for i in range(len(grid_tmp)):
      for j in range(len(grid_tmp[0])):
        self.assertEqual(self.grid[i][j], grid_tmp[i][j])

  def test_get_states(self):
    self.assertEqual(len(self.gw_deterministic.get_states()), 14)

  def test_get_actions(self):
    self.assertEqual(len(self.gw_deterministic.get_actions((0, 0))), 2)
    self.assertEqual(len(self.gw_deterministic.get_actions((2, 0))), 2)
    self.assertEqual(len(self.gw_deterministic.get_actions((2, 4))), 2)
    self.assertEqual(len(self.gw_deterministic.get_actions((0, 4))), 2)
    self.assertEqual(len(self.gw_deterministic.get_actions((1, 0))), 2)

  def test_get_reward(self):
    self.assertEqual(self.gw_deterministic.get_reward((0, 0)), 0)
    self.assertEqual(self.gw_deterministic.get_reward((0, 4)), 10.0)
    self.assertEqual(self.gw_deterministic.get_reward((1, 4)), -10.0)

  def test_trans_prob_deter(self):
    self.assertEqual(
        len(
            self.gw_deterministic.get_transition_states_and_probs(
                (0, 0), 0)), 1)
    self.assertEqual(
        self.gw_deterministic.get_transition_states_and_probs(
            (0, 0), 0)[0][0], (0, 1))
    self.assertEqual(
        self.gw_deterministic.get_transition_states_and_probs(
            (0, 0), 0)[0][1], 1)

    self.assertEqual(
        len(
            self.gw_deterministic.get_transition_states_and_probs(
                (0, 0), 1)), 1)
    self.assertEqual(
        self.gw_deterministic.get_transition_states_and_probs(
            (0, 0), 1)[0][0], (0, 0))
    self.assertEqual(
        self.gw_deterministic.get_transition_states_and_probs(
            (0, 0), 1)[0][1], 1)

  def test_trans_prob_non_deter(self):
    self.assertEqual(
        len(
            self.gw_non_deterministic.get_transition_states_and_probs(
                (0, 0), 0)), 3)
    self.assertEqual(
        self.gw_non_deterministic.get_transition_states_and_probs(
            (0, 0), 0)[0][0], (0, 1))
    self.assertEqual(
        self.gw_non_deterministic.get_transition_states_and_probs(
            (0, 0), 0)[0][1], 0.8)

    self.assertTrue(
        self.gw_non_deterministic.get_transition_states_and_probs(
            (0, 0), 0)[1][1] - 0.1 < 1e-5)
    self.assertTrue(
        self.gw_non_deterministic.get_transition_states_and_probs(
            (0, 0), 0)[2][1] - 0.1 < 1e-5)

    self.assertEqual(
        len(
            self.gw_non_deterministic.get_transition_states_and_probs(
                (1, 0), 0)), 3)

  def test_terminals(self):
    self.assertTrue(self.gw_deterministic.is_terminal((0, 4)))
    self.assertTrue(self.gw_deterministic.is_terminal((1, 4)))

if __name__ == '__main__':
  unittest.main()
