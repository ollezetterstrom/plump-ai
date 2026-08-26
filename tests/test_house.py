from plump.env.engine import PlumpEnv
from plump.config.settings import terminal_reward, display_points


def test_zero_scoring():
    assert terminal_reward(0, 0) == 5.0
    assert display_points(0, 0) == 5
    assert terminal_reward(1, 1) == 11.0
    assert terminal_reward(1, 0) == -2.0


def test_leader():
    env = PlumpEnv()
    env.bids = [1, 3, 3, 2]
    assert env.get_leader() == 1
    env.bids = [2, 2, 2, 2]
    assert env.get_leader() == 0
    env.bids = [-1, 1, 2, 0]
    assert env.get_leader() == 0
