from typing import List

import numpy as np

from vgc.balance.meta import MetaData
from vgc.behaviour import TeamBuildPolicy, BattlePolicy
from vgc.behaviour.BattlePolicies import TypeSelector
from vgc.datatypes.Constants import DEFAULT_PKM_N_MOVES, DEFAULT_TEAM_SIZE
from vgc.datatypes.Objects import Pkm, PkmTemplate, PkmFullTeam, PkmRoster, PkmTeam
from vgc.engine.PkmBattleEnv import PkmBattleEnv


class RandomTeamBuilder(TeamBuildPolicy):
    """
    Agents that selects teams randomly.
    """

    def __init__(self):
        self.roster = None

    def requires_encode(self) -> bool:
        return False

    def close(self):
        pass

    def set_roster(self, roster: PkmRoster, ver: int = 0):
        self.roster = roster

    def get_action(self, meta: MetaData) -> PkmFullTeam:
        n_pkms = len(self.roster)
        members = np.random.choice(n_pkms, 3, False)
        pre_selection: List[PkmTemplate] = [self.roster[i] for i in members]
        team: List[Pkm] = []
        for pt in pre_selection:
            moves: List[int] = np.random.choice(DEFAULT_PKM_N_MOVES, DEFAULT_PKM_N_MOVES, False)
            team.append(pt.gen_pkm(moves))
        return PkmFullTeam(team)


class FixedTeamBuilder(TeamBuildPolicy):
    """
    Agents that always selects the same team.
    """

    def __init__(self):
        self.roster = None

    def requires_encode(self) -> bool:
        return False

    def close(self):
        pass

    def set_roster(self, roster: PkmRoster, ver: int = 0):
        self.roster = roster

    def get_action(self, meta: MetaData) -> PkmFullTeam:
        roster = list(self.roster)
        pre_selection: List[PkmTemplate] = roster[0:3]
        team: List[Pkm] = []
        for pt in pre_selection:
            team.append(pt.gen_pkm([0, 1, 2, 3]))
        return PkmFullTeam(team)


def run_battles(pkm0, pkm1, agent0, agent1, n_battles):
    wins = [0, 0]
    t0 = PkmTeam([pkm0])
    t1 = PkmTeam([pkm1])
    env = PkmBattleEnv((t0, t1), encode=(agent0.requires_encode(), agent1.requires_encode()))
    for _ in range(n_battles):
        s = env.reset()
        t = False
        while not t:
            a0 = agent0.get_action(s[0])
            a1 = agent1.get_action(s[1])
            s, _, t, _ = env.step([a0, a1])
        wins[env.winner] += 1
    return wins


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def select_next(matchup_table, n_pkms) -> List[int]:
    average_winrate = np.sum(matchup_table, axis=1) / n_pkms
    policy = softmax(average_winrate)
    return np.random.choice(n_pkms, 3, False, p=policy)


class IndividualPkmCounter(TeamBuildPolicy):
    """
    Counter the team composition we believe will be selected by an opponent. We disregard synergies in teams as in the
    original algorithms which were tested over pkm GO and look for individual pairwise win rates and coverage.
    Contrary to the source paper, the meta is not the win rate directly but instead the usage rate, which we assume is
    a direct implication of the win rate. We use epistemic reasoning to find the meta counter teams and play in an
    unpredictable fashion.
    Source: https://ieee-cog.org/2021/assets/papers/paper_192.pdf
    """
    matchup_table = None
    n_pkms = -1
    pkms = None

    def __init__(self, agent0: BattlePolicy = TypeSelector(), agent1: BattlePolicy = TypeSelector(), n_battles=10):
        self.agent0 = agent0
        self.agent1 = agent1
        self.n_battles = n_battles
        self.policy = None
        self.ver = -1

    def requires_encode(self) -> bool:
        return False

    def close(self):
        pass

    def set_roster(self, roster: PkmRoster, ver: int = 0):
        """
        Instead of storing the roster, we fill a pairwise match up table where each entry has the estimated win rate
        from a row pkm against a column pkm.
        """
        if self.ver < ver:
            self.ver = ver
            IndividualPkmCounter.pkms = []
            for pt in roster:
                IndividualPkmCounter.pkms.append(pt.gen_pkm([0, 1, 2, 3]))
            IndividualPkmCounter.n_pkms = len(roster)
            IndividualPkmCounter.matchup_table = np.zeros((IndividualPkmCounter.n_pkms, IndividualPkmCounter.n_pkms))
            for i, pkm0 in enumerate(IndividualPkmCounter.pkms):
                for j, pkm1 in enumerate(IndividualPkmCounter.pkms[i:]):
                    if j == 0:  # p0 == p1
                        IndividualPkmCounter.matchup_table[i][i] = 0.5
                    else:
                        wins = run_battles(pkm0, pkm1, self.agent0, self.agent1, self.n_battles)
                        IndividualPkmCounter.matchup_table[i][i + j] = wins[0] / self.n_battles
                        IndividualPkmCounter.matchup_table[i + j][i] = wins[1] / self.n_battles
            average_winrate = np.sum(IndividualPkmCounter.matchup_table, axis=1) / IndividualPkmCounter.n_pkms
            # pre compute policy
            self.policy = softmax(average_winrate)

    def get_action(self, meta: MetaData) -> PkmFullTeam:
        members: List[int] = np.random.choice(IndividualPkmCounter.n_pkms, 3, False, p=self.policy)
        return PkmFullTeam([IndividualPkmCounter.pkms[members[0]], IndividualPkmCounter.pkms[members[1]],
                            IndividualPkmCounter.pkms[members[2]]])
