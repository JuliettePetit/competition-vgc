from abc import ABC
from typing import Optional

from elo import INITIAL

from vgc.behaviour import BattlePolicy, TeamSelectionPolicy, TeamBuildPolicy, TeamPredictor, BalancePolicy
from vgc.behaviour.BalancePolicies import IdleBalancePolicy
from vgc.behaviour.BattlePolicies import RandomPlayer
from vgc.behaviour.TeamBuildPolicies import RandomTeamBuilder
from vgc.behaviour.TeamPredictors import NullTeamPredictor
from vgc.behaviour.TeamSelectionPolicies import RandomTeamSelectionPolicy
from vgc.datatypes.Objects import PkmFullTeam

random_battle_policy = RandomPlayer()
random_selector_policy = RandomTeamSelectionPolicy()
random_team_build_policy = RandomTeamBuilder()
idle_balance_policy = IdleBalancePolicy()
null_team_predictor = NullTeamPredictor()


class Competitor(ABC):

    @property
    def battle_policy(self) -> BattlePolicy:
        return random_battle_policy

    @property
    def team_selection_policy(self) -> TeamSelectionPolicy:
        return random_selector_policy

    @property
    def team_build_policy(self) -> TeamBuildPolicy:
        return random_team_build_policy

    @property
    def team_predictor(self) -> TeamPredictor:
        return null_team_predictor

    @property
    def balance_policy(self) -> BalancePolicy:
        return idle_balance_policy

    @property
    def name(self) -> str:
        return ""


class CompetitorManager:

    def __init__(self, c: Competitor):
        self.competitor: Competitor = c
        self.team: Optional[PkmFullTeam] = None
        self.elo: float = INITIAL
