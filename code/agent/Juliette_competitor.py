from vgc.behaviour import BattlePolicy, TeamSelectionPolicy
from vgc.behaviour.BattlePolicies import GUIPlayer, FirstPlayer
from vgc.behaviour.TeamSelectionPolicies import GUITeamSelectionPolicy
from vgc.competition.Competitor import Competitor


class JulietteCompetitor(Competitor):

    def __init__(self, name: str = "Example"):
        self._name = name
        self._battle_policy = FirstPlayer()

    @property
    def name(self):
        return self._name

    @property
    def battle_policy(self) -> BattlePolicy:
        return self._battle_policy


class GUIJulietteCompetitor(JulietteCompetitor):

    def __init__(self, name: str = ""):
        super().__init__(name)

    @property
    def team_selection_policy(self) -> TeamSelectionPolicy:
        return GUITeamSelectionPolicy()

    @property
    def battle_policy(self) -> BattlePolicy:
        return GUIPlayer()