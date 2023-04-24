from typing import Tuple

from vgc.balance.meta import MetaData
from vgc.behaviour import TeamPredictor
from vgc.datatypes.Objects import PkmFullTeam


class NullTeamPredictor(TeamPredictor):

    def requires_encode(self) -> bool:
        return False

    def close(self):
        pass

    def get_action(self, d: Tuple[PkmFullTeam, MetaData]) -> PkmFullTeam:
        return PkmFullTeam()
