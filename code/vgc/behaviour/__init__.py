from abc import ABC, abstractmethod
from typing import Any, Set, Union, List, Tuple

from vgc.balance import DeltaRoster
from vgc.balance.meta import MetaData
from vgc.balance.restriction import VGCDesignConstraints
from vgc.datatypes.Objects import PkmFullTeam, GameState, PkmRoster


class Behaviour(ABC):

    @abstractmethod
    def get_action(self, s) -> Any:
        pass

    @abstractmethod
    def requires_encode(self) -> bool:
        pass

    @abstractmethod
    def close(self):
        pass


class BattlePolicy(Behaviour):

    @abstractmethod
    def get_action(self, s: Union[List[float], GameState]) -> int:
        pass


class TeamSelectionPolicy(Behaviour):

    @abstractmethod
    def get_action(self, s: Tuple[PkmFullTeam, PkmFullTeam]) -> Set[int]:
        pass


class TeamBuildPolicy(Behaviour):

    @abstractmethod
    def set_roster(self, roster: PkmRoster, ver: int = 0):
        pass

    @abstractmethod
    def get_action(self, s: MetaData) -> PkmFullTeam:
        pass


class TeamPredictor(Behaviour):

    @abstractmethod
    def get_action(self, s: Tuple[PkmFullTeam, MetaData]) -> PkmFullTeam:
        pass


class BalancePolicy(Behaviour):

    @abstractmethod
    def get_action(self, s: Tuple[PkmRoster, MetaData, VGCDesignConstraints]) -> DeltaRoster:
        pass
