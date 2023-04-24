import itertools
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List

import numpy as np

from vgc.balance import DeltaRoster
from vgc.balance.archtype import std_move_dist, std_pkm_dist, std_team_dist
from vgc.datatypes.Objects import PkmMove, PkmFullTeam, PkmRoster, PkmMoveRoster

PkmId = int
MoveId = int


class MetaData(ABC):

    @abstractmethod
    def update_with_team(self, team: PkmFullTeam):
        pass

    @abstractmethod
    def update_with_delta_roster(self, delta: DeltaRoster):
        pass

    @abstractmethod
    def get_global_pkm_usage(self, pkm_id: PkmId) -> float:
        pass

    @abstractmethod
    def get_global_move_usage(self, move: PkmMove) -> float:
        pass

    @abstractmethod
    def get_pair_usage(self, pkm_ids: Tuple[PkmId, PkmId]) -> float:
        pass

    @abstractmethod
    def get_team(self, t) -> PkmFullTeam:
        pass

    @abstractmethod
    def get_n_teams(self) -> int:
        pass


class StandardMetaData(MetaData):

    def __init__(self, _max_history_size: int = 1e5, unlimited: bool = False, pkm_dist=std_pkm_dist,
                 move_dist=std_move_dist):
        # listings - moves, pkm, teams
        self._moves: PkmMoveRoster = []
        self._pkm: PkmRoster = []
        # global usage rate - moves, pkm
        self._move_usage: Dict[MoveId, int] = {}
        self._pkm_usage: Dict[PkmId, int] = {}
        # similarity matrix - moves, pkm
        self._d_move: Dict[Tuple[MoveId, MoveId], float] = {}
        self._d_pkm: Dict[Tuple[PkmId, PkmId], float] = {}
        self._d_overall_team = 0.0
        # history buffer - moves, pkm, teams
        self._move_history: List[PkmMove] = []
        self._pkm_history: List[PkmId] = []
        self._teammates_history: Dict[Tuple[PkmId, PkmId], int] = {}
        self._team_history: List[PkmFullTeam] = []
        # total usage count - moves, pkm, teams
        self._total_move_usage = 0
        self._total_pkm_usage = 0
        # if meta history size
        self._max_move_history_size: int = _max_history_size * 12
        self._max_pkm_history_size: int = _max_history_size * 3
        self._max_team_history_size: int = _max_history_size
        self._unlimited = unlimited
        # distance metrics
        self.pkm_dist = pkm_dist
        self.move_dist = move_dist

    def set_moves_and_pkm(self, roster: PkmRoster, move_roster: PkmMoveRoster):
        self._pkm = roster
        self._moves = move_roster
        for pkm in self._pkm:
            self._pkm_usage[pkm.pkm_id] = 0
        for move in self._moves:
            self._move_usage[move.move_id] = 0
        for m0, m1 in itertools.product(self._moves, self._moves):
            self._d_move[(m0.move_id, m1.move_id)] = self.move_dist(m0, m1)
        for p0, p1 in itertools.product(self._pkm, self._pkm):
            self._d_pkm[(p0.pkm_id, p1.pkm_id)] = self.pkm_dist(p0, p1, move_distance=lambda x, y: self._d_move[
                x.move_id, y.move_id])

    def update_with_delta_roster(self, delta: DeltaRoster):
        delta.apply(self._pkm)
        # clean history
        for pkm in self._pkm:
            self._pkm_usage[pkm.pkm_id] = 0
        for move in self._moves:
            self._move_usage[move.move_id] = 0
        self._total_move_usage = 0
        self._total_pkm_usage = 0
        # update similarity matrix
        for p0, p1 in itertools.product(self._pkm, self._pkm):
            self._d_pkm[(p0.pkm_id, p1.pkm_id)] = self.pkm_dist(p0, p1, move_distance=lambda x, y: self._d_move[
                x.move_id, y.move_id])
        # history buffer - moves, pkm, teams
        self._move_history: List[PkmMove] = []
        self._pkm_history: List[PkmId] = []
        self._teammates_history: Dict[Tuple[PkmId, PkmId], int] = {}
        self._team_history: List[PkmFullTeam] = []

    def update_with_team(self, team: PkmFullTeam):
        self._team_history.append(team.get_copy())
        # update distance
        for _team in self._team_history:
            self._d_overall_team = std_team_dist(team, _team,
                                                 pokemon_distance=lambda x, y: self._d_pkm[x.pkm_id, y.pkm_id])
        # update usages
        for pkm in team.pkm_list:
            self._pkm_usage[pkm.pkm_id] += 1
            for move in pkm.moves:
                self._move_usage[move.move_id] += 1
        for pkm0, pkm1 in itertools.product(team.pkm_list, team.pkm_list):
            if pkm0 != pkm1:
                pair = (pkm0.pkm_id, pkm1.pkm_id)
                if pair not in self._teammates_history.keys():
                    self._teammates_history[pair] = 1
                else:
                    self._teammates_history[pair] += 1
        # update total usages
        self._total_pkm_usage += 3
        self._total_move_usage += 12
        # remove from history past defined maximum length
        if len(self._team_history) > self._max_team_history_size and not self._unlimited:
            team, won = self._team_history.pop(0)
            for pkm0, pkm1 in itertools.product(team.pkm_list, team.pkm_list):
                if pkm0 != pkm1:
                    pair = (pkm0.pkm_id, pkm1.pkm_id)
                    self._teammates_history[pair] -= 1
                    if self._teammates_history[pair] == 0:
                        del self._teammates_history[pair]
            for _team in self._team_history:
                self._d_overall_team -= std_team_dist(team, _team, pokemon_distance=lambda x, y: self._d_pkm[x, y])
        if len(self._pkm_history) > self._max_pkm_history_size and not self._unlimited:
            for _ in range(3):
                old_pkm = self._pkm_history.pop(0)
                self._pkm_usage[old_pkm] -= 1
            self._total_pkm_usage -= 3
        if len(self._move_history) > self._max_move_history_size and not self._unlimited:
            for _ in range(12):
                old_move = self._move_history.pop(0)
                self._move_usage[old_move.move_id] -= 1
            self._total_move_usage -= 12

    def get_global_pkm_usage(self, pkm_id: PkmId) -> float:
        return self._pkm_usage[pkm_id] / max(1.0, self._total_pkm_usage)

    def get_global_move_usage(self, move: PkmMove) -> float:
        return self._move_usage[move.move_id] / max(1.0, self._total_move_usage)

    def get_pair_usage(self, pair: Tuple[PkmId, PkmId]) -> float:
        if pair not in self._teammates_history.keys():
            return 0.0
        return self._teammates_history[pair] / (
                self._pkm_usage[pair[0]] + self._pkm_usage[pair[1]] - self._teammates_history[pair])

    def get_team(self, t) -> PkmFullTeam:
        return self._team_history[t]

    def get_n_teams(self) -> int:
        return len(self._team_history)


def default_eval_func(meta: StandardMetaData, base_roster: PkmRoster) -> float:
    n_pkms = len(base_roster)
    dist = 0.0
    for i in range(n_pkms):
        dist += std_pkm_dist(meta._pkm[i], base_roster[i])
    dist /= n_pkms
    usage = np.array([meta.get_global_pkm_usage(i) for i in range(n_pkms)])
    balance = -np.std(usage)
    return 0.5 * dist + 0.5 * balance
