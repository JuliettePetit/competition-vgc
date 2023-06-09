""" Remote Clients """
from multiprocessing.connection import Client
from typing import Set

from vgc.balance import DeltaRoster
from vgc.behaviour import BattlePolicy, TeamSelectionPolicy, TeamBuildPolicy, TeamPredictor, BalancePolicy
from vgc.competition.Competition import Competitor
from vgc.datatypes.Objects import PkmFullTeam, PkmTeam, PkmRoster

ENCODE_TIMEOUT = 1.0
CLOSE_TIMEOUT = 1.0


class ProxyBattlePolicy(BattlePolicy):

    def __init__(self, conn: Client, timeout: float):
        self.conn: Client = conn
        self.timeout: float = timeout

    def get_action(self, s) -> int:
        # self.conn.settimeout(self.timeout)
        self.conn.send(('BattlePolicy', 'get_action', s))
        action: int = self.conn.recv()
        return action

    def requires_encode(self) -> bool:
        # self.conn.settimeout(ENCODE_TIMEOUT)
        self.conn.send(('BattlePolicy', 'requires_encode',))
        action: bool = self.conn.recv()
        return action

    def close(self):
        # self.conn.settimeout(CLOSE_TIMEOUT)
        self.conn.send(('BattlePolicy', 'close'))


class ProxyTeamSelectionPolicy(TeamSelectionPolicy):

    def __init__(self, conn: Client, timeout: float):
        self.conn: Client = conn
        self.timeout: float = timeout

    def get_action(self, s) -> Set[int]:
        # self.conn.settimeout(self.timeout)
        self.conn.send(('TeamSelectionPolicy', 'get_action', s))
        action: Set[int] = self.conn.recv()
        return action

    def requires_encode(self) -> bool:
        # self.conn.settimeout(ENCODE_TIMEOUT)
        self.conn.send(('TeamSelectionPolicy', 'requires_encode',))
        action: bool = self.conn.recv()
        return action

    def close(self):
        # self.conn.settimeout(CLOSE_TIMEOUT)
        self.conn.send(('TeamSelectionPolicy', 'close'))


class ProxyTeamBuildPolicy(TeamBuildPolicy):

    def __init__(self, conn: Client, timeout: float):
        self.conn: Client = conn
        self.timeout: float = timeout

    def set_roster(self, roster: PkmRoster, ver: int = 0):
        # self.conn.settimeout(self.timeout)
        self.conn.send(('TeamBuildPolicy', 'set_roster', (roster, ver)))

    def get_action(self, s) -> PkmFullTeam:
        # self.conn.settimeout(self.timeout)
        self.conn.send(('TeamBuildPolicy', 'get_action', s))
        action: PkmFullTeam = self.conn.recv()
        return action

    def requires_encode(self) -> bool:
        # self.conn.settimeout(ENCODE_TIMEOUT)
        self.conn.send(('TeamBuildPolicy', 'requires_encode',))
        action: bool = self.conn.recv()
        return action

    def close(self):
        # self.conn.settimeout(CLOSE_TIMEOUT)
        self.conn.send(('TeamBuildPolicy', 'close'))


class ProxyTeamPredictor(TeamPredictor):

    def __init__(self, conn: Client, timeout: float):
        self.conn: Client = conn
        self.timeout: float = timeout

    def get_action(self, s) -> PkmTeam:
        # self.conn.settimeout(self.timeout)
        self.conn.send(('TeamPredictor', 'get_action', s))
        action: PkmTeam = self.conn.recv()[0]
        return action

    def requires_encode(self) -> bool:
        # self.conn.settimeout(ENCODE_TIMEOUT)
        self.conn.send(('TeamPredictor', 'requires_encode',))
        action: bool = self.conn.recv()
        return action

    def close(self):
        # self.conn.settimeout(CLOSE_TIMEOUT)
        self.conn.send(('TeamPredictor', 'close'))


class ProxyBalancePolicy(BalancePolicy):

    def __init__(self, conn: Client, timeout: float):
        self.conn: Client = conn
        self.timeout: float = timeout

    def get_action(self, s) -> DeltaRoster:
        # self.conn.settimeout(self.timeout)
        self.conn.send(('BalancePolicy', 'get_action', s))
        action: DeltaRoster = self.conn.recv()
        return action

    def requires_encode(self) -> bool:
        # self.conn.settimeout(ENCODE_TIMEOUT)
        self.conn.send(('BalancePolicy', 'requires_encode',))
        action: bool = self.conn.recv()
        return action

    def close(self):
        # self.conn.settimeout(CLOSE_TIMEOUT)
        self.conn.send(('BalancePolicy', 'close'))


GET_TEAM_TIMEOUT = 1.0
SET_TEAM_TIMEOUT = 1.0
META_DATA_TIMEOUT = 1.0
NAME_TIMEOUT = 1.0
RESET_TIMEOUT = 1.0
CHANGE_TEAM_TIMEOUT = 1.0


class ProxyCompetitor(Competitor):

    def __init__(self, conn: Client):
        self.conn = conn
        self.battlePolicy = ProxyBattlePolicy(conn, 1.0)
        self.teamSelectionPolicy = ProxyTeamSelectionPolicy(conn, 1.0)
        self.teamBuildPolicy = ProxyTeamBuildPolicy(conn, 1.0)
        self.teamPredictor = ProxyTeamPredictor(conn, 1.0)
        self.balancePolicy = ProxyBalancePolicy(conn, 1.0)

    @property
    def battle_policy(self) -> BattlePolicy:
        return self.battlePolicy

    @property
    def team_selection_policy(self) -> TeamSelectionPolicy:
        return self.teamSelectionPolicy

    @property
    def team_build_policy(self) -> TeamBuildPolicy:
        return self.teamBuildPolicy

    @property
    def team_predictor(self) -> TeamPredictor:
        return self.teamPredictor

    @property
    def balance_policy(self) -> BalancePolicy:
        return self.balancePolicy

    @property
    def name(self) -> str:
        # self.conn.settimeout(NAME_TIMEOUT)
        self.conn.send(('name',))
        action: str = self.conn.recv()
        return action
