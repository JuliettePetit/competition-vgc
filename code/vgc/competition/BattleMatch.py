import random
from random import sample
from typing import Tuple, List, Optional

from vgc.balance.meta import MetaData
from vgc.behaviour import BattlePolicy
from vgc.competition.Competitor import Competitor, CompetitorManager
from vgc.datatypes.Constants import DEFAULT_MATCH_N_BATTLES, DEFAULT_TEAM_SIZE, DEFAULT_N_ACTIONS
from vgc.datatypes.Objects import PkmFullTeam, PkmTeam
from vgc.engine.HiddenInformation import hide_team
from vgc.engine.PkmBattleEnv import PkmBattleEnv
from vgc.util.generator.PkmTeamGenerators import PkmTeamGenerator


def team_selection(c: Competitor, my_team: PkmFullTeam, opp_team: PkmFullTeam,
                   full_team_size=DEFAULT_TEAM_SIZE) -> Tuple[PkmTeam, PkmTeam]:
    try:
        team_ids = list(c.team_selection_policy.get_action((my_team, opp_team)))
        opp_ids = list(c.team_selection_policy.get_action((opp_team, my_team)))
    except:
        team_ids = sample(range(full_team_size), DEFAULT_TEAM_SIZE)
        opp_ids = sample(range(full_team_size), DEFAULT_TEAM_SIZE)
    return my_team.get_battle_team(team_ids), opp_team.get_battle_team(opp_ids)


class BattleMatch:

    def __init__(self, competitor0: CompetitorManager, competitor1: CompetitorManager,
                 n_battles: int = DEFAULT_MATCH_N_BATTLES, debug: bool = False, render: bool = False,
                 meta_data: Optional[MetaData] = None, random_teams=False, update_meta=False):
        self.n_battles: int = n_battles
        self.cms: Tuple[CompetitorManager, CompetitorManager] = (competitor0, competitor1)
        self.wins: List[int] = [0, 0]
        self.debug = debug
        self.render_mode = 'ux' if render else 'console'
        self.finished = False
        self.meta_data = meta_data
        self.random_teams = random_teams
        self.update_meta = update_meta

    def run(self):
        c0 = self.cms[0].competitor
        c1 = self.cms[1].competitor
        team0 = self.cms[0].team
        team1 = self.cms[1].team
        a0 = c0.battle_policy
        a1 = c1.battle_policy
        # fully hide team information
        team0.hide()
        team1.hide()
        b = 0
        while b < self.n_battles:
            # reveal pkm identities
            team0.reveal_pkm()
            team1.reveal_pkm()
            # current information copy instances
            team0_view = team0.get_copy()
            hide_team(team0_view)
            team1_view = team1.get_copy()
            hide_team(team1_view)
            # full team predictions
            team1_p = self.__team_prediction(c0, team0_view)
            team0_p = self.__team_prediction(c1, team1_view)
            # self team selection and opponent prediction
            battle_team0, battle_team1_p = team_selection(c0, team0, team1_p)
            battle_team1, battle_team0_p = team_selection(c1, team1, team0_p)
            # hide pkm identities
            team0.hide_pkm()
            team1.hide_pkm()
            b += 1
            if self.debug:
                print('BATTLE ' + str(b) + '\n')
            winner = self.__run_battle(a0, a1, battle_team0, battle_team1, battle_team1_p, battle_team0_p)
            if self.wins[winner] > self.n_battles // 2:
                break
        if self.debug:
            print('MATCH RESULTS ' + str(self.wins) + '\n')
        a0.close()
        a1.close()
        if self.update_meta:
            self.meta_data.update_with_team(team0)
            self.meta_data.update_with_team(team1)
        self.finished = True

    def __team_prediction(self, c: Competitor, opp_team_view: PkmFullTeam) -> PkmFullTeam:
        if self.meta_data is None:
            return PkmFullTeam()
        else:
            try:
                return c.team_predictor.get_action((opp_team_view, self.meta_data))
            except:
                return PkmFullTeam()

    def __run_battle(self, a0: BattlePolicy, a1: BattlePolicy, team0: PkmTeam, team1: PkmTeam,
                     team1_p: Optional[PkmTeam], team0_p: Optional[PkmTeam]) -> int:
        env = PkmBattleEnv((team0, team1), debug=self.debug, encode=(a0.requires_encode(), a1.requires_encode()))
        env.set_predictions(team1_p, team0_p)
        s = env.reset()
        if self.debug:
            env.render(self.render_mode)
        t = False
        while not t:
            try:
                act0 = a0.get_action(s[0])
            except:
                act0 = random.randint(0, DEFAULT_N_ACTIONS-1)
            try:
                act1 = a1.get_action(s[1])
            except:
                act1 = random.randint(0, DEFAULT_N_ACTIONS-1)
            a = [act0, act1]
            s, _, t, v = env.step(a)
            if self.debug:
                env.render(self.render_mode)
        self.wins[env.winner] += 1
        return env.winner

    def winner(self) -> int:
        """
        Get winner.
        """
        return 0 if self.wins[0] > self.wins[1] else 1


class RandomTeamsBattleMatch(BattleMatch):

    def __init__(self, gen: PkmTeamGenerator, competitor0: CompetitorManager, competitor1: CompetitorManager,
                 n_battles: int = DEFAULT_MATCH_N_BATTLES, debug: bool = False, render: bool = False,
                 meta_data: Optional[MetaData] = None, random_teams=False):
        super().__init__(competitor0, competitor1, n_battles, debug, render, meta_data, random_teams)
        self.gen: PkmTeamGenerator = gen

    def run(self):
        a0 = self.cms[0].competitor.battle_policy
        a1 = self.cms[1].competitor.battle_policy
        tie = True
        while tie:
            team0 = self.gen.get_team().get_battle_team([0, 1, 2])
            team1 = self.gen.get_team().get_battle_team([0, 1, 2])
            if self.debug:
                print('BATTLE\n')
            winner0 = self.__run_battle(a0, a1, team0, team1, None, None)
            self.wins[winner0] += 1
            winner1 = self.__run_battle(a0, a1, team1, team0, None, None)
            self.wins[winner1] += 1
            tie = winner0 != winner1
        if self.debug:
            print('MATCH RESULTS ' + str(self.wins) + '\n')
        a0.close()
        a1.close()
        self.finished = True
