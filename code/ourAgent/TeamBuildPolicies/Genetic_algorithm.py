from random import Random
from typing import List

import numpy as np
from vgc.datatypes.Objects import PkmMove
from vgc.competition.StandardPkmMoves import STANDARD_MOVE_ROSTER
from vgc.datatypes.Types import PkmType

from vgc.balance.meta import MetaData
from vgc.behaviour import TeamBuildPolicy, BattlePolicy
from vgc.behaviour.BattlePolicies import TypeSelector, RandomPlayer
from vgc.datatypes.Constants import DEFAULT_PKM_N_MOVES, DEFAULT_TEAM_SIZE
from vgc.datatypes.Objects import Pkm, PkmTemplate, PkmFullTeam, PkmRoster, PkmTeam
from vgc.engine.PkmBattleEnv import PkmBattleEnv

class GeneticAlgorithmTeamBuilding(TeamBuildPolicy):
    """

    """

    def __init__(self, agent0: BattlePolicy = TypeSelector(), agent1: BattlePolicy = TypeSelector(), n_battles=10):
        self.roster = List[Pkm]
        self.agent0 = agent0
        self.agent1 = agent1
        self.n_battles = n_battles
        self.policy = None
        self.ver = -1

    def requires_encode(self) -> bool:
        return False

    def close(self):
        pass

    def set_roster(self, roster: PkmRoster, n: int = 21, ver: int = 0):
        """
        set the n best pokemon roster
        """
        pkm_list: List[Pkm] = []
        for pt in roster:
            pkm_list.append(pt.gen_pkm([0, 1, 2, 3]))
        purify_roster(pkm_list)



    def get_action(self, meta: MetaData) -> PkmFullTeam:
        """
        return the action given by the agent 
        """
        return []
    



class Chromosome():
    """
        a chromosome representing an individual, is constituted of different genes represented by attributes
    """
    def __init__(self, team: List[Pkm]) -> None:
        self.actual_team = team
        self.minimum_hp: int = 100
        self.types: List[PkmType] = []
        self.moves: List[PkmMove] = []
        self.find_genes_from_pkm()

    def find_pkm_from_genes(self):
        pass

    def find_genes_from_pkm(self):
        min_hp = 0
        for pkm in self.actual_team:
            self.types.append(pkm.type)
            self.moves += pkm.moves
            min_hp = min(pkm.max_hp, min_hp)
        self.minimum_hp = min_hp


class GeneticAlgorithm():
    """
        Application of a genetic algorithm as a team building agent
    """
    def __init__(self) -> None:
        self.nb_pop: int = 7
        self.population: List[Chromosome] = []
        self.nb_iterations: int = 100
        self.current_generation: int = 0
        


    def initialize_population(self, roster:list[Pkm]):
        """
        initializes the first generation of teams
        """
        roster = roster[:len(roster) - int(len(roster)%3)]
        list = []
        for i in range (len(roster)//3):
            team = roster[i : i+2]
            c = Chromosome(team)
            self.population.append(c)
        pass

    def fit_population(self):
        """
        matches within the roster, each team has a fit value and a set of genes.
        """
        pass

    def crossover(self, chromosome1: Chromosome, chromosome2: Chromosome):
        """
        make a cross over of different genes of two individuals
        """
        pass

    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """
        mutate an individual's genes
        """
        pass


def purify_roster(pkm_list: List[Pkm]):
    pkm_scores = [0] * len(pkm_list)
    pkm_ranking: List[int] = []
    battle_agent = RandomPlayer()
    for i, pkm0 in enumerate (pkm_list):
            for j, pkm1 in enumerate (pkm_list[i+1:]):
                score = score_matchup(pkm0, pkm1, battle_agent)
                pkm_scores[j] += score[i]
                pkm_scores[i+j] += score[i+j]
            insert_pkm_ranking(i, pkm_ranking)
    return trim_roster(pkm_list)


def score_matchup(pkm0: Pkm, pkm1: Pkm, matchup_agent: BattlePolicy, nb_matchup: int = 3):
    t0 = PkmTeam([pkm0])
    t1 = PkmTeam([pkm1])
    final_scores = [0,0]
    env= PkmBattleEnv((t0, t1), encode=(False, False))
    for _ in range(nb_matchup):
        s = env.reset()
        t = False
        while not t:
            a0 = matchup_agent.get_action(s[0])
            a1 = matchup_agent.get_action(s[1])
            s, r, t, _ = env.step([a0, a1])
            final_scores[0] += r[0]
            final_scores[1] += r[1]
    return final_scores

def insert_pkm_ranking(pkm_id: int, pkm_ranking: List[int], pkm_scores: List[int]):
    inserted = False
    i = 0
    score = pkm_scores[pkm_id]
    lgth = len(pkm_ranking)
    while i < lgth and not inserted:
        if pkm_scores[pkm_ranking[i]] < score:
            pkm_ranking.insert(i, pkm_id)
            inserted = True
        i += 1
    if not inserted:
        pkm_ranking.append(pkm_id)

def trim_roster(roster: List[Pkm], roster_size: int = 21):
    return roster[:roster_size]
