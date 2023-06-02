from __future__ import annotations
from copy import deepcopy
from random import Random
import random
from typing import Dict, List, Tuple

import numpy as np
from vgc.datatypes.Objects import PkmMove
from vgc.competition.StandardPkmMoves import STANDARD_MOVE_ROSTER
from vgc.datatypes.Types import PkmType

from vgc.balance.meta import MetaData
from vgc.behaviour import TeamBuildPolicy, BattlePolicy
from vgc.behaviour.BattlePolicies import FirstPlayer, Minimax, TypeSelector, RandomPlayer
from vgc.datatypes.Constants import DEFAULT_PKM_N_MOVES, DEFAULT_TEAM_SIZE, MAX_HIT_POINTS
from vgc.datatypes.Objects import Pkm, PkmTemplate, PkmFullTeam, PkmRoster, PkmTeam
from vgc.engine.PkmBattleEnv import PkmBattleEnv

move_roster_len = len(STANDARD_MOVE_ROSTER)

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
        self.roster = purify_roster(pkm_list)
        self.genetic = GeneticAlgorithm(self.roster)
        


    def get_action(self, meta: MetaData) -> PkmFullTeam:
        """
        return the action given by the agent 
        """
        
        return PkmFullTeam(self.genetic.start())
    



class Chromosome():
    """
        a chromosome representing an individual, is constituted of different genes represented by attributes
    """
    def __init__(self) -> None:
        self.minimum_hp: int = 100
        self.types: List[PkmType] = []
        self.moves: List[PkmMove] = []
        self.fitness : float = 0.0

    def set_team(self, team: List[Pkm]):
        self.actual_team = team
        self.find_genes_from_pkm()



    def find_pkm_from_genes(self, roster: List[Pkm]):
        """
            sets the actual team from the chromosome's genes 
        """
        potential_pkm: List[Pkm] = []
        for pkm in roster :
            if pkm.hp >= self.minimum_hp:
                potential_pkm.append(pkm)

        team: List[Pkm] = []
        team_type: List[PkmType] = []
        team_moves: List[PkmMove] = []

        while len(team) < 3:
            max_score = 0
            chosen_pkm = None

            for pkm in potential_pkm:
                pkm_score = self.score_moves(pkm, team_moves) + self.score_type(pkm, team_type) + pkm.hp / MAX_HIT_POINTS
                if pkm_score > max_score:
                    max_score = pkm_score
                    chosen_pkm = pkm

            team.append(chosen_pkm)
            team_moves += chosen_pkm.moves
            team_type.append(chosen_pkm.type)
        self.actual_team = team
        self.find_genes_from_pkm()

    def score_moves(self, pkm: Pkm, team_moves: List[PkmMove]):
        """
            returns the score of a pokemon compared to the chromosome moves
        """
        score = 0
        for move in pkm.moves:
            if move in self.moves and not move in team_moves:
                score += 1
        return score
    
    def score_type(self, pkm: Pkm, team_types: List[PkmType]):
        """
            returns the score of a pokemon compared to the chromosome type
        """
        return 1 if pkm.type in self.types and not pkm.type in team_types else 0

    def find_genes_from_pkm(self):
        """
            turn the actual team in corresponding genes
        """
        self.types = []
        self.moves = []
        min_hp = 1000
        for pkm in self.actual_team:
            self.types.append(pkm.type)
            self.moves += pkm.moves
            min_hp = min(pkm.max_hp, min_hp)
        self.minimum_hp = min_hp

    def __lt__(self, other: Chromosome):
        return self.fitness < other.fitness


class GeneticAlgorithm():
    """
        Application of a genetic algorithm as a team building agent. 
    """
    def __init__(self, roster: List[Pkm], mutation_chances: float = 0.10, crossover_points: int = 6) -> None:
        self.nb_pop: int = 15
        self.population: List[Chromosome] = []
        self.nb_iterations: int = 30
        self.current_generation: int = 0
        self.mutation_chances = mutation_chances
        self.crossover_points = crossover_points
        self.roster = roster
        self.initialize_population(roster)

    def start(self):
        while self.current_generation < self.nb_iterations:
            self.fit_population()
            self.population = self.population[:self.nb_pop] + self.new_generation() 
            self.current_generation += 1
        self.fit_population()
        return self.population[0].actual_team



    def initialize_population(self, roster:list[Pkm]):
        """
        initializes the first generation of teams
        """
        roster = roster[:len(roster) - int(len(roster)%3)]
        list = []
        for i in range (len(roster)//3):
            team = roster[i : i+3]
            c = Chromosome()
            c.set_team(team)
            self.population.append(c)

    def fit_population(self):
        """
            matches within the roster to determine each chromosome's fitness value.
        """
        nb_matchup = 5
        over = (len(self.population) - 1) * nb_matchup
        self.reset_fitness()
        battle_agent = FirstPlayer()
        for i, ch in enumerate(self.population):
            for chi in self.population[i + 1: ]:
                score = score_matchup(ch.actual_team, chi.actual_team, battle_agent, nb_matchup)
                ch.fitness += score[0]
                chi.fitness += score[1]
            print(ch.fitness)
            ch.fitness = ch.fitness / over
        self.population.sort(reverse=True)
        print(f'-------------------------\nGeneration {self.current_generation} : \n  - The best fitness is {self.population[0].fitness}\n  - The worst fitness is {self.population[len(self.population)-1].fitness} ')

    def new_generation(self):
        next_generation:List[Chromosome] = []
        parents = self.selection()
        for i, ch in enumerate(parents):
            for chi in parents[i + 1:]:
                children = self.crossover(ch, chi)
                self.mutate(children[0])
                self.mutate(children[1])
                next_generation += [children[0], children[1]]
        return next_generation
        

    def reset_fitness(self):
        for ch in self.population:
            ch.fitness = 0

    def selection(self) -> List[Chromosome]:
        """
            selects the pairs of parents to create the next generation
        """
        nb_parents = 4
        selection = self.population[:nb_parents]
        return selection

    def crossover(self, ch1: Chromosome, ch2: Chromosome) -> Chromosome:
        """
        make a cross over of different genes of two individuals
        """
        rdm = Random()
        crossover_points = []
        genes1 = ch1.moves + ch1.types
        genes2 = ch2.moves + ch2.types

        child1_genes = []
        child2_genes = []

        while len(crossover_points) < self.crossover_points :
            nb = rdm.randint(0, len(genes1) - 1)
            if not nb in crossover_points :
                crossover_points.append(nb)

        # crossover_points = random.choices([x for x in range (len(ch1.moves) + len(ch1.types))], 4)

        # crossover
        child1 = Chromosome()
        child2 = Chromosome()

        for i in range (len(genes1)):
            if i in crossover_points:
                child1_genes.append(genes2[i])
                child2_genes.append(genes1[i])
            else:
                child1_genes.append(genes1[i])
                child2_genes.append(genes2[i])

        type_position = len(child1_genes) - len(ch1.types) 
        child1.moves = child1_genes[:type_position]
        child2.moves = child2_genes[:type_position]

        child1.types = child1_genes[type_position:]
        child2.types = child2_genes[type_position:]

        child1.find_pkm_from_genes(self.roster)
        child2.find_pkm_from_genes(self.roster)

        return child1, child2
        

    def mutate(self, chromosome: Chromosome):
        """
            mutate an individual's genes
        """
        moves = []
        for m in chromosome.moves:
            if random.random() < self.mutation_chances:
                moves.append(get_random_move_from_roster([m]))
            else:
                moves.append(m)
        types = []
        for t in chromosome.types:
            if random.random() < self.mutation_chances:
                types.append(get_random_type([t]))
            else:
                types.append(t)                  
        chromosome.moves = moves
        chromosome.types = types


def purify_roster(pkm_list: List[Pkm]):
    pkm_scores = [0] * len(pkm_list)
    pkm_ranking: List[int] = []
    battle_agent = RandomPlayer()
    for i, pkm0 in enumerate (pkm_list):
            for j, pkm1 in enumerate (pkm_list[i+1:]):
                score = score_matchup([pkm0], [pkm1], battle_agent, 3)
                pkm_scores[j] += score[0]
                pkm_scores[i+j -1] += score[1]
            insert_pkm_ranking(i, pkm_ranking, pkm_scores)
    return trim_roster(pkm_list)


def score_matchup(team0: List[Pkm], team1: List[Pkm], matchup_agent: BattlePolicy, nb_matchup: int) -> List[int]:
    """
        executes a match between two pokemon
    """
    t0 = PkmTeam([team0[0]] + team0[1:3])
    t1 = PkmTeam([team1[0]] + team1[1:3])
    final_scores = [0,0]
    env= PkmBattleEnv((t0, t1), encode=(False, False))
    for _ in range(nb_matchup):
        s = env.reset()
        t = False
        while not t:
            a0 = matchup_agent.get_action(s[0])
            a1 = matchup_agent.get_action(s[1])
            s, r, t, _ = env.step([a0, a1])
            # final_scores[0] += r[0]
            # final_scores[1] += r[1]
        final_scores[env.winner] += 1
    s = env.reset()
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


def get_random_move_from_roster(forbidden_moves: List[PkmMove]) -> PkmMove:
    move = random.choice(STANDARD_MOVE_ROSTER)
    while move in forbidden_moves:
        move = random.choice(STANDARD_MOVE_ROSTER)
    return move

def get_random_type(forbidden_types: List[PkmType]) -> PkmType:
    l = list(PkmType)
    pkm_type = random.choice(l) 
    while pkm_type in forbidden_types :
        pkm_type = random.choice(l) 
    return pkm_type