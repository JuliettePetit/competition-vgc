from copy import deepcopy
from typing import List

import PySimpleGUI as sg
import numpy as np
from enum import IntEnum


from vgc.behaviour import BattlePolicy
from vgc.datatypes.Constants import DEFAULT_PKM_N_MOVES, DEFAULT_PARTY_SIZE, TYPE_CHART_MULTIPLIER, DEFAULT_N_ACTIONS
from vgc.datatypes.Objects import PkmMove, GameState, Pkm, PkmTeam
from vgc.datatypes.Types import PkmStat, PkmStatus, PkmType, WeatherCondition


class RandomPlayer(BattlePolicy):
    """
    Agent that selects actions randomly.
    """

    def __init__(self, switch_probability: float = .15, n_moves: int = DEFAULT_PKM_N_MOVES,
                 n_switches: int = DEFAULT_PARTY_SIZE):
        super().__init__()
        self.n_actions: int = n_moves + n_switches
        self.pi: List[float] = ([(1. - switch_probability) / n_moves] * n_moves) + (
                [switch_probability / n_switches] * n_switches)

    def requires_encode(self) -> bool:
        return False

    def close(self):
        pass

    def get_action(self, g: GameState) -> int:
        print(g.teams[1].stage[PkmStat.ATTACK])
        return np.random.choice(self.n_actions, p=self.pi)



class OneTurnLookahead(BattlePolicy):
    """
    Greedy heuristic based agent designed to encapsulate a greedy strategy that prioritizes damage output.
    Source: http://www.cig2017.com/wp-content/uploads/2017/08/paper_87.pdf
    """

    def requires_encode(self) -> bool:
        return False

    def close(self):
        pass

    def get_action(self, g: GameState):
        # get weather condition
        weather = g.weather.condition

        # get my pkms
        my_team = g.teams[0]
        my_pkms = [my_team.active] + my_team.party

        # get opp team
        opp_team = g.teams[1]
        opp_active = opp_team.active
        opp_active_type = opp_active.type
        opp_defense_stage = opp_team.stage[PkmStat.DEFENSE]

        # get most damaging move from all my pkms
        damage: List[float] = []
        for i, pkm in enumerate(my_pkms):
            if i == 0:
                my_attack_stage = my_team.stage[PkmStat.ATTACK]
            else:
                my_attack_stage = 0
            for move in pkm.moves:
                if pkm.hp == 0.0:
                    damage.append(0.0)
                else:
                    damage.append(estimate_damage(move.type, pkm.type, move.power, opp_active_type, my_attack_stage,
                                                  opp_defense_stage, weather))
        move_id = int(np.argmax(damage))

        # decide between using an active pkm move or switching
        if move_id < 4:
            return move_id  # use current active pkm best damaging move
        if 4 <= move_id < 8:
            return 4  # switch to first party pkm
        else:
            return 5  # switch to second party pkm


def evaluate_matchup(pkm_type: PkmType, opp_pkm_type: PkmType, moves_type: List[PkmType]) -> float:
    # determine defensive matchup
    defensive_matchup = 0.0
    for mtype in moves_type + [opp_pkm_type]:
        defensive_matchup = min(TYPE_CHART_MULTIPLIER[mtype][pkm_type], defensive_matchup)
    return defensive_matchup


class TypeSelector(BattlePolicy):
    """
    Type Selector is a variation upon the One Turn Lookahead agent that utilizes a short series of if-else statements in
    its decision making.
    Source: http://www.cig2017.com/wp-content/uploads/2017/08/paper_87.pdf
    """

    def requires_encode(self) -> bool:
        return False

    def close(self):
        pass

    def get_action(self, g: GameState):
        # get weather condition
        weather = g.weather.condition

        # get my pkms
        my_team = g.teams[0]
        my_active = my_team.active
        my_party = my_team.party
        my_attack_stage = my_team.stage[PkmStat.ATTACK]

        # get opp team
        opp_team = g.teams[1]
        opp_active = opp_team.active
        opp_defense_stage = opp_team.stage[PkmStat.DEFENSE]

        # estimate damage my active pkm moves
        damage: List[float] = []
        for move in my_active.moves:
            damage.append(estimate_damage(move.type, my_active.type, move.power, opp_active.type, my_attack_stage,
                                          opp_defense_stage, weather))

        # get most damaging move
        move_id = int(np.argmax(damage))

        #  If this damage is greater than the opponents current health we knock it out
        if damage[move_id] >= opp_active.hp:
            return move_id

        # If not, check if are a favorable match. If we are lets give maximum possible damage.
        if evaluate_matchup(my_active.type, opp_active.type, list(map(lambda m: m.type, opp_active.moves))) >= 1.0:
            return move_id

        # If we are not switch to the most favorable matchup
        matchup: List[float] = []
        not_fainted = False
        for pkm in my_party:
            if pkm.hp == 0.0:
                matchup.append(0.0)
            else:
                not_fainted = True
                matchup.append(
                    evaluate_matchup(my_active.type, opp_active.type, list(map(lambda m: m.type, opp_active.moves))))

        if not_fainted:
            return int(np.argmax(matchup)) + 4

        # If our party has no non fainted pkm, lets give maximum possible damage with current active
        return move_id


class BFSNode:

    def __init__(self):
        self.a = None
        self.g = None
        self.parent = None
        self.depth = 0
        self.eval = 0.0


class BreadthFirstSearch(BattlePolicy):
    """
    Basic tree search algorithm that traverses nodes in level order until it finds a state in which the current opponent
    Pokemon is fainted. As a non-adversarial algorithm, the agent selfishly assumes that the opponent uses ”forceskip”
    (by selecting an invalid switch action).
    Source: http://www.cig2017.com/wp-content/uploads/2017/08/paper_87.pdf
    """

    def __init__(self):
        self.root = BFSNode()
        self.node_queue: List = [self.root]

    def requires_encode(self) -> bool:
        return False

    def close(self):
        pass

    def get_action(self, g) -> int:  # g: PkmBattleEnv
        self.root.g = g
        while len(self.node_queue) > 0:
            current_parent = self.node_queue.pop(0)
            # expand nodes of current parent
            for i in range(DEFAULT_N_ACTIONS):
                s, _, _, _ = current_parent.g.step([i, 99])  # opponent select an invalid switch action
                if s[0].teams[0].active.hp == 0:
                    continue
                elif s[0].teams[1].active.hp == 0:
                    a = i
                    while current_parent != self.root:
                        a = current_parent.a
                        current_parent = current_parent.parent
                    return a
                else:
                    node = BFSNode()
                    node.parent = current_parent
                    node.a = i
                    node.g = deepcopy(s[0])
                    self.node_queue.append(node)
        # if victory is not possible return arbitrary action
        return 0


def minimax_eval(s: GameState, depth):
    mine = s.teams[0].active
    opp = s.teams[1].active
    return mine.hp / mine.max_hp - 3 * opp.hp / opp.max_hp - 0.3 * depth


class Minimax(BattlePolicy):
    """
    Tree search algorithm that deals with adversarial paradigms by assuming the opponent acts in their best interest.
    Each node in this tree represents the worst case scenario that would occur if the player had chosen a specific
    choice.
    Source: http://www.cig2017.com/wp-content/uploads/2017/08/paper_87.pdf
    """

    def __init__(self):
        self.root = BFSNode()
        self.node_queue: List = [self.root]

    def requires_encode(self) -> bool:
        return False

    def close(self):
        pass

    def get_action(self, g) -> int:  # g: PkmBattleEnv
        self.root.g = g
        while len(self.node_queue) > 0:
            current_parent = self.node_queue.pop(0)
            # expand nodes of current parent
            for i in range(DEFAULT_N_ACTIONS):
                for j in range(DEFAULT_N_ACTIONS):  # opponent acts with his best interest, we iterate all joint actions
                    s, _, _, _ = current_parent.g.step([i, j])  # opponent select an invalid switch action
                    # gnore any node in which any of the agent's Pokemon faints
                    if s[0].teams[0].active.hp == 0:
                        continue
                    elif s[0].teams[1].active.hp == 0:
                        a = i
                        while current_parent != self.root:
                            a = current_parent.a
                            current_parent = current_parent.parent
                        return a
                    else:
                        node = BFSNode()
                        node.parent = current_parent
                        node.depth = node.parent.depth + 1
                        node.a = i
                        node.g = deepcopy(s[0])
                        node.eval = minimax_eval(s[0], node.depth)
                        self.node_queue.append(node)
                        # this could be improved by inserting with order
                        self.node_queue.sort(key=lambda n: n.eval, reverse=True)
        # if victory is not possible return arbitrary action
        return 0


class PrunedBFS(BattlePolicy):
    """
    Utilize domain knowledge as a cost-cutting measure by making modifications to the Breadth First Search agent.
    We do not simulate any actions that involve using a damaging move with a resisted type, nor does it simulate any
    actions that involve switching to a Pokemon with a subpar type matchup. Additionally, rather than selfishly
    assuming the opponent skips their turn in each simulation, the agent assumes its opponent is a One Turn Lookahead
    agent.
    Source: http://www.cig2017.com/wp-content/uploads/2017/08/paper_87.pdf
    """

    def __init__(self):
        self.root = BFSNode()
        self.node_queue: List = [self.root]
        self.opp = OneTurnLookahead()

    def requires_encode(self) -> bool:
        return False

    def close(self):
        pass

    def get_action(self, g) -> int:  # g: PkmBattleEnv
        self.root.g = g
        while len(self.node_queue) > 0:
            current_parent = self.node_queue.pop(0)
            # expand nodes of current parent
            for i in range(DEFAULT_N_ACTIONS):
                teams = current_parent.g.teams
                # skip traversing tree with non very effective moves
                if i < 4 and TYPE_CHART_MULTIPLIER[teams[0].active.moves[i].type][teams[1].active.type] < 0.5:
                    continue
                # skip traversing tree with switches to pokemons that are a bad type matchup against opponent active
                if i >= 4:
                    for move in teams[1].active.moves:
                        if move.power > 0.0 and TYPE_CHART_MULTIPLIER[move.type][teams[0].active.type] > 1.0:
                            continue
                # assume opponent follows OneTurnLookahead strategy
                j = self.opp.get_action(GameState((teams[1], teams[0]), current_parent.g.weather))
                s, _, _, _ = current_parent.g.step([i, j])
                if s[0].teams[0].active.hp == 0:
                    continue
                elif s[0].teams[1].active.hp == 0:
                    a = i
                    while current_parent != self.root:
                        a = current_parent.a
                        current_parent = current_parent.parent
                    return a
                else:
                    node = BFSNode()
                    node.parent = current_parent
                    node.a = i
                    node.g = deepcopy(s[0])
                    self.node_queue.append(node)
        # if victory is not possible return arbitrary action
        return 0


class GUIPlayer(BattlePolicy):

    def __init__(self, n_party: int = DEFAULT_PARTY_SIZE, n_moves: int = DEFAULT_PKM_N_MOVES):
        print(n_party)
        self.weather = sg.Text('                                                        ')
        self.opponent = sg.Text('                                                         ')
        self.active = sg.Text('                                                        ')
        self.moves = [sg.ReadFormButton('Move ' + str(i), bind_return_key=True) for i in range(n_moves)]
        self.party = [
            [sg.ReadFormButton('Switch ' + str(i), bind_return_key=True),
             sg.Text('                                      ')] for i in range(n_party)]
        layout = [[self.weather], [self.opponent], [self.active], self.moves] + self.party
        self.window = sg.Window('Pokemon Battle Engine', layout)
        self.window.Finalize()

    def requires_encode(self) -> bool:
        return False

    def get_action(self, g: GameState) -> int:
        """
        Decision step.

        :param g: game state
        :return: action
        """
        # weather
        self.weather.Update('Weather: ' + g.weather.condition.name)

        # get opp team
        opp_team = g.teams[1]
        opp_active = opp_team.active
        opp_active_type = opp_active.type
        opp_active_hp = opp_active.hp
        print(opp_active_hp)
        opp_status = opp_active.status
        opp_text = 'Opp: ' + opp_active_type.name + ' ' + str(opp_active_hp) + ' HP' + (
            '' if opp_status == PkmStatus.NONE else opp_status.name)
        opp_attack_stage = opp_team.stage[PkmStat.ATTACK]
        if opp_attack_stage != 0:
            opp_text += ' ATK ' + str(opp_attack_stage)
        opp_defense_stage = opp_team.stage[PkmStat.DEFENSE]
        if opp_defense_stage != 0:
            opp_text += ' DEF ' + str(opp_defense_stage)
        opp_speed_stage = opp_team.stage[PkmStat.SPEED]
        if opp_speed_stage != 0:
            opp_text += ' SPD ' + str(opp_speed_stage)
        self.opponent.Update(opp_text)

        # active
        my_team = g.teams[0]
        my_active = my_team.active
        my_active_type = my_active.type
        my_active_hp = my_active.hp
        my_status = my_active.status
        active_text = 'You: ' + my_active_type.name + ' ' + str(my_active_hp) + ' HP' + (
            '' if my_status == PkmStatus.NONE else my_status.name)
        active_attack_stage = my_team.stage[PkmStat.ATTACK]
        if active_attack_stage != 0:
            active_text += ' ATK ' + str(active_attack_stage)
        active_defense_stage = my_team.stage[PkmStat.DEFENSE]
        if active_defense_stage != 0:
            active_text += ' DEF ' + str(active_defense_stage)
        active_speed_stage = my_team.stage[PkmStat.SPEED]
        if active_speed_stage != 0:
            active_text += ' SPD ' + str(active_speed_stage)
        self.active.Update(active_text)

        # party
        my_party = my_team.party
        for i, pkm in enumerate(my_party):
            party_type = pkm.type
            party_hp = pkm.hp
            party_status = pkm.status
            party_text = party_type.name + ' ' + str(party_hp) + ' HP' + (
                '' if party_status == PkmStatus.NONE else party_status.name) + ' '
            self.party[i][1].Update(party_text)
            self.party[i][0].Update(disabled=(party_hp == 0.0))
        # moves
        my_active_moves = my_active.moves
        for i, move in enumerate(my_active_moves):
            move_power = move.power
            move_type = move.type
            self.moves[i].Update(str(PkmMove(power=move_power, move_type=move_type)))
        event, values = self.window.read()
        return self.__event_to_action(event)

    def __event_to_action(self, event):
        for i in range(len(self.moves)):
            if event == self.moves[i].get_text():
                return i
        for i in range(len(self.party)):
            if event == self.party[i][0].get_text():
                return i + DEFAULT_PKM_N_MOVES
        return -1

    def close(self):
        self.window.close()


class RuleBased(BattlePolicy):

    def __init__(self):
        self.rules = BattleRules()


    def requires_encode(self) -> bool:
        return False

    def close(self):
        pass

    def get_action(self, g: GameState) -> int:
        choice = self.rules.choose_action(g)
        print("feur")
        print(choice)
        return choice


class BattleRules():

    def choose_action(self, g: GameState) -> int:
        my_team = g.teams[0]
        my_stages = my_team.stage
        active_pkm = my_team.active

        opponennt_team = g.teams[1]
        opponent_stages = opponennt_team.stage
        opponent_active_pkm = opponennt_team.active
        choice = -1

        choice, _ = self.get_best_attack(my_team, my_stages, opponent_active_pkm, opponent_stages, g.weather)
        if (choice == -1):
            choice = self.get_best_switch(my_team, opponent_active_pkm, opponent_stages, g.weather)


        return choice


    def get_best_attack(self, my_team:PkmTeam, my_stages:List[int], opp_pkm: Pkm, opp_stages: List[int], weather:WeatherCondition) -> int:
        """
            Gets the best attack of a pokemon if that pokemon attacks this round
        """

        my_max_damage, best_attack = estimate_max_damage(my_team.active.moves, my_team.active.type, my_stages[PkmStat.ATTACK], opp_stages[PkmStat.DEFENSE], my_team.active.type, weather)
        opponent_max_damage, _ = estimate_max_damage(opp_pkm.moves, opp_pkm.type, opp_stages[PkmStat.ATTACK],my_stages[PkmStat.DEFENSE], opp_pkm.type, weather )

        if (self.should_attack(my_max_damage, opponent_max_damage, my_stages, opp_stages, my_team, opp_pkm, weather)):
            return best_attack, my_max_damage
        return -1, 0
    


    def should_attack(self, max_damage_dealt: int, max_damage_taken:int, my_stages:List[int], opp_stages: List[int], my_team: PkmTeam, opp_pkm: Pkm, weather:WeatherCondition, risky:bool = False) -> bool:
        if (can_free_kill(max_damage_dealt, my_stages[PkmStat.SPEED], opp_stages[PkmStat.SPEED], opp_pkm.hp)):
            return True
        elif (can_free_kill(max_damage_taken, opp_stages[PkmStat.SPEED], my_stages[PkmStat.SPEED], my_team.active.hp) and not can_switch_free_kill(my_team, opp_pkm, opp_stages, weather)):
            return False
        else:
            return True
        """(has_super_effective(my_team.active.moves, opp_pkm.type)) or
                    (not super_effective_type(opp_pkm.type, my_team.active.type)) and
                    (my_team.party[0].fainted and my_team.party[1].fainted()))  #the opponent's type isn't super effective"""
        

    


    def get_best_switch(self, my_team: PkmTeam, opp_pkm: Pkm, stage:List[int], weather: WeatherCondition) -> int:
        """
            Get the best switch possible if this player needs to switch
        """
        best_switch = -1
        opponent_max_damage = my_team.party[0].hp
        best_output_damages = opponent_max_damage #lowest number of hp lost over the whole party


        for pkm in my_team.party:
            
            damages, _ = estimate_max_damage(pkm.moves, opp_pkm.type, 0, stage[PkmStat.DEFENSE], pkm.type, weather)

            opponent_max_damage = min(damages, opponent_max_damage)

            if (opponent_max_damage < pkm.hp and opponent_max_damage < best_output_damages):
                best_switch = my_team.party.index(pkm)
                best_output_damages = opponent_max_damage
            
            """
            attack, output_damages = self.get_best_attack(my_team, [0] * 3, opp_pkm, stage, weather)
            if (attack >= 0 and output_damages > best_output_damages and not pkm.fainted()):
                best_output_damages = output_damages
                best_switch = my_team.party.index(pkm)
            """
        if (best_switch < 0 ):
            return 0
        return best_switch

def has_super_effective(moves:List[PkmMove], pkm_type: PkmType):
    for move in moves:
        if super_effective_type(move.type, pkm_type):
            return True
        
def super_effective_type(att_type: PkmType, def_type: PkmType):
    return TYPE_CHART_MULTIPLIER[att_type][def_type] > 1

def can_switch_free_kill(team_to_check: PkmTeam, current_pkm: Pkm, opp_attack:List[int],  weather: WeatherCondition) -> bool:
    """
        Check if at least one pkm in the team can survive an attack of the opponent's pkm
    """
    can_switch = False
    for pkm in team_to_check.party:
        max_dam = estimate_max_damage(current_pkm.moves, current_pkm.type,opp_attack[PkmStat.ATTACK], 0, pkm.type, weather)
        if (can_switch and not can_free_kill(max_dam, opp_attack[PkmStat.SPEED], 0, pkm.hp, False )):
            can_switch = True
    return can_switch


def strong(my_pkm : PkmType, opp_pkm : PkmType) -> bool:
    return TYPE_CHART_MULTIPLIER[my_pkm][opp_pkm] == 2.

def receive_null_dmg(my_pkm : PkmType, opp_pkm : PkmType) -> bool :
    opp_Null =  TYPE_CHART_MULTIPLIER[opp_pkm][my_pkm] == 0.
    my_pkm_not_null =  TYPE_CHART_MULTIPLIER[my_pkm][opp_pkm] != 0.
    return opp_Null & my_pkm_not_null

def estimate_damage(move_type: PkmType, pkm_type: PkmType, move_power: float, opp_pkm_type: PkmType,
                    attack_stage: int, defense_stage: int, weather: WeatherCondition) -> float:
    stab = 1.5 if move_type == pkm_type else 1.
    if (move_type == PkmType.WATER and weather == WeatherCondition.RAIN) or (
            move_type == PkmType.FIRE and weather == WeatherCondition.SUNNY):
        weather = 1.5
    elif (move_type == PkmType.WATER and weather == WeatherCondition.SUNNY) or (
            move_type == PkmType.FIRE and weather == WeatherCondition.RAIN):
        weather = .5
    else:
        weather = 1.
    stage_level = attack_stage - defense_stage
    stage = (stage_level + 2.) / 2 if stage_level >= 0. else 2. / (np.abs(stage_level) + 2.)
    damage = TYPE_CHART_MULTIPLIER[move_type][opp_pkm_type] * stab * weather * stage * move_power
    return damage



def can_free_kill(attacking_max_damage: int, attacking_speed:int, defending_speed:int, defending_hp:int, risky:bool = True) -> bool:
    '''
        return true if a pokemon can free kill another, meaning, if a pokemon has enough attack and speed to kill the opponent's pkm without taking a hit
        the risky parameter decides if a free kill includes a speed tie (50% chance the pkm will go second)
    '''
    if risky:
        return attacking_speed >= defending_speed and attacking_max_damage >= defending_hp
    return attacking_speed > defending_speed and attacking_max_damage >= defending_hp

def estimate_max_damage(moves: list[PkmMove], attacking_type: PkmType, attack_stage: int, defense_stage: int, defensive_type: PkmType, weather:WeatherCondition):
    '''
        estimates the max damage a pkm can do to another, and return the damage number and the attack which deals it
    '''
    max_damage = 0
    move_id = -1
    for move in moves:
        dam = estimate_damage(move.type, defensive_type, move.power, attacking_type, attack_stage, defense_stage, weather)
        if max_damage < dam :
            max_damage = dam
            move_id = moves.index(move)
    return max_damage, move_id



class Target_ennemy_stat(IntEnum):
    SPEED = -1
    ATTACK = -5
    DEFENSE = -5

class Status_priority(IntEnum):
    SLEEP = 0
    FROZEN = 1
    CONFUSED = 2
    PARALYZED = 3
    POISONED = 4
    BURNED = 5