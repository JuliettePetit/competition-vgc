from collections import defaultdict
from copy import deepcopy
from typing import List

import PySimpleGUI as sg
import numpy as np
from vgc.engine import PkmBattleEnv

from vgc.behaviour import BattlePolicy
from vgc.datatypes.Constants import DEFAULT_PKM_N_MOVES, DEFAULT_PARTY_SIZE, TYPE_CHART_MULTIPLIER, DEFAULT_N_ACTIONS
from vgc.datatypes.Objects import PkmMove, GameState
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
        return np.random.choice(self.n_actions, p=self.pi)


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
        self.g: PkmBattleEnv = None
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
                    # ignore any node in which any of the agent's Pokemon faints
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

class FirstPlayer(BattlePolicy):
    """
    Agent rules based
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
        # get weather condition
        weather = g.weather.condition

        # get my pkms
        my_team = g.teams[0]
        my_active = my_team.active

        # get opp team
        opp_team = g.teams[1]
        opp_active = opp_team.active
        opp_active_type = opp_active.type
        opp_defense_stage = opp_team.stage[PkmStat.DEFENSE]

        # if it's sunny
        if(weather == 1):
            deal_with_weather(my_active.type, my_team.party[0], my_team.party[1], 2, 1)

        # if it's raining 
        if(weather == 2):
            deal_with_weather(my_active.type, my_team.party[0], my_team.party[1], 1, 2)

        # if there is a sandstorm
        if(weather == 3):
            # I switch to a rock if i have one
            if(my_team.party[0].type == 3):
                switch(my_team.party[0], 0)
            elif(my_team.party[1].type == 3):
                switch(my_team.party[1], 1)
            else:
                pass


        # if pkm is strong, or is the only one alive, he uses his best attack
        if (strong(my_active.type, opp_active_type) or (my_team.party[0].fainted and my_team.party[1].fainted)):
            # estimate damage my active pkm moves
            damage: List[float] = []
            for move in my_active.moves:
                damage.append(estimate_damage(move.type, my_active.type, move.power, opp_active.type, my_team.stage[PkmStat.ATTACK],
                                            opp_defense_stage, weather))

            # get most damaging move
            return int(np.argmax(damage))
        
        # if opp pkm makes no damage and I do, I attack (least damage)  => I am not strong
        if (receive_null_dmg(my_active.type, opp_active_type)):
            # estimate damage my active pkm moves
            damage: List[float] = []
            for move in my_active.moves:
                damage.append(estimate_damage(move.type, my_active.type, move.power, opp_active.type, my_team.stage[PkmStat.ATTACK],
                                            opp_defense_stage, weather))

            # get least damaging move to save the most powerful to later
            return int(np.argmin(damage))
        
        # if one of my other pkm has this (and I don't), I switch
            # with the first 
        if(strong(my_team.party[0].type, opp_active_type) or receive_null_dmg(my_team.party[0].type, opp_active_type)):
            switch(my_team.party[0],0)
            # with the second
        if(strong(my_team.party[1].type, opp_active_type) or receive_null_dmg(my_team.party[1].type, opp_active_type)):
            switch(my_team.party[1],1)
            
        # if all my pokemon are weak against opponent
            # I switch if the active one is dealing 0 dmg or is confused or paralyzed
        if(TYPE_CHART_MULTIPLIER[my_active][opp_active] == 0 or my_active.status==1 or my_active.status==3):
            if(strong(my_team.party[0].type, opp_active_type) or receive_null_dmg(my_team.party[0].type, opp_active_type)):
                switch(my_team.party[0],0)
            else:
                switch(my_team.party[1],1)
            # I don't switch if I have stages
        #if()
            # I switch if I'm under 100 hp or the opp is strong against me and not the others
        if(strong(opp_active, my_active) and not strong(opp_active, my_team.party[0])):
            switch(my_team.party[0],0)
        elif(strong(opp_active, my_active) and not strong(opp_active, my_team.party[1])):
            switch(my_team.party[1],1)
            
        # strongest attack
        # estimate damage my active pkm moves
        damage: List[float] = []
        for move in my_active.moves:
            damage.append(estimate_damage(move.type, my_active.type, move.power, opp_active.type, my_team.stage[PkmStat.ATTACK],
                                        opp_defense_stage, weather))

        # get least damaging move to save the most powerful for later
        return int(np.argmax(damage))

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

def switch(pkm, nb: int):
    # switch if alive
    if(not pkm.fainted):
        if(nb == 0):
            return 4
        else:
            return 5

def deal_with_weather(my_active_type: PkmType, first_pkm, second_pkm, type_avoided: PkmType, type_wanted: PkmType):
    # switch if my active pokemon is of the avoided type and one of my pkm is not
    if(my_active_type == type_avoided and (first_pkm.type != type_avoided or second_pkm.type != type_avoided)):
        # if my team have a wanted type pokemon, I will take it
        if(first_pkm.type == type_wanted or second_pkm.type == type_avoided or  first_pkm.hp > second_pkm.hp):
            switch(first_pkm,0)
        # else I will take one that is not of the avoided type
        else: 
            switch(second_pkm,1)
    else:
        pass

    # switch if I have a wanted type pokemon 
    if(first_pkm.type == type_wanted):
        switch(first_pkm,0)
    elif(second_pkm.type == type_wanted):
        # switch if alive
        switch(second_pkm,1)
    else:
        pass




class MonteCarloNode():
    def __init__(self, state: GameState, untried_actions, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = [0, 0] #defaultdict(int)
        # self._results[1] = 0
        # self._results[-1] = 0
        #self._untried_actions:list = None
        self._untried_actions:list = untried_actions
        return

    


class MonteCarloPlayer(BattlePolicy):
    """
    Monte Carlo tree search based
    """

    #def __init__(self):
        #self.root = MonteCarloNode()
        #self.node_queue: List = [self.root]
    
    def random_opponent_action(self, g):
        #min = Minimax()
        #return min.get_action(g)
        return 0

    def requires_encode(self) -> bool:
        return False

    def close(self):
        pass

    def untried_actions(self, g):
        self._untried_actions = self.get_legal_actions(g)
        return self._untried_actions

    def n(self, node:MonteCarloNode):
        return node._number_of_visits
    
    def q(self, node:MonteCarloNode):
        wins = node._results[1]
        loses = node._results[0]
        return wins - loses


    def expand(self, node: MonteCarloNode):
        state = deepcopy(node.state)
        action = node._untried_actions.pop()   #our
        opp_action = self.random_opponent_action(state) #opponent
        actions = [action, opp_action]
        state.step(actions)# state of the entire game
        untried_actions = self.untried_actions(state)
        child_node = MonteCarloNode(
            state, untried_actions, parent=node, parent_action=action)

        node.children.append(child_node)
        return child_node 
    
    def is_game_over(self, g: GameState):
        return (g.teams[0].active.fainted() and len(g.teams[0].get_not_fainted())==0) or (g.teams[1].active.fainted() and len(g.teams[1].get_not_fainted())==0) #note: not fainted check only in party and not the current pkm

    def is_terminal_node(self, node: MonteCarloNode):
        return self.is_game_over(node.state)

    def rollout(self, node: MonteCarloNode):
        current_rollout_state = deepcopy(node.state)
        
        while not self.is_game_over(current_rollout_state):
        
            possible_moves = self.get_legal_actions(current_rollout_state)
            #print(current_rollout_state.teams[0].active.hp)

            action = self.rollout_policy(possible_moves)
            opp_action = self.random_opponent_action(current_rollout_state) #opponent
            actions = [action, opp_action]
            current_rollout_state.step(actions)

        return self.game_result(current_rollout_state)
    
    def backpropagate(self, result, node:MonteCarloNode):
        node._number_of_visits += 1.
        node._results[result] += 1.
        if node.parent:
            self.backpropagate(result, node.parent)

    def is_fully_expanded(self, node:MonteCarloNode):
        return len(node._untried_actions) == 0

    def best_child(self, node:MonteCarloNode, c_param=0.1):
        choices_weights =[]
        for c in node.children:
            tmp = self.q(c) / self.n(c)
            choices_weights.append((self.q(c) / self.n(c)) + c_param * np.sqrt((2 * np.log(self.n(node)) / self.n(c))))
            #print(choices_weights)
        return node.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]

    def _tree_policy(self, node: MonteCarloNode):
        current_node = node
        while not self.is_terminal_node(current_node):

            if not self.is_fully_expanded(current_node):
                return self.expand(current_node)
            else:
                current_node = self.best_child(current_node)
        return current_node

    def best_action(self,node: MonteCarloNode):
        simulation_no = 100
        for i in range(simulation_no):
            v = self._tree_policy(node)
            reward = self.rollout(v)
            self.backpropagate(reward, v)

        for c in node.children:
            print(f"action {c.parent_action}., loses: {c._results[0]}.")
            print(f"action {c.parent_action}., wins: {c._results[1]}.")
            print("-----------------------------------------------------")
        return self.best_child(node, c_param=0.)
    
    def get_legal_actions(self, g: GameState): 
        actions = []
        my_team = g.teams[0]
        my_moves = my_team.active.moves
        #switch if alive
        nb=0
        for pkm in my_team.party:
            if(not pkm.fainted):
                if(nb == 0):
                    nb = 1
                    actions.append(4)
                else:
                    actions.append(5)
        #attack if available
        move_nb = 0
        for move in my_moves:
            if move.pp > 0: #move.prob > 0 and 
                actions.append(move_nb)
            move_nb += 1
        return actions


    def game_result(self, g: GameState):
        '''
        Returns 1 or 0 or -1 depending
        on your state corresponding to win,
        draw or a loss.
        '''
        if(g.teams[0].active.fainted() and len(g.teams[0].get_not_fainted())==0):
            return 0 #-1
        if(g.teams[1].active.fainted() and len(g.teams[1].get_not_fainted())==0):
            return 1 #1
        """if len(g.teams[0].get_not_fainted()) <= 0 :
            return 0"""

    def get_action(self, g: GameState) -> int:  # g: PkmBattleEnv
        untried_actions = self.untried_actions(g)
        root = MonteCarloNode(g, untried_actions)
        selected_node : MonteCarloNode = self.best_action(root)
        print(selected_node.parent_action)
        return selected_node.parent_action