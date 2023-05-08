from __future__ import annotations

from copy import deepcopy
from typing import List

import PySimpleGUI as sg
import numpy as np
from enum import IntEnum

from vgc.competition.StandardPkmMoves import Struggle
from vgc.behaviour import BattlePolicy
from vgc.datatypes.Constants import DEFAULT_PKM_N_MOVES, DEFAULT_PARTY_SIZE, TYPE_CHART_MULTIPLIER, DEFAULT_N_ACTIONS, STATE_DAMAGE, SPIKES_2, SPIKES_3
from vgc.datatypes.Objects import PkmMove, GameState, Pkm, PkmTeam
from vgc.datatypes.Types import PkmStat, PkmStatus, PkmType, WeatherCondition, PkmEntryHazard


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


class AlphaBetaPruned(BattlePolicy):
    def __init__(self, depth: int):
        self.depth = 3
        #we can't really take the real impact of status so we give modifiers.
        #here, confused, asleep, paralyzed and frozen all give 0 because we always take the "worst case for us" (the min), this means that, in our calculations, the opponent will always break free in 1 turn, or ignore the ailment.
            
        self.worst_case_status = [PkmStatus.PARALYZED, PkmStatus.FROZEN]

        

    def requires_encode(self) -> bool:
        return False

    def close(self):
        pass

    def get_action(self, g: GameState) -> int:
        print("-----------------")
        player = 0
        tree = GameMatrix(g, 0)
        for i in range(DEFAULT_N_ACTIONS):
            for a in range(DEFAULT_N_ACTIONS):
                pass


    def alpha_beta_evaluation(self, state: GameMatrix, parent_state: GameMatrix)-> int:
        my_team = g.teams[0]
        my_active = my_team.active
        my_moves = my_active.moves

        opp_team = g.teams[1]
        opp_active = opp_team.active
        opp_moves = opp_active.moves
        
    
        
    def take_step(self, g: GameState, opp_choice: int, my_choice: int)-> GameMatrix:
        """
            simulate a battle round with certain specificities

            since we take te worst case for us, we will consider that :
                - every status that is not mandatory (ie. that is not burned or poisoned) will take effect EVERYTIME for us, and NEVER for the ennemy
                - we always go second if there is a speed tie
        """

        player_0_attack = my_choice < DEFAULT_PKM_N_MOVES
        player_1_attack = my_choice < DEFAULT_PKM_N_MOVES

        self.simulate_switch(g, [my_choice, opp_choice])


        will_play = self.will_play_next_turn(g.teams[0])

        if player_1_attack:
            self.simulate_attack(g.teams[1].active,g.teams[0].active, g.teams[1].stage, g.teams[0].stage, g.teams[1].active.moves[opp_choice])

        if player_0_attack:
            if will_play:
                self.simulate_attack(g.teams[0].active,g.teams[1].active, g.teams[0].stage, g.teams[1].stage, g.teams[0].active.moves[my_choice])
            if g.teams[0].confused:
                g.teams[0].active.hp -= STATE_DAMAGE

        self.simulate_weather_damage(g)
        self.simulate_status_damage(g)

        self.simulate_post_battle(g)

        return GameMatrix(g, will_play)

    def simulate_weather_damage(self, g: GameState):
        for team in g.teams:
            pkm = team.active
            if will_take_weather_damage(pkm, g.weather.condition):
                pkm.hp = max(0, pkm.hp - STATE_DAMAGE)

    def simulate_status_damage(self, g: GameState):
        for team in g.teams:
            pkm = team.active
            if pkm.status == PkmStatus.POISONED or pkm.status == PkmStatus.BURNED:
                pkm.hp = max(0, pkm.hp - STATE_DAMAGE)
    
    def simulate_post_battle(self, g: GameState):
        if self.weather.condition != WeatherCondition.CLEAR:
            self.n_turns_no_clear += 1

            # clear weather if appropriated
            if self.n_turns_no_clear > 5:
                self.weather.condition = WeatherCondition.CLEAR
                self.n_turns_no_clear = 0


    def simulate_switch(self, g: GameState, actions: List[int]):
        """
            there is no randomness in a switch, this means that this code is, except for a few small changes, the one used in the switch of the PkmBatleEnv file (minus the print statements)
        """
        for i, team in enumerate(g.teams):
            pos = actions[i] - DEFAULT_PKM_N_MOVES
            if 0 <= pos < (team.size() - 1):
                if not team.party[pos].fainted():
                    team.switch(pos)
                    deal_hazard_damage(g, team.active)
                    

    def simulate_attack(self, my_pkm: Pkm,opp_pkm: Pkm, my_stages: List[int], opp_stages: List[int], move: PkmMove):


        if move.pp > 0:
            move.pp -= 1
        else:
            move = Struggle


        fixed_damage = move.fixed_damage
        if fixed_damage > 0. and TYPE_CHART_MULTIPLIER[move.type][opp_pkm.type] > 0.:
            damage = fixed_damage
        else:
            stab = 1.5 if move.type == opp_pkm.type else 1.
            if (move.type == PkmType.WATER and self.weather.condition == WeatherCondition.RAIN) or (
                    move.type == PkmType.FIRE and self.weather.condition == WeatherCondition.SUNNY):
                weather = 1.5
            elif (move.type == PkmType.WATER and self.weather.condition == WeatherCondition.SUNNY) or (
                    move.type == PkmType.FIRE and self.weather.condition == WeatherCondition.RAIN):
                weather = .5
            else:
                weather = 1.
            stage_level = my_stages[PkmStat.ATTACK] - opp_stages[PkmStat.DEFENSE]
            stage = (stage_level + 2.) / 2 if stage_level >= 0. else 2. / (np.abs(stage_level) + 2.)
            multiplier = TYPE_CHART_MULTIPLIER[move.type][opp_pkm.type] if move != Struggle else 1.0
            damage = multiplier * stab * weather * stage * move.power
        opp_pkm.hp = max(0, opp_pkm.hp - damage)
        my_pkm.hp = min(my_pkm.max_hp, my_pkm.hp + move.recover)


    def will_play_next_turn(self, team: PkmTeam) -> bool:
        return team.active.status in self.worst_case_status or ((team.confused or team.active.asleep()) and will_free_next_turn(team))


def will_take_weather_damage(pkm: Pkm, weather: WeatherCondition):
    return (weather == WeatherCondition.SANDSTORM and (pkm.type != PkmType.ROCK and pkm.type != PkmType.GROUND and pkm.type != PkmType.STEEL)) or (weather == WeatherCondition.HAIL and (pkm.type != PkmType.ICE))


def deal_hazard_damage(team: PkmTeam, pkm: Pkm):
    spikes = team.entry_hazard[PkmEntryHazard]
    if spikes and pkm.type != PkmType.FLYING:
        hp_pre_haz = pkm.hp
        pkm.hp -= STATE_DAMAGE if spikes <= 1 else SPIKES_2 if spikes == 2 else SPIKES_3
        pkm.hp = 0. if pkm.hp < 0. else pkm.hp
    return pkm.hp - hp_pre_haz

def will_free_next_turn(team: PkmTeam):
    return (team.n_turns_confused == 3 or not team.confused) and (not team.active.asleep() or team.active.n_turns_asleep == 3 )




class GameMatrix():
    def __init__(self, state: GameState, player_has_played : bool):
        self.state: GameState = state
        self.value = 0
        self.is_goal = state.teams[1].active.fainted()
        self.player_has_played = player_has_played

        '''
            payoff matrix, with a shape of :
                                                    opp_move
                                move1   move2   move3   move4   swch1   swch2
                    move1   [    res     res     res     res     res     res    ]
                    move2   [    res     res     res     res     res     res    ]
            my_move move3   [    res     res     res     res     res     res    ]
                    move4   [    res     res     res     res     res     res    ]
                    swch1   [    res     res     res     res     res     res    ]
                    swch2   [    res     res     res     res     res     res    ]

        '''
        self.matrix: List[List[tuple(int, GameMatrix)]] = [ [tuple(0, None) * DEFAULT_N_ACTIONS] * DEFAULT_N_ACTIONS ] 
        print(self.matrix)

    def evaluate(self,parent: GameMatrix, fct: callable):
        self.value = fct(self.state, parent)

    def set_payoff(self, moves:tuple(int,int), value: int, state: GameMatrix ):
        self.matrix[moves[0]][moves[1]] = tuple(value, state)
    





