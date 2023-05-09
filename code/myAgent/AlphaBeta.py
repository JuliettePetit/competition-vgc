from __future__ import annotations
from ast import Tuple
from copy import deepcopy
from typing import List

import numpy as np

from vgc.behaviour import BattlePolicy
from vgc.competition.Competitor import Competitor
from vgc.competition.StandardPkmMoves import Struggle
from vgc.datatypes.Constants import DEFAULT_PKM_N_MOVES, SPIKES_2, SPIKES_3, STATE_DAMAGE, TYPE_CHART_MULTIPLIER
from vgc.datatypes.Objects import GameState, Pkm, PkmMove, PkmTeam
from vgc.datatypes.Types import PkmEntryHazard, PkmStat, PkmStatus, PkmType, WeatherCondition

class AlphaBetaCompetitor(Competitor):
    def __init__(self, name: str = "AlphaBeta"):
        self._name = name
        self._battle_policy = AlphaBeta(2)

    @property
    def name(self):
        return self._name

    @property
    def battle_policy(self) -> BattlePolicy:
        return self._battle_policy

class AlphaBeta(BattlePolicy):
    def __init__(self, max_depth: int):
        self._max_depth = max_depth  
        self.worst_case_status = [PkmStatus.PARALYZED, PkmStatus.FROZEN, PkmStatus.SLEEP]
        

    def requires_encode(self) -> bool:
        return False

    def close(self):
        pass

    def get_action(self, g: GameState) -> int:
        print("-----------------")
        origin = AlphaBetaNode(g.teams[0], g.teams[1], g.weather, None, 0, 0, True, -1)
        depth = 0
        current_parent = origin
        nodes_stack = [origin]

        #recursive version:
        node: AlphaBetaNode = self.alpha_beta_search(origin, g, 0, -1000, 1000)[1]


        #non-recursive version
        '''
        while len(nodes_stack) > 0:
            current_node = nodes_stack[len(nodes_stack) -1]
            current_node.prune()
            if len(current_node.left_actions) <= 0 or (current_node.depth > self._max_depth and current_node.turn_end):
                nodes_stack.pop()
            else:
                action = current_node.left_actions.pop()
        '''

        choice_node: AlphaBetaNode = node
        print('------------------------------------------------------')
        print(node.value)
        print(node.action)

        while not node == origin:
            choice_node = node
            node = choice_node.parent
            


        print(choice_node.action)
        return choice_node.action

                


            
    
    def alpha_beta_search(self, node: AlphaBetaNode, g: GameState, depth: int, alpha: int, beta: int) -> Tuple(int, AlphaBetaNode):
        """
            recursive version of the alpha beta algorithm in all of it's spaghetti code beauty
        """
        print('------------------------------------------------------------')
        print("PARENT :")
        print(f'player : {node.player} \naction : {node.action}')
        print(f'possible actions : {node.possible_action}')
        print(f"depth : {depth}")

        print('\n')

        if len(node.possible_action) <= 0 or (depth >= self._max_depth and node.turn_end) or node.is_terminal:
            print("node value and choice, end of tree")
            print(node.value)
            print(node.action)
            return node.value, node
        else:
            new_node: AlphaBetaNode = None
            for action in node.possible_action:
                new_g = deepcopy(g)
                new_state = self.take_step(node, new_g, action[1], not node.turn_end, action[0])
                if node.player == 0:
                    value = -1000
                    val, new_node = self.alpha_beta_search(new_state, new_g, depth + 1, alpha, beta)
                    value = max(value, val)
                    if value > beta:
                        break
                    alpha = max(alpha, value)
                    
                else:
                    value = 1000
                    val, new_node = self.alpha_beta_search(new_state, new_g, depth + 1, alpha, beta)
                    value = min(value, val)
                    if value < alpha:
                        break
                    beta = min(beta, value)

            return new_node.value, new_node
            

            
                
            


    
        
    
        
    def take_step(self, parent: AlphaBetaNode, g: GameState, my_choice: int, should_turn_end:bool, player:int)-> AlphaBetaNode:
        """
            simulate a battle round with certain specificities

            since we take te worst case for us, we will consider that :
                - every status that is not mandatory (ie. that is not burned or poisoned) will take effect EVERYTIME for us, and NEVER for the ennemy
                - we always go second if there is a speed tie

            the round is cut in two so that it can be accomodated to a tree with the alpha-beta algorithm, this means that, at anytime, only one move is used in this function. 

            the turn order is kept by the fact that the potential moves from a certain states allow for different players order depending on their choices, in that sense, switching, for example, will always be done first

            if the current state indicates the end of a turn, then end of turn effect will all apply at once.

            will break if one pokemon faint, meaning the algorithm won't be able to generate proper future outcomes, as we are not dealing with that case, 
            but we actually don't care about this because having a fainted pokemon on either side is considered a terminal state (good or bad), meaning we won't even try to generate it
        """
        player_attack = my_choice < DEFAULT_PKM_N_MOVES
        self.simulate_switch(g, my_choice, player)
        will_play = True
        if (player == 0):
            will_play = self.will_play_next_turn(g.teams[0])
        
        if player_attack:
            if will_play:
                self.simulate_attack(g.teams[player].active,g.teams[(player + 1) % 2].active, g.teams[player].stage, g.teams[(player + 1) %2].stage, g.teams[player].active.moves[my_choice], g.weather)
                if g.teams[player].confused:
                    g.teams[player].active.hp -= STATE_DAMAGE
        if should_turn_end:
            self.simulate_weather_damage(g)
            self.simulate_status_damage(g)
            self.simulate_post_battle(g)
        return AlphaBetaNode(g.teams[0], g.teams[1], g.weather, parent, parent.get_next_attacking_player(), parent.depth + 1, should_turn_end, my_choice)

    def simulate_weather_damage(self, g: GameState):
        """
            simulate the damage dealt by the weather
        """
        for team in g.teams:
            pkm = team.active
            if will_take_weather_damage(pkm, g.weather.condition):
                pkm.hp = max(0, pkm.hp - STATE_DAMAGE)

    def simulate_status_damage(self, g: GameState):
        """
            simulate damages dealt by status ailment 
        """
        for team in g.teams:
            pkm = team.active
            if pkm.status == PkmStatus.POISONED or pkm.status == PkmStatus.BURNED:
                pkm.hp = max(0, pkm.hp - STATE_DAMAGE)
    
    def simulate_post_battle(self, g: GameState):
        """
        simulate if the weather will be cleared or not
        """
        if g.weather.condition != WeatherCondition.CLEAR:
            g.n_turns_no_clear += 1
            if g.n_turns_no_clear > 5:
                g.weather.condition = WeatherCondition.CLEAR
                g.n_turns_no_clear = 0


    def simulate_switch(self, g: GameState, action: int, player:int ):
        """
            simulate a switch.
            there is no randomness in a switch, this means that this code is, except for a few small changes, the one used in the switch of the PkmBatleEnv file (minus the print statements)
        """
        pos = action - DEFAULT_PKM_N_MOVES
        if 0 <= pos < (g.teams[player].size() - 1):
            if not g.teams[player].party[pos].fainted():
                g.teams[player].switch(pos)
                deal_hazard_damage(g.teams[player], g.teams[player].active)
                    

    def simulate_attack(self, my_pkm: Pkm,opp_pkm: Pkm, my_stages: List[int], opp_stages: List[int], move: PkmMove, weather_condition: WeatherCondition):
        """
            simulate one attack
        """
        if move.pp > 0:
            move.pp -= 1
        else:
            move = Struggle

        fixed_damage = move.fixed_damage

        if fixed_damage > 0. and TYPE_CHART_MULTIPLIER[move.type][opp_pkm.type] > 0.:
            damage = fixed_damage
        else:
            damage = estimate_damage(move.type, my_pkm.type, move.power, opp_pkm.type, my_stages[PkmStat.ATTACK], opp_stages[PkmStat.DEFENSE], weather_condition)

        opp_pkm.hp = max(0, opp_pkm.hp - damage)
        my_pkm.hp = min(my_pkm.max_hp, my_pkm.hp + move.recover)


    def will_play_next_turn(self, team: PkmTeam) -> bool:
        """
            return true only if we are sure that our player WILL play the next turn (he doesn't have any status problem)
        """
        return team.active.status in self.worst_case_status or ((team.confused or team.active.asleep()) and will_free_next_turn(team))
    
def evaluate(node: AlphaBetaNode)-> int:
    parent = node.parent
    difference_my_hp = get_difference_old_new_hp(node.my_pkm, node.parent.my_pkm, node.parent.my_party)
    difference_opp_hp = get_difference_old_new_hp(node.opp_pkm, node.parent.opp_pkm, node.parent.opp_party)
    opp_modifiers = get_stats_modifiers(node.opp_pkm) + get_status_modifiers(node.opp_pkm) + get_faint_modifiers(node.my_pkm)
    my_modifiers = get_stats_modifiers(node.my_pkm) + get_status_modifiers(node.my_pkm) + get_faint_modifiers(node.my_pkm)
    type_modifiers = get_type_advantage_modifiers(node.my_pkm, node.opp_pkm)
    val = 1.25 * difference_opp_hp - difference_my_hp + opp_modifiers - my_modifiers + type_modifiers
    print(f'node value: {val}')
    print(f'my pokemon: {node.my_pkm}, \nopponnents pkm: {node.opp_pkm}')

    print(f'with values :    \n  - difference_my_hp = {difference_my_hp} \
                            \n  - difference_opp_hp = {difference_opp_hp} \
                            \n  - opp_modifiers = {opp_modifiers} \
                            \n  - my_modifiers = {my_modifiers} \
                            \n  - type_modifiers = {type_modifiers}')
    return val

def get_stats_modifiers(pkm: Pkm) -> int:
    return 0

def get_status_modifiers(pkm: Pkm) -> int:
    return 0

def get_faint_modifiers(pkm: Pkm) -> int:
    return 0

def get_difference_old_new_hp(active: Pkm, old_active: Pkm, old_party: List[Pkm]) -> int:
    if not old_active == active:
        for pkm in old_party:
            if pkm == active:
                return pkm.hp - active.hp
    else:
        return old_active.hp - active.hp

def get_type_advantage_modifiers(pkm: Pkm, opp_Pkm: Pkm):
    return 10 * TYPE_CHART_MULTIPLIER[pkm.type][opp_Pkm.type] - 20 * TYPE_CHART_MULTIPLIER[opp_Pkm.type][pkm.type]

class AlphaBetaNode():
    """
        Class which stores the result of the PART of a turn, meaning, one move made by one player, kind of represents a 'gamestate' in itself
    """


    def __init__(self, my_party: PkmTeam, opp_party: PkmTeam, weather:WeatherCondition, parent: AlphaBetaNode, player:int, depth: int, turn_end: bool, action: int):


        
        #the action chosen
        self.action = action

        #the player doing the move
        self.player = player


        #true if the node represents the end of a turn
        self.turn_end = turn_end

        self.my_pkm = my_party.active
        self.my_stages = my_party.stage
        self.my_party = my_party.party

        self.opp_pkm = opp_party.active
        self.opp_stages = opp_party.stage
        self.opp_party = opp_party.party

        #is one of the pokemon fainted ?
        self.is_terminal = self.my_pkm.hp == 0 or self.opp_pkm.hp == 0
        


        #all possible actions of every player starting from this game state
        self.possible_action = self.find_possible_actions_stripped(my_party.active.moves, opp_party.active.moves, my_party.active.type, opp_party.active.type, my_party.stage, opp_party.stage, weather, my_party.party, opp_party.party )

        #the current depth from the origin node
        self.depth = depth

        #the parent node
        self.parent = parent

        #the children nodes
        self.children = []

        print('new node created :')
        print(f'  -player : {player} \n  -action : {action}')
        print(f'  -possible actions : {self.possible_action}')

        #the value of the play, calculated by the eval function
        self.value = 0
        if parent != None:
            self.value = evaluate(self)

        


    

    def add_child(self, child: AlphaBetaNode):
        self.children.append(child)

    def get_next_attacking_player(self) -> int:
        """
            get who the next player to attack will be, does NOT retrieve who is the next player to PLAY, because of priorities in some actions (i.e switching)
        """
        if self.turn_end:
            return 0 if self.my_stages[PkmStat.SPEED] > self.opp_stages[PkmStat.SPEED] else 1
        else:
            return (self.player + 1) % 2

    def find_best_child_value(self) -> int:
        best_node = None
        for i in self.children:
            pass

    def not_fainted_pkm_actions(self, player: int, party: List[Pkm]) -> List[Tuple(int, int)]:
        """
            retrieve the list of pokemon a player can switch to
        """
        not_fainted = []
        for i, pkm in enumerate(party):
            if not pkm.fainted():
                not_fainted.append((player, i + 4))
        return not_fainted
    

    def find_possible_actions_stripped(self, my_moves: List[PkmMove], opp_moves: List[PkmMove], pkm_type: PkmType, opp_pkm_type: PkmType, my_stage:List[int], opp_stage: List[int], weather: WeatherCondition, my_party: List[Pkm], opp_party: List[Pkm]) -> List[Tuple(int, int)]:
        """
            Find the next possible moves for every player.

            If it is the middle of a turn, then only one player may be able to do a move, but if it is the end of a turn, then both players may make a move.

            This means that the tree depends on the move done to know which player will play first

            the actions are stored in a list of tuple, which has the form of (player of the move, move)
        """
        actions = []
        next_attacking_player = self.get_next_attacking_player()
        print(f'next attack player: {next_attacking_player}')
        if next_attacking_player == 0:
            actions.append((0,find_best_damaging_move(my_moves, pkm_type, opp_pkm_type, my_stage[PkmStat.ATTACK], opp_stage[PkmStat.DEFENSE], weather)))
            actions = actions + self.not_fainted_pkm_actions(0, my_party)
        else:
            actions.append((1, find_best_damaging_move(opp_moves, opp_pkm_type, pkm_type, opp_stage[PkmStat.ATTACK], my_stage[PkmStat.DEFENSE], weather)))
            actions = actions + self.not_fainted_pkm_actions(1, opp_party)

        if self.turn_end:
            if next_attacking_player == 0:
                actions = actions + self.not_fainted_pkm_actions(1, opp_party)
            else:
                actions = actions + self.not_fainted_pkm_actions(0, my_party)
        return actions



def will_take_weather_damage(pkm: Pkm, weather: WeatherCondition):
    return (weather == WeatherCondition.SANDSTORM and (pkm.type != PkmType.ROCK and pkm.type != PkmType.GROUND and pkm.type != PkmType.STEEL)) or (weather == WeatherCondition.HAIL and (pkm.type != PkmType.ICE))


def deal_hazard_damage(team: PkmTeam, pkm: Pkm):
    spikes = team.entry_hazard[PkmEntryHazard.SPIKES]
    if spikes and pkm.type != PkmType.FLYING:
        pkm.hp -= STATE_DAMAGE if spikes <= 1 else SPIKES_2 if spikes == 2 else SPIKES_3
        pkm.hp = 0. if pkm.hp < 0. else pkm.hp


def will_free_next_turn(team: PkmTeam):
    return (team.n_turns_confused == 3 or not team.confused) and (not team.active.asleep() or team.active.n_turns_asleep == 3 )


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

def find_best_damaging_move(moves: List[PkmMove], pkm_type: PkmType, opp_pkm_type: PkmType, attack_stage: int, defense_stage: int, weather: WeatherCondition) -> int:
    best_damage = 0
    best_move = None

    for move in moves:
        damage = estimate_damage(move.type, pkm_type, move.power, opp_pkm_type, attack_stage, defense_stage, weather)
        
        if damage >= best_damage:
            best_damage = damage
            best_move = move
    return moves.index(best_move)