import time
from vgc.balance.meta import MetaData
from vgc.behaviour import TeamBuildPolicy
from vgc.behaviour.BattlePolicies import Minimax, TypeSelector
from vgc.datatypes.Constants import TYPE_CHART_MULTIPLIER
from vgc.datatypes.Objects import Pkm, PkmFullTeam, PkmRoster, PkmTeam
from typing import Dict, Tuple, List
from vgc.datatypes.Types import PkmType

from vgc.engine.PkmBattleEnv import PkmBattleEnv


class FlagShipComplementarity(TeamBuildPolicy):
    """
        The idea of this team builder is simple : 
            for the set roster phase :
                First, we rank every pokemon in the roster by making them battle agains each other.
                we take that rank and only keep the x first pokemon to trim the roster.
                we then fill the final roster with pokemon whose type doesn't appear in the roster for now.

                we then try to find which pokemon have the best sinergies in our reduced roster and 

            for the team build part

        team builder which centers the build around a main pokemon (the flagship pokemon). 
    """

    def __init__(self, roster_size: int = 20, nb_matchup: int = 3):
        #lis of score where the placement is the id of the pkm
        self.pkm_scores: List[int]

        #list of pkm ids sorted by score (descending)
        self.pkm_ranking: List[int] = []

        self.pkm_complementarity: Dict[Pkm, List[Pkm]] = dict()

        self.matchup_agent = TypeSelector()
        self.nb_matchup = nb_matchup

        self.roster_size = roster_size

        #final roster after trimming
        self.usable_roster : List[Pkm] = []


    def requires_encode(self) -> bool:
        return False

    def close(self):
        pass

    def set_roster(self, roster: PkmRoster):
        self.pkm_scores = [0] * len(roster)
        pkms = []
        for pt in roster:
            pkms.append(pt.gen_pkm([0, 1, 2, 3]))
        for i, pkm0 in enumerate (pkms):
            for j, pkm1 in enumerate (pkms[i:]):
                if j != 0:
                    self.score_matchup(i, pkm0, j + i, pkm1)
            self.insert_pkm_ranking(i)
        self.trim_and_set_roster(pkms)



    def find_sinergies_in_usable_roster(self):
        for pkm in self.usable_roster:
            self.pkm_complementarity[pkm] += self.find_pkm_type_sinergies(pkm)
            self.pkm_complementarity[pkm] += self.find_pkm_move_sinergies(pkm)
    
    def find_pkm_type_sinergies(self, pkm: Pkm):
        comp = []
        for other in self.usable_roster:
            if is_complementary(pkm.type, other.type):
                comp.append(other)
                
        return [comp]

    def find_pkm_move_sinergies(self, pkm: Pkm):
        for pkm in self.usable_roster:      #potential expansion : we consider moves instead of simply type
            pass
        return []

    def trim_and_set_roster(self, roster: List[Pkm]):
        pkms = []
        viewed_type = []
        final_ranking = self.pkm_ranking[:self.roster_size]
        final_score = self.pkm_scores[:self.roster_size]

        for i in final_ranking:
            pkms.append(roster[i])
            if not roster[i].type in viewed_type:
                viewed_type.append(roster[i].type)

        fill_in_roster = self.fill_missing_pkm(roster, viewed_type)
        
        while len(fill_in_roster) > 0:
            final_ranking.append(self.pkm_ranking[fill_in_roster[0]])
            final_score.append(self.pkm_scores[fill_in_roster[0]])
            pkms.append(roster[fill_in_roster[0]])
            fill_in_roster.pop(0)
       

        self.usable_roster = pkms 
        self.roster_size = len(pkms)
        self.pkm_scores = final_score
        self.pkm_ranking = final_ranking

        for pkm in self.usable_roster:
            pkm.reset()
            self.pkm_complementarity[pkm] = []



    

    def fill_missing_pkm(self, roster: List[Pkm], viewed_type: List[PkmType]) -> List[Pkm]:
        fill_in_roster = []
        if len(viewed_type) < len(PkmType):
            fill_in_roster = self.fill_type(roster, viewed_type)
        return fill_in_roster
            


    def fill_type(self, roster: List[Pkm], viewed_type: List[PkmType]) -> List[Pkm]:
        pkms = []
        for i in self.pkm_ranking[self.roster_size + 1:]:
            if not roster[i].type in viewed_type:
                viewed_type.append(roster[i].type)
                pkms.append(i)
                if len(viewed_type) >= len(PkmType):
                    break
        return pkms

            
    def insert_pkm_ranking(self, pkm_id: int):
        inserted = False
        i = 0
        score = self.pkm_scores[pkm_id]
        lgth = len(self.pkm_ranking)
        while i < lgth and not inserted:
            if self.pkm_scores[self.pkm_ranking[i]] < score:
                self.pkm_ranking.insert(i, pkm_id)
                inserted = True
            i += 1
        if not inserted:
            self.pkm_ranking.append(pkm_id)

    def score_matchup(self, pkm0_id: int, pkm0: Pkm, pkm1_id: int, pkm1: Pkm):
        t0 = PkmTeam([pkm0])
        t1 = PkmTeam([pkm1])
        final_scores = [0,0]
        env= PkmBattleEnv((t0, t1), encode=(False, False))
        for _ in range(self.nb_matchup):
            s = env.reset()
            t = False
            while not t:
                a0 = self.matchup_agent.get_action(s[0])
                a1 = self.matchup_agent.get_action(s[1])
                s, r, t, _ = env.step([a0, a1])
                final_scores[0] += r[0]
                final_scores[1] += r[1]
        self.pkm_scores[pkm0_id] += final_scores[0]
        self.pkm_scores[pkm1_id] += final_scores[1]


    def get_flagship(self):
        return self.usable_roster[0]
    
    def get_complementary(self, pkm: Pkm):
        if len(self.pkm_complementarity[pkm]) > 1:
            return self.pkm_complementarity[pkm][:2]
        else:
            return self.usable_roster[1:3]

    def get_action(self, meta: MetaData) -> PkmFullTeam: 
        flagship = self.get_flagship()
        team = self.get_complementary(flagship)
        team.insert(0, flagship)
        return PkmFullTeam(team)
    
def is_complementary(pkm_0_type: PkmType, pkm_1_type: PkmType):
    res = 0
    for i in TYPE_CHART_MULTIPLIER:
        res += abs(TYPE_CHART_MULTIPLIER[pkm_0_type][i] - TYPE_CHART_MULTIPLIER[pkm_1_type][i])
    return True if res >= 3 else False