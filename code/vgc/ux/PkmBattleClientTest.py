from multiprocessing.connection import Client

from vgc.engine.PkmBattleEnv import PkmBattleEnv

from vgc.behaviour.BattlePolicies import FirstPlayer
from vgc.behaviour.BattlePolicies import Minimax
from vgc.behaviour.BattlePolicies import MonteCarloPlayer
from vgc.util.generator.PkmTeamGenerators import RandomTeamGenerator

a0 = MonteCarloPlayer()
a1 = FirstPlayer()
address = ('localhost', 8000)
gen = RandomTeamGenerator()
full_team0 = gen.get_team()
full_team1 = gen.get_team()
conn = Client(address, authkey='VGC AI'.encode('utf-8'))
env = PkmBattleEnv((full_team0.get_battle_team([0, 1, 2]), full_team1.get_battle_team([0, 1, 2])), debug=True,
                   conn=conn, encode=(a0.requires_encode(), a1.requires_encode()))
env.reset()
t = False
ep = 0
n_battles = 3
while ep < n_battles:
    s = env.reset()
    env.render(mode='ux')
    ep += 1
    while not t:
        a = [a0.get_action(s[0]), a1.get_action(s[1])]
        s, _, t, v = env.step(a)

        env.render(mode='ux')
    t = False
env.close()
