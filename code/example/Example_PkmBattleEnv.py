from vgc.behaviour.BattlePolicies import RandomPlayer
from vgc.datatypes.Objects import PkmTeam
from vgc.engine.PkmBattleEnv import PkmBattleEnv


def main():
    a0 = RandomPlayer()
    a1 = RandomPlayer()
    t0 = PkmTeam()
    t1 = PkmTeam()
    env = PkmBattleEnv((t0, t1), debug=True, encode=(a0.requires_encode(), a1.requires_encode()))
    env.reset()
    t = False
    ep = 0
    n_battles = 3
    while ep < n_battles:
        s = env.reset()
        env.render()
        ep += 1
        while not t:
            o0 = s[0]
            o1 = s[1]
            a = [a0.get_action(o0), a1.get_action(o1)]
            s, _, t, v = env.step(a)
            env.render()
        t = False
    a0.close()


if __name__ == '__main__':
    main()
