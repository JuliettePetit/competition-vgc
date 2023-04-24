import unittest

from vgc.datatypes.Objects import PkmFullTeam
from vgc.datatypes.Types import PkmType
from vgc.engine.HiddenInformation import null_pkm
from vgc.engine.PkmBattleEnv import PkmBattleEnv
from vgc.util.generator.PkmRosterGenerators import RandomPkmRosterGenerator


class TestForwardModel(unittest.TestCase):
    team0 = None
    team1 = None

    @classmethod
    def setUpClass(cls):
        generator = RandomPkmRosterGenerator()
        roster = generator.gen_roster()
        pkms = [roster[i].gen_pkm([0, 1, 2, 3]) for i in range(6)]
        cls.team0 = PkmFullTeam(pkms[0:3]).get_battle_team([0, 1, 2])
        cls.team1 = PkmFullTeam(pkms[3:6]).get_battle_team([0, 1, 2])

    def test_state_type(self):
        env = PkmBattleEnv((self.team0, self.team1), encode=(False, True))
        s0, s1 = env.reset()
        self.assertIs(type(s0), PkmBattleEnv)
        self.assertIsNot(type(s1), PkmBattleEnv)

    def test_forward_model(self):
        env = PkmBattleEnv((self.team0, self.team1), encode=(False, False))
        s0, s1 = env.reset()

        my_team_0 = s0.teams[0]
        opp_team_0 = s0.teams[1]
        my_active_0 = my_team_0.active
        opp_active_0 = opp_team_0.active
        my_first_0 = my_team_0.party[0]
        opp_first_0 = opp_team_0.party[0]

        if my_active_0.type != PkmType.NORMAL:
            self.assertNotEqual(my_active_0.type, null_pkm.type)
        if my_active_0.hp != 240.0:
            self.assertNotEqual(my_active_0.hp, null_pkm.hp)
        self.assertTrue(my_active_0.public)

        if opp_active_0.type != PkmType.NORMAL:
            self.assertNotEqual(opp_active_0.type, null_pkm.type)
        if opp_active_0.hp != 240.0:
            self.assertNotEqual(opp_active_0.hp, null_pkm.hp)
        self.assertTrue(opp_active_0.public)

        if my_first_0.type != PkmType.NORMAL:
            self.assertNotEqual(my_first_0.type, null_pkm.type)
        if my_first_0.hp != 240.0:
            self.assertNotEqual(my_first_0.hp, null_pkm.hp)
        self.assertFalse(my_first_0.public)

        if opp_first_0.type != PkmType.NORMAL:
            self.assertEqual(opp_first_0.type, null_pkm.type)
        if opp_first_0.hp != 240.0:
            self.assertEqual(opp_first_0.hp, null_pkm.hp)
        self.assertFalse(opp_first_0.public)

        my_team_1 = s1.teams[0]
        opp_team_1 = s1.teams[1]
        my_active_1 = my_team_1.active
        opp_active_1 = opp_team_1.active
        my_first_1 = my_team_1.party[0]
        opp_first_1 = opp_team_1.party[0]

        if my_active_1.type != PkmType.NORMAL:
            self.assertNotEqual(my_active_1.type, null_pkm.type)
        if my_active_1.hp != 240.0:
            self.assertNotEqual(my_active_1.hp, null_pkm.hp)
        self.assertTrue(my_active_1.public)

        if opp_active_1.type != PkmType.NORMAL:
            self.assertNotEqual(opp_active_1.type, null_pkm.type)
        if opp_active_1.hp != 240.0:
            self.assertNotEqual(opp_active_1.hp, null_pkm.hp)
        self.assertTrue(opp_active_1.public)

        if my_first_1.type != PkmType.NORMAL:
            self.assertNotEqual(my_first_1.type, null_pkm.type)
        if my_first_1.hp != 240.0:
            self.assertNotEqual(my_first_1.hp, null_pkm.hp)
        self.assertFalse(my_first_1.public)

        if opp_first_1.type != PkmType.NORMAL:
            self.assertEqual(opp_first_1.type, null_pkm.type)
        if opp_first_1.hp != 240.0:
            self.assertEqual(opp_first_1.hp, null_pkm.hp)
        self.assertFalse(opp_first_1.public)

        s, _, _, _ = env.step([4, 4])  # both players switch
        s0, s1 = s

        my_team_0 = s0.teams[0]
        opp_team_0 = s0.teams[1]
        my_active_0 = my_team_0.active
        opp_active_0 = opp_team_0.active
        my_first_0 = my_team_0.party[0]
        opp_first_0 = opp_team_0.party[0]

        if my_active_0.type != PkmType.NORMAL:
            self.assertNotEqual(my_active_0.type, null_pkm.type)
        if my_active_0.hp != 240.0:
            self.assertNotEqual(my_active_0.hp, null_pkm.hp)
        self.assertTrue(my_active_0.public)

        if opp_active_0.type != PkmType.NORMAL:
            self.assertNotEqual(opp_active_0.type, null_pkm.type)
        if opp_active_0.hp != 240.0:
            self.assertNotEqual(opp_active_0.hp, null_pkm.hp)
        self.assertTrue(opp_active_0.public)

        if my_first_0.type != PkmType.NORMAL:
            self.assertNotEqual(my_first_0.type, null_pkm.type)
        if my_first_0.hp != 240.0:
            self.assertNotEqual(my_first_0.hp, null_pkm.hp)
        self.assertTrue(my_first_0.public)

        if opp_first_0.type != PkmType.NORMAL:
            self.assertNotEqual(opp_first_0.type, null_pkm.type)
        if opp_first_0.hp != 240.0:
            self.assertNotEqual(opp_first_0.hp, null_pkm.hp)
        self.assertTrue(opp_first_0.public)

        my_team_1 = s1.teams[0]
        opp_team_1 = s1.teams[1]
        my_active_1 = my_team_1.active
        opp_active_1 = opp_team_1.active
        my_first_1 = my_team_1.party[0]
        opp_first_1 = opp_team_1.party[0]

        if my_active_1.type != PkmType.NORMAL:
            self.assertNotEqual(my_active_1.type, null_pkm.type)
        if my_active_1.hp != 240.0:
            self.assertNotEqual(my_active_1.hp, null_pkm.hp)
        self.assertTrue(my_active_1.public)

        if opp_active_1.type != PkmType.NORMAL:
            self.assertNotEqual(opp_active_1.type, null_pkm.type)
        if opp_active_1.hp != 240.0:
            self.assertNotEqual(opp_active_1.hp, null_pkm.hp)
        self.assertTrue(opp_active_1.public)

        if my_first_1.type != PkmType.NORMAL:
            self.assertNotEqual(my_first_1.type, null_pkm.type)
        if my_first_1.hp != 240.0:
            self.assertNotEqual(my_first_1.hp, null_pkm.hp)
        self.assertTrue(my_first_1.public)

        if opp_first_1.type != PkmType.NORMAL:
            self.assertNotEqual(opp_first_1.type, null_pkm.type)
        if opp_first_1.hp != 240.0:
            self.assertNotEqual(opp_first_1.hp, null_pkm.hp)
        self.assertTrue(opp_first_1.public)

        my_move_0 = my_active_0.moves[0]
        opp_move_0 = opp_active_0.moves[0]

        if my_move_0.type != PkmType.NORMAL:
            self.assertNotEqual(my_move_0.type, null_pkm.moves[0].type)
        if my_move_0.power != 240.0:
            self.assertNotEqual(my_move_0.power, null_pkm.moves[0].power)
        self.assertFalse(my_move_0.public)

        if opp_move_0.type != PkmType.NORMAL:
            self.assertEqual(opp_move_0.type, null_pkm.moves[0].type)
        if opp_move_0.power != 240.0:
            self.assertEqual(opp_move_0.power, null_pkm.moves[0].power)
        self.assertFalse(opp_move_0.public)

        my_move_1 = my_active_1.moves[0]
        opp_move_1 = opp_active_1.moves[0]

        if my_move_1.type != PkmType.NORMAL:
            self.assertNotEqual(my_move_1.type, null_pkm.moves[0].type)
        if my_move_1.power != 240.0:
            self.assertNotEqual(my_move_1.power, null_pkm.moves[0].power)
        self.assertFalse(my_move_1.public)

        if opp_move_1.type != PkmType.NORMAL:
            self.assertEqual(opp_move_1.type, null_pkm.moves[0].type)
        if opp_move_1.power != 240.0:
            self.assertEqual(opp_move_1.power, null_pkm.moves[0].power)
        self.assertFalse(opp_move_1.public)

        s, _, _, _ = env.step([0, 0])  # both players use first move
        s0, s1 = s

        my_team_0 = s0.teams[0]
        opp_team_0 = s0.teams[1]
        my_active_0 = my_team_0.active
        opp_active_0 = opp_team_0.active

        my_move_0 = my_active_0.moves[0]
        opp_move_0 = opp_active_0.moves[0]

        if my_move_0.type != PkmType.NORMAL:
            self.assertNotEqual(my_move_0.type, null_pkm.moves[0].type)
        if my_move_0.power != 240.0:
            self.assertNotEqual(my_move_0.power, null_pkm.moves[0].power)
        self.assertTrue(my_move_0.public)

        if opp_move_0.type != PkmType.NORMAL:
            self.assertNotEqual(opp_move_0.type, null_pkm.moves[0].type)
        if opp_move_0.power != 240.0:  # TODO
            self.assertNotEqual(opp_move_0.power, null_pkm.moves[0].power)
        self.assertTrue(opp_move_0.public)

        my_team_1 = s1.teams[0]
        opp_team_1 = s1.teams[1]
        my_active_1 = my_team_1.active
        opp_active_1 = opp_team_1.active

        my_move_1 = my_active_1.moves[0]
        opp_move_1 = opp_active_1.moves[0]

        if my_move_1.type != PkmType.NORMAL:
            self.assertNotEqual(my_move_1.type, null_pkm.moves[0].type)
        if my_move_1.power != 240.0:
            self.assertNotEqual(my_move_1.power, null_pkm.moves[0].power)
        self.assertTrue(my_move_1.public)

        if opp_move_1.type != PkmType.NORMAL:
            self.assertNotEqual(opp_move_1.type, null_pkm.moves[0].type)
        if opp_move_1.power != 240.0:
            self.assertNotEqual(opp_move_1.power, null_pkm.moves[0].power)
        self.assertTrue(opp_move_1.public)


if __name__ == '__main__':
    unittest.main()
