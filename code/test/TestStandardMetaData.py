import unittest
from copy import deepcopy

from vgc.balance import DeltaPkm, DeltaRoster
from vgc.balance.meta import StandardMetaData
from vgc.competition.StandardPkmMoves import STANDARD_MOVE_ROSTER
from vgc.datatypes.Objects import PkmFullTeam
from vgc.util.generator.PkmRosterGenerators import RandomPkmRosterGenerator


class TestStandardMetaData(unittest.TestCase):
    roster = None
    move_roster = None

    @classmethod
    def setUpClass(cls):
        generator = RandomPkmRosterGenerator()
        cls.roster = generator.gen_roster()
        cls.move_roster = generator.base_move_roster

    def setUp(self):
        self.meta_data = StandardMetaData()
        self.meta_data.set_moves_and_pkm(self.roster, self.move_roster)

    def test_init(self):
        self.assertEqual(self.meta_data.get_n_teams(), 0)
        self.assertEqual(len(self.meta_data._pkm_usage), len(self.roster))
        self.assertEqual(len(self.meta_data._move_usage), len(self.move_roster))
        self.assertEqual(len(self.meta_data._moves), len(self.move_roster))
        self.assertEqual(len(self.meta_data._pkm), len(self.roster))
        self.assertEqual(len(self.meta_data._d_move), len(self.move_roster) ** 2)
        self.assertEqual(len(self.meta_data._d_pkm), len(self.roster) ** 2)

    def test_update_with_team(self):
        pkms = [self.roster[i].gen_pkm([0, 1, 2, 3]) for i in range(4)]
        full_team = PkmFullTeam(pkms[0:3])
        self.meta_data.update_with_team(full_team)
        self.assertEqual(self.meta_data.get_n_teams(), 1)
        self.assertEqual(self.meta_data.get_global_pkm_usage(pkms[0].pkm_id), 1 / 3)
        self.assertEqual(self.meta_data.get_pair_usage((pkms[0].pkm_id, pkms[1].pkm_id)), 1)
        self.assertEqual(self.meta_data.get_pair_usage((pkms[0].pkm_id, pkms[3].pkm_id)), 0)
        full_team_2 = PkmFullTeam(pkms[1:4])
        self.meta_data.update_with_team(full_team_2)
        self.assertEqual(self.meta_data.get_n_teams(), 2)
        self.assertEqual(self.meta_data.get_global_pkm_usage(pkms[0].pkm_id), 1 / 6)
        self.assertEqual(self.meta_data.get_pair_usage((pkms[0].pkm_id, pkms[1].pkm_id)), 1 / 2)
        self.assertEqual(self.meta_data.get_pair_usage((pkms[1].pkm_id, pkms[2].pkm_id)), 1)

    def test_delta_roster_apply(self):
        pkm = self.roster[0]
        copy_pkm = deepcopy(pkm)
        self.assertTrue(pkm == copy_pkm)
        copy2_pkm = deepcopy(pkm)
        delta_pkm = DeltaPkm(pkm.max_hp - 20, pkm.type, {1: STANDARD_MOVE_ROSTER[10]})
        delta_pkm.apply(copy_pkm)
        self.assertNotEqual(pkm, copy_pkm)
        self.test_update_with_team()
        pkms = [self.roster[i].gen_pkm([0, 1, 2, 3]) for i in range(4)]
        full_team = PkmFullTeam(pkms[0:3])
        self.meta_data.update_with_team(full_team)
        full_team_2 = PkmFullTeam(pkms[1:4])
        self.meta_data.update_with_team(full_team_2)
        delta_roster = DeltaRoster({0: delta_pkm})
        self.assertEqual(pkm, copy2_pkm)
        delta_roster.apply(self.roster)
        pkm = self.roster[0]
        self.assertNotEqual(pkm.max_hp, copy2_pkm.max_hp)
        self.assertEqual(pkm.type, copy2_pkm.type)
        self.assertNotEqual(pkm.moves[1], copy2_pkm.moves[1])
        self.assertEqual(pkm.moves[0], copy2_pkm.moves[0])
        self.assertNotEqual(pkm, copy2_pkm)
