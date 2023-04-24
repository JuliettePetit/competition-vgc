import unittest
from copy import deepcopy

from vgc.competition.StandardPkmMoves import Psychic, HydroPump, Thunder, FireBlast
from vgc.datatypes.Objects import Pkm
from vgc.datatypes.Types import PkmType
from vgc.engine.HiddenInformation import set_pkm, hide_pkm


class TestHiddenInformation(unittest.TestCase):

    def test_set_pkm_1(self):
        pkm = Pkm()
        pkm_p = Pkm(PkmType.ICE, 100.0, move0=Psychic, move1=HydroPump)
        set_pkm(pkm, pkm_p)
        self.assertEqual(pkm.type, pkm_p.type)
        self.assertEqual(pkm.hp, pkm_p.hp)
        self.assertEqual(pkm.moves[0], pkm_p.moves[0])
        self.assertEqual(pkm.moves[1], pkm_p.moves[1])

    def test_set_pkm_2(self):
        pkm = Pkm()
        pkm_c = deepcopy(pkm)
        pkm.reveal_pkm()
        pkm.moves[0].reveal()
        pkm_p = Pkm(PkmType.ICE, 100.0, move0=Psychic, move1=HydroPump)
        set_pkm(pkm, pkm_p)
        self.assertNotEqual(pkm.type, pkm_p.type)
        self.assertNotEqual(pkm.hp, pkm_p.hp)
        self.assertNotEqual(pkm.moves[0].type, pkm_p.moves[0].type)
        self.assertEqual(pkm.moves[1].type, pkm_p.moves[1].type)
        self.assertEqual(pkm.type, pkm_c.type)
        self.assertEqual(pkm.hp, pkm_c.hp)
        self.assertEqual(pkm.moves[0].type, pkm_c.moves[0].type)
        self.assertNotEqual(pkm.moves[1].type, pkm_c.moves[1].type)

    def test_hide_pkm_1(self):
        pkm = Pkm(PkmType.ICE, 100.0, move0=Psychic, move1=HydroPump, move2=Thunder, move3=FireBlast)
        pkm_c = deepcopy(pkm)
        hide_pkm(pkm)
        self.assertNotEqual(pkm.type, pkm_c.type)
        self.assertNotEqual(pkm.hp, pkm_c.hp)
        self.assertNotEqual(pkm.moves[0].type, pkm_c.moves[0].type)
        self.assertNotEqual(pkm.moves[1].type, pkm_c.moves[1].type)
        self.assertNotEqual(pkm.moves[2].type, pkm_c.moves[2].type)
        self.assertNotEqual(pkm.moves[3].type, pkm_c.moves[3].type)

    def test_hide_pkm_2(self):
        pkm = Pkm(PkmType.ICE, 100.0, move0=Psychic, move1=HydroPump, move2=Thunder, move3=FireBlast)
        pkm_c = deepcopy(pkm)
        pkm.reveal_pkm()
        hide_pkm(pkm)
        self.assertEqual(pkm.type, pkm_c.type)
        self.assertEqual(pkm.hp, pkm_c.hp)
        self.assertNotEqual(pkm.moves[0].type, pkm_c.moves[0].type)
        self.assertNotEqual(pkm.moves[1].type, pkm_c.moves[1].type)
        self.assertNotEqual(pkm.moves[2].type, pkm_c.moves[2].type)
        self.assertNotEqual(pkm.moves[3].type, pkm_c.moves[3].type)

    def test_hide_pkm_3(self):
        pkm = Pkm(PkmType.ICE, 100.0, move0=Psychic, move1=HydroPump, move2=Thunder, move3=FireBlast)
        pkm_c = deepcopy(pkm)
        pkm.moves[0].reveal()
        pkm.moves[2].reveal()
        hide_pkm(pkm)
        self.assertFalse(pkm.revealed)
        self.assertNotEqual(pkm.type, pkm_c.type)
        self.assertNotEqual(pkm.hp, pkm_c.hp)
        self.assertNotEqual(pkm.moves[0].type, pkm_c.moves[0].type)
        self.assertNotEqual(pkm.moves[1].type, pkm_c.moves[1].type)
        self.assertNotEqual(pkm.moves[2].type, pkm_c.moves[2].type)
        self.assertNotEqual(pkm.moves[3].type, pkm_c.moves[3].type)

    def test_hide_pkm_4(self):
        pkm = Pkm(PkmType.ICE, 100.0, move0=Psychic, move1=HydroPump, move2=Thunder, move3=FireBlast)
        pkm_c = deepcopy(pkm)
        pkm.reveal_pkm()
        pkm.moves[0].reveal()
        pkm.moves[2].reveal()
        hide_pkm(pkm)
        self.assertTrue(pkm.revealed)
        self.assertEqual(pkm.type, pkm_c.type)
        self.assertEqual(pkm.hp, pkm_c.hp)
        self.assertEqual(pkm.moves[0].type, pkm_c.moves[0].type)
        self.assertNotEqual(pkm.moves[1].type, pkm_c.moves[1].type)
        self.assertEqual(pkm.moves[2].type, pkm_c.moves[2].type)
        self.assertNotEqual(pkm.moves[3].type, pkm_c.moves[3].type)


if __name__ == '__main__':
    unittest.main()
