import unittest

from vgc.balance.restriction import VGCDesignConstraints, RosterBoundedSizeRule, RosterFixedSizeRule, \
    UnbannableRule, MoveRosterBoundedSizeRule, MoveRosterFixedSizeRule, MovesUnchangeableRule, TypeUnchangeableRule, \
    MaxHPUnchangeableRule
from vgc.datatypes.Types import PkmType
from vgc.util.generator.PkmRosterGenerators import RandomPkmRosterGenerator


class TestVGCDesignConstraints(unittest.TestCase):

    def test_check_every_rule(self):
        gen = RandomPkmRosterGenerator()
        roster = gen.gen_roster()
        constraints = VGCDesignConstraints(roster)
        constraints.add_global_rule(RosterBoundedSizeRule(roster, max_size=len(roster)))
        constraints.add_global_rule(RosterFixedSizeRule(roster))
        constraints.add_global_rule(UnbannableRule(roster, roster[0]))
        constraints.add_pkm_rule(roster[1], MoveRosterBoundedSizeRule(roster))
        constraints.add_pkm_rule(roster[2], MoveRosterFixedSizeRule(roster, roster[2].moves))
        constraints.add_pkm_rule(roster[3], MovesUnchangeableRule(roster, roster[3].moves))
        constraints.add_pkm_rule(roster[4], TypeUnchangeableRule(roster, roster[4].type))
        constraints.add_pkm_rule(roster[5], MaxHPUnchangeableRule(roster, roster[5].max_hp))
        failed = constraints.check_every_rule(roster)
        self.assertEqual(len(failed), 0)
        roster[5].max_hp -= 1.0
        failed = constraints.check_every_rule(roster)
        self.assertEqual(len(failed), 1)
        roster[4].type = PkmType.NORMAL if roster[4].type != PkmType.NORMAL else PkmType.FIRE
        failed = constraints.check_every_rule(roster)
        self.assertEqual(len(failed), 2)
        roster[3].moves.pop()
        failed = constraints.check_every_rule(roster)
        self.assertEqual(len(failed), 3)
        roster[2].moves.pop()
        failed = constraints.check_every_rule(roster)
        self.assertEqual(len(failed), 4)
        roster.pop()
        failed = constraints.check_every_rule(roster)
        self.assertEqual(len(failed), 5)
