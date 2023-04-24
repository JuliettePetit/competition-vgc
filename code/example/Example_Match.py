from agent.Example_Competitor import ExampleCompetitor
from vgc.behaviour.BattlePolicies import Minimax
from vgc.competition.BattleMatch import BattleMatch
from vgc.competition.Competitor import CompetitorManager
from vgc.util.generator.PkmRosterGenerators import RandomPkmRosterGenerator
from vgc.util.generator.PkmTeamGenerators import RandomTeamFromRoster


def main():
    roster = RandomPkmRosterGenerator().gen_roster()
    tg = RandomTeamFromRoster(roster)
    c0 = ExampleCompetitor("Player 1")
    c0._battle_policy = Minimax()  # switch agent to test
    cm0 = CompetitorManager(c0)
    cm0.team = tg.get_team()
    c1 = ExampleCompetitor("Player 2")
    cm1 = CompetitorManager(c1)
    cm1.team = tg.get_team()
    match = BattleMatch(cm0, cm1, debug=True)
    match.run()


if __name__ == '__main__':
    main()
