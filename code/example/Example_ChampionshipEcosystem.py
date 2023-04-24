from agent.Example_Competitor import ExampleCompetitor
from vgc.balance.meta import StandardMetaData
from vgc.competition.Competitor import CompetitorManager
from vgc.ecosystem.ChampionshipEcosystem import ChampionshipEcosystem
from vgc.util.generator.PkmRosterGenerators import RandomPkmRosterGenerator

N_PLAYERS = 16


def main():
    generator = RandomPkmRosterGenerator()
    roster = generator.gen_roster()
    move_roster = generator.base_move_roster
    meta_data = StandardMetaData()
    meta_data.set_moves_and_pkm(roster, move_roster)
    ce = ChampionshipEcosystem(roster, meta_data, debug=True)
    for i in range(N_PLAYERS):
        cm = CompetitorManager(ExampleCompetitor("Player %d" % i))
        ce.register(cm)
    ce.run(n_epochs=10, n_league_epochs=10)


if __name__ == '__main__':
    main()
