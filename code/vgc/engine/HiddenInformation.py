from typing import Optional

from vgc.datatypes.Objects import Pkm, PkmMove, PkmFullTeam

null_pkm_move = PkmMove()
null_pkm = Pkm()


def set_moves(pkm: Pkm, prediction: Optional[Pkm]):
    for i in range(len(pkm.moves)):
        if not pkm.moves[i].revealed:
            if prediction is not None:
                pkm.moves[i] = prediction.moves[i]
            else:
                pkm.moves[i] = null_pkm.moves[i]


def set_pkm(pkm: Pkm, prediction: Optional[Pkm]):
    if pkm.revealed:
        set_moves(pkm, prediction)
    elif prediction is not None:
        pkm.type = prediction.type
        pkm.hp = prediction.hp
        set_moves(pkm, prediction)
    else:
        pkm.type = null_pkm.type
        pkm.hp = null_pkm.hp
        for move in pkm.moves:
            move.hide()
        set_moves(pkm, null_pkm)


def hide_pkm(pkm: Pkm):
    set_pkm(pkm, None)


def hide_team(team: PkmFullTeam):
    for pkm in team.pkm_list:
        hide_pkm(pkm)
