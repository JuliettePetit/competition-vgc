# Pokémon vgc, cog competition 2023
## Description

We aim to complete battle track and championship track.
This git will be used to develop an agent in the context of an internship in Japan.
The participants are Mathis Foussac and Juliette Petit

## Rules:

### General gameplay rules:

### Competition rules:
- Pokémon only have one type
- Official type chart is used (see below)
- There are only 3 stats : ATTACK, DEFENSE, SPEED
- Stats have stages affecting them (stages can range from -5 to +5)
- Same Type Attack Bonus (STAB) applies： if a move is of the type of the Pokémon using it, it has a +50% bonus
/////////////- There are team battles (3 Pokémon) and solo battles (3 ~ 6 Pokémon)
- Weathers and their effects apply, possible weathers are : 
    - clear: no effect,
    - sunny: fire moves strengthened, water moves weakened,
    - rain: fire moves weakened, water moves strengthened,
    - sandstorm: damage to Pokémon that are not steel/ground/rock type, rock types have a defense boost,
    - hail: damage to Pokémon that are ice type, 
effects are taken from the official Pokémon page as they are not specified here.
- Pokémon teams are not permitted to have duplicate specimens.
- Individual Pokémon must have points distributed along stat slots so that the total sum of move powers and effects does not exceed a predetermined maximum.
    - The type of the Pokémon and the type of each of its moves are not considered for the total points.
    - Each special ability is worth one point.
    - Each 30 HP/30 power is worth one point.
    - Each Pokémon can be given a maximum of 11 points.
    - Each Pokémon will have 120 HP, 30 Power, 30 Power, and 30 Power as base stats that do not count toward point distribution.
    - Each Pokémon maximum HP is truncated to 240: we assume that you only lose your points if you spend them on a pokemon that has 240hp.
- Each move can only have a maximum of one effect: each abilities (move) can do only one effect in addition to damage.

*Pokémon official chart* 
![typechart](/competition-vgc/typechart.png)
(Source: https://pt.m.wikipedia.org/wiki/Ficheiro:Pokemon_Type_Chart.svg)

### Battle track specific rules:
### Championship track specific rules:

- a roster of 51 Pokémon will be used.


## Registration:

We must only make a single submission even when entering multiple tracks. If multiple submissions by the same contestant are made, only the earliest will be considered.
Failing to indicate at least one track for participation will make the submission be disregarded.
Submissions after 30th of June (UTC Summer Time) will be disregarded.

We need to specify 

- Name(s)
- Email(s)
- Affiliations
- Which tracks we wish to participate in. (one or more)
- Battle Track
- Championship Track
- Meta-Game Balance Track
- Agent Name String (UTF-8) 
- Submission File (compressed into a single .zip or .tar.gz file)
- If we consent that our code is publicly archived and shared in the VGC AI Framework.

[VGC AI Competition 1st Edition registration link (2023)](https://forms.gle/buvmMjCMfqzGnNtm9)

