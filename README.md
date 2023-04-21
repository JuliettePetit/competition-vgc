# Pokémon vgc, cog competition 2023
## Description

We aim to complete battle track and championship track.
This git will be used to develop an agent in the context of an internship in Japan.
The participants are Mathis Foussac and Juliette Petit

## Rules:

### Competition rules:

An entrant must submit a Competitor module that implements the required behaviors for the specific track.

<table>
<thead>
<tr>
<th></th>
<th>Battle Track</th>
<th>Championship Track</th>
<th>Meta-Game Balance Track</th>
</tr>
</thead>
<tbody>
<tr>
<td>Team Predictor</td>
<td></td>
<td align="center">X</td>
<td align="center"></td>
</tr>
<tr>
<td>Team Selection Policy</td>
<td align="center"></td>
<td align="center">X</td>
<td align="center"></td>
</tr>
<tr>
<td>Battle Policy</td>
<td align="center">X</td>
<td align="center">X</td>
<td align="center"></td>
</tr>
<tr>
<td>Team Builder Policy</td>
<td align="center"></td>
<td align="center">X</td>
<td align="center"></td>
</tr>
<tr>
<td>Game Balance Policy</td>
<td align="center"></td>
<td align="center"></td>
<td align="center">X</td>
</tr>
</tbody>
</table>

- Submitted agents must be able to communicate with the VGC AI framework through its communication protocol, as they will be run in isolated processes. The source code provides examples of how to construct a remote agent. The name method of the submitted agent must return the competitor's registration name.
- There is a time limit for deciding a move action during a Pokémon battle (value still under consideration).
- There is a time limit for team building (value still under consideration).
- There is a primary and secondary memory limit for the agents (value still under consideration).



### General gameplay rules:

**Pokémon characteristics**
- Pokémon only have one type
- Official type chart is used (see below)
- There are only 3 stats : ATTACK, DEFENSE, SPEED
- Stats have stages affecting them (stages can range from -5 to +5)
- Individual Pokémon must have points distributed along stat slots so that the total sum of move powers and effects does not exceed a predetermined maximum.
    - The type of the Pokémon and the type of each of its moves are not considered for the total points.
    - Each special ability is worth one point.
    - Each 30 HP/30 power is worth one point.
    - Each Pokémon can be given a maximum of 11 points.
    - Each Pokémon will have 120 HP, 30 Power, 30 Power, and 30 Power as base stats that do not count toward point distribution.
    - Each Pokémon maximum HP is truncated to 240
- A Pokémon must be unique in his team

**Pokémon's attacks characteristics**
- Same Type Attack Bonus (STAB) applies： if a move is of the type of the Pokémon using it, it has a +50% bonus
- Weathers and their effects apply, possible weathers are : 
    - clear: no effect,
    - sunny: fire moves strengthened, water moves weakened,
    - rain: fire moves weakened, water moves strengthened,
    - sandstorm: damage to Pokémon that are not steel/ground/rock type, rock types have a defense boost,
    - hail: damage to Pokémon that are ice type. 
Effects are taken from the official Pokémon page as they are not specified here.
- Status and their effects apply, they are applied at the beginning or end of a turn, possible status are:
    - no status,
    - paralyzed: ground/electric types are immune, there is a 25% chance to lose your turn,
    - poisoned: poison/steel types are immune,the pokemon lose ⅛ of total hp,
    - confused: there is a 33% chance to take ⅛ of total hp AND to lose your turn,
    - sleep: lose your turn,
    - frozen: lose your turn,
    - burned: take ⅛ of total hp,
    - Confused: sleep, and frozen effects are guaranteed to go away after 5 turns, but can also randomly go away before that, the other effects can’t be deleted.
Effects are kept when the pokemon is switched out, except for confusion.
- Entry hazards apply, entry hazards are effects that happens anytime a pokemon is switched in, possible entry hazards are :
    - none,
    - spikes: flying types are immune, it has 3 stages, each stage deals respectively, ⅛ , ⅙ ,¼ of the pokemon’s max hp.
A pokemon can faint from entry hazards.
- Accuracy applies  (move have accuracy and can fail)
Attack formula is : 
damage = (attack - defense) * move_power * type_bonus * STAB * weather_bonus 
- Each move can only have a maximum of one effect: each abilities (move) can do only one effect in addition to damage.
 
 **Pokémon Team Building**
- Pokémon teams are not permitted to have duplicate specimens.



*Pokémon official chart* 
![typechart](/typechart.png)
(Source: https://pt.m.wikipedia.org/wiki/Ficheiro:Pokemon_Type_Chart.svg)

### Battle track specific rules:

- the stats points mentioned before are randomly choose
- Teams are composed of 3 to 6 Pokémon

### Championship track specific rules:

- Teams are exactly 3 Pokémon
- a roster of 51 Pokémon will be used


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

