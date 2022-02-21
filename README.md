# POGE - Partially Observable Gridworld Enviroment 

## World Generation 

Each gridworld is defined in its own txt file. 
The file is divided in sections:
- Layout
- Abstraction
- Behaviour
- Rewards

Each section starts with the `===<section name>===`.

`Layout` section is necessity for minimal working examples.
All other sections are optional.
ALl sections should contain the same border structure.
In the `Layout` section, we define the layout of the enviroment and the starting and 
goal positions. `Abstraction` defines abstract outputs player will receive once he is on a tile. If no abstract output 
is given (for a tile or for the whole layout), observations consist of x and y coordinates.
`Behaviour` defines the transition probabilities between tiles. If not defined, environment will be deterministic.
`Rewards` define the reward structure, that is wheter a positive or a negative reward is received once player reaches 
a certain tile.

More details about each aspect of the world generations are presented under the example.

Reserved characters:
- `#` - wall
- `D` - door in the wall
- `E` - starting player location
- `G` - goal location
- `T` - terminals

## Example World
```
===Layout===

#########
#   #   #
#   D  E#
#   #   #
##D######
#   # G #
#   D   #
#   #   #
#########

===Abstraction===

#########
#111#222#
#111D222#
#111#222#
##D######
#333#444#
#333D444#
#333#444#
#########

===Behaviour===

#########
#1  # 3 #
# 2 D  E#
#1  # 2 #
##D######
# 1  #2 #
#   D   #
# 2 #3  #
#########

1-up-[up:0.75, right:0.25]
1-down-[down:0.75, right:0.25]
2-left-[left:0.8, up:0.1, down:0.1]
3-right-[right:0.7, down:0.2, left:0.1]

===Rewards===

#########
# a #   #
# a D   #
#*  #   #
##D######
#   #   #
# b D   #
#*  # *c#
#########

a:1
b:2
c:3
*:-5
```

### Abstraction

If `is_partially_obs` parameter of the constructor is set to True, 
instead of observing x and y coordinates, a value found on the tile will be returned. 
If tile is left empty, (x,y) coordinates will be returned.

### Behaviour

To add stochastic behaviour to the environment we create "rules". Each tile of the environment can be assigned to single (or no) rule.
If no rule is assigned behaviour of the tile will be deterministic.

In the rules file, first we lay out the map found in layout file and assign `rule_id` (integer or char) to tiles that we want to behave in certain way.
Once the layout has been defined, we need to declare rules for each `rule_id`.
They are of the form

```
<rule_id>-<action>-[(<new_action>:<probability>)+]
```
where action is the action we are trying to execute, and new action is the action that can occur with declared probability.
If no action is specified for a rule, it will remain deterministic.

### Rewards

Intermediate rewards can be declared by assigning an symbol to a tile. Then the reward of the tile will equal to the integer value mapping to the symbol.
Mapping of symbols to rewards can be seen in the example. It follows the `<symbol>:<reward>` syntax.

If `one_time_rewards` param is set to True (by default it is), then the reward for each tile will be received once per episode. 
Else it will be received every time player is on the reward tile.