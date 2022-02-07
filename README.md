## World Generation

Each world/enviroment consists of 3 files:
- ***<env_name>.txt*** - A layout file. `E` is the starting position, `G` is the goal. Walls are marked by `#`, and doors in the walls by `D`. Once door is reached, last action will be repeated once more, so to pass through the door.
- ***<env_name>_abstraction.txt*** - File that corresponds to the layout files, but all fields (except for doors and walls), are replaced with their abstract output.
- ***<env_name>_rules.txt*** - File that contains rules about the stochastic behaviour of the environment. If empty, environment will be deterministic.

### Example: Office World:
- Layout
```
#########
#   #   #
#   D  E#
#   #   #
##D######
#   #G  #
#   D   #
#   #   #
#########
```
- Abstraction
```
#########
#111#222#
#111D222#
#111#222#
##D######
#333#444#
#333D444#
#333#444#
#########
```
- Rules

To add stochastic behaviour to the environment we create "rules". Each tile of the enviroment can be assigned to single (or no) rule.
If no rule is assigned behaviour of the tile will be deterministic.

In the rules file, first we lay out the map found in layout file and assign `rule_id` (integer or char) to tiles that we want to behave in certain way.
Once the layout has been defined, we need to declare rules for each `rule_id`.
They are of the form

```
<rule_id>-<action>-[(<new_action>:<probability>)+]
```
where action is the action we are trying to execute, and new action is the action that can occur with declared probability.
If no action is specified for a rule, it will remain deterministic.

```
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
3-right-[right:0.5, down:0.25, left:0.25]
```