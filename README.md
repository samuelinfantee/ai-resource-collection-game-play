# AI Resource Collection Game (Project 1 Part 3)
 
The goal was to make two agents play against each other on a 5×5 grid, where they move around, pick up resources, and try to deliver them back to their bases. Under the hood, the decision-making is powered by **minimax search** and **alpha–beta pruning**.

---

## Features
- Grid-based game with different terrain types and movement cost.
- Resource pick-up and drop-off system with limited backpack capacity.
- Each agent is implemented with:
  - Random policy (baseline)
  - Minimax with a custom heuristic
  - Alpha–beta pruning for efficiency
- Collects stats: nodes expanded, runtime, cutoffs.
- Visualization of player movement over the board (using `matplotlib`).

---

## Heuristic
The evaluation function weights **delivered > carried > distance**, so agents prioritize winning conditions (delivering), then immediate advantage (carrying), then distance pressure (don’t wander around forever).  
It’s simple but works well in practice.

---

## Example run
When you run the game with minimax vs minimax, you’ll see a full playthrough printed turn by turn. At the end, you get a summary like
## Search stats over the whole game.
## Minimax (plain): nodes expanded=1670, runtime=9.95 ms.
## Alpha–beta : nodes expanded=1200, runtime=6.48 ms, cutoffs=163
## Player A wins! Delivered A=5, B=4

## 8) How to run
- Make sure you have Python 3.9+ installed. Install dependencies:
- pip install matplotlib
- Then run the code "sinfante_Intro_AI_Project 1 part 3.py"
- (Optional) Change the agents or map in the __main__ section at the bottom of the file:
- result = play_game(
    map_name="Map4",
    agentA="minimax",
    agentB="random",
    depth=3,
    max_turns=500
)


