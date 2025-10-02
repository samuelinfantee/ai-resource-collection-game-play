# AI Resource Collection Game (Project 1 Part 3)

This is my implementation of the two-player resource collection game for Intro to AI.  
The goal was to make two agents play against each other on a 5×5 grid, where they move around, pick up resources, and try to deliver them back to their bases. Under the hood, the decision-making is powered by **minimax search** and **alpha–beta pruning**.

I tried to keep the code clean and readable, with comments explaining each major piece. The fun part is comparing how plain minimax vs. alpha–beta perform in terms of expanded nodes and runtime.

---

## Features
- Grid-based game with different terrain types and movement costs.
- Resource pick-up and drop-off system with limited backpack capacity.
- Agents implemented with:
  - Random policy (baseline)
  - Minimax with a custom heuristic
  - Alpha–beta pruning for efficiency
- Collects stats: nodes expanded, runtime, cutoffs.
- Visualization of player movement over the board (using `matplotlib`).

---

## Heuristic
The evaluation function weights **delivered > carried > distance**, so agents prioritize winning conditions (delivering), then immediate advantage (carrying), then distance pressure (don’t wander around forever).  
It’s simple but works surprisingly well in practice.

---

## Example run
When you run the game with minimax vs minimax, you’ll see a full playthrough printed turn by turn. At the end, you get a summary like:


