import math
from dataclasses import dataclass
from typing import Tuple, FrozenSet
import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random


#Intro to AI - Project 1 part 3
#By: Samuel Infante Trillos, U42773730

#Resource position_player_Agents.
resources = {
    "stones": [(3,4), (2,1), (2,2)],
    "irons": [(4,0), (4,3), (1,3)],
    "crystals": [(0,4), (1,1), (3,1)]
}


# 5x5 terrain grids: 
TEST_MAPS = {
    "Map1": [
        ['grassland','grassland','grassland','grassland','grassland'],
        ['grassland','grassland','grassland','grassland','grassland'],
        ['grassland','grassland','hills',     'grassland','grassland'],
        ['grassland','grassland','grassland','grassland','grassland'],
        ['grassland','grassland','grassland','grassland','grassland'],
    ],
    "Map2": [
        ['grassland','hills','hills','grassland','grassland'],
        ['grassland','hills','swamp','swamp','grassland'],
        ['hills','hills','hills','swamp','hills'],
        ['grassland','grassland','hills','grassland','grassland'],
        ['grassland','hills','grassland','hills','grassland'],
    ],
    "Map3": [
        ['grassland','grassland','grassland','grassland','hills'],
        ['grassland','mountain','mountain','mountain','grassland'],
        ['grassland','hills','hills','hills','grassland'],
        ['grassland','mountain','mountain','mountain','grassland'],
        ['hills','grassland','hills','grassland','hills'],
    ],
    "Map4": [
        ['grassland','mountain','grassland','mountain','hills'],
        ['mountain','grassland','mountain','mountain','grassland'],
        ['mountain','hills','grassland','mountain','swamp'],
        ['swamp','swamp','mountain','swamp','mountain'],
        ['hills','grassland','hills','swamp','swamp'],
    ],
}



#Terrain types and their movement costs.
terrain_types = {"grassland": 1, "hills": 2, "swamp": 3, "mountain": 4}
#Goal state for the resources to be collected.
goal = {"stones": 3, "irons": 2, "crystals": 1}
#Maximum capacity of resources in the backpack at any time.
backpack_capacity = 2
grid_x, grid_y = 5, 5  #Dimensions of the grid.


#--------------------------------------------------------------------------------------------------------
#Dataset
#--------------------------------------------------------------------------------------------------------
# Fixed order for the three resource types.
resource_order = ('stones', 'irons', 'crystals')

# Quick index lookup: 'stones' -> 0, 'irons' -> 1, 'crystals' -> 2
resource_id = {r: i for i, r in enumerate(resource_order)}

BASE_PLAYER_A_LOCATION = (0, 0) #initial position of player A
BASE_PLAYER_B_LOCATION = (4, 4) #initial position of player B

point = Tuple[int, int]
triplet = Tuple[int,int,int]
resource_position = Tuple[int,int,str]
resource_set = FrozenSet[resource_position]

@dataclass(frozen=True)
class State:
    position_player_A: point #Position of player A
    position_player_B: point #Position of player B
    backpack_A: triplet #Backpack contents from player A
    backpack_B: triplet#Backpack contents from player B
    delivered_A: triplet #Delivered resources from player A
    delivered_B: triplet #Delivered resources from player B
    remaining: resource_set #Positions of the remaining resources
    player_turn: str #Which player has the moving turn at the actual time

# Stats holder for search instrumentation
@dataclass
class SearchStats:
    nodes_expanded: int = 0
    cutoffs: int = 0       

#Both players start at their bases, backpacks empty, delivered = 0, all resources present, A moves first.
def make_start_state() -> State:
    #we calculate the resource locations
    rem = []
    for rname, locs in resources.items():
        for (x, y) in locs:
            entry = (x,y,rname)
            rem.append(entry)
    return State(
        position_player_A=BASE_PLAYER_A_LOCATION,
        position_player_B=BASE_PLAYER_B_LOCATION,
        backpack_A=(0, 0, 0),
        backpack_B=(0, 0, 0),
        delivered_A=(0, 0, 0),
        delivered_B=(0, 0, 0),
        remaining=frozenset(rem),
        player_turn='A'
    )

def count_resources_delivered_base(delivered: Tuple[int, int, int]) -> int: #Total count of items delivered to base (sum of the three resource types)
    return delivered[0] + delivered[1] + delivered[2]

#--------------------------------------------------------------------------------------------------------
#Helper functions
#--------------------------------------------------------------------------------------------------------

#1) //
def get_successors(position_player_Agent): #We use this function to get all valid successor positions from the current position in the grid.
    directions = [(0,1), (1,0), (0, -1), (-1,0)]  #Right, Down, Left, Up
    successors = []
    for d in directions:
        new_x, new_y = position_player_Agent[0] + d[0], position_player_Agent[1] + d[1]
        if 0 <= new_x < grid_x and 0 <= new_y < grid_y: #Each successor must be within the grid boundaries.
            successors.append((new_x, new_y)) #We add the successor if its valid.
    return successors

#2) //
def step_cost(a, b): #This function calculates the cost to move from tile a to tile b. The cost is determined by the terrain type of tile b. We check this cost in the terrain_grid variable.
    tx, ty = b
    tname = terrain_grid[tx][ty] 
    return terrain_types[tname]


#4) //
#Return list of possible moves for current player
#Not allowing to move into the opponents position.
#We already use the created function "get_successors", and remove the opponent position from the list.
def legal_moves(state: State):

    #get current and opponent positions
    positions = {'A': state.position_player_A, 'B': state.position_player_B}
    cur = positions[state.player_turn]
    opponent = positions['B' if state.player_turn == 'A' else 'A']

    possible = get_successors(cur) #Get all possible moves from current position.

    moves = []
    for (nx, ny) in possible:
        if (nx, ny) == opponent: #We cannot move into the opponent's position.
            continue
        moves.append((nx, ny))
    return moves


#5) Try to pickup if possible //
def get_resource_from_location(inventory_list, remaining_resources_set, new_position):

    #If the backpack is full, we cannot pick up more resources.
    current_inventory = sum(inventory_list)
    space_left = backpack_capacity - current_inventory
    if space_left <= 0:
        return
    
    matching_entry = None
    for entry in remaining_resources_set:
        x, y, kind = entry
        if (x, y) == new_position:
            matching_entry = entry
            break

    # Nothing to pick up here
    if matching_entry is None:
        return

    # Update inventory and remove the resource from the map
    _, _, kind = matching_entry
    idx = resource_id[kind] 
    inventory_list[idx] += 1
    remaining_resources_set.remove(matching_entry)


#6) Makes the future move effective by updating location and depositing at base if we are located there or picking up a resource if we are located at a tile with a resource. //
def apply_move(current: State, successor: Tuple[int, int]) -> State:

    #Get the current information from the state
    remaining_resources = set(current.remaining)
    deliv_A = list(current.delivered_A)
    deliv_B = list(current.delivered_B)
    inv_A = list(current.backpack_A)
    inv_B = list(current.backpack_B)

    #Who moves
    A_turn = (current.player_turn == "A")

    #update new positions after move:
    new_position_player_A = successor if A_turn == True else current.position_player_A
    new_position_player_B = current.position_player_B if A_turn else successor

    #Get active player's attributes
    active_inventory = inv_A if A_turn else inv_B
    active_delivered = deliv_A if A_turn else deliv_B
    active_base_location = BASE_PLAYER_A_LOCATION if A_turn else BASE_PLAYER_B_LOCATION
    active_position = new_position_player_A if A_turn else new_position_player_B

    #Deposit resources at base if we are carrying any resource
    if active_position == active_base_location and sum(active_inventory) > 0:
        for i in range(3):
            active_delivered[i] += active_inventory[i]
            active_inventory[i] = 0
    
    #If we are at a tile with resources, try to pick it up:
    get_resource_from_location(active_inventory, remaining_resources, active_position)

    #update the next turn:
    next_turn = "B" if A_turn else "A"

    # Build new frozen state
    return State(
        position_player_A=new_position_player_A,
        position_player_B=new_position_player_B,
        backpack_A=tuple(inv_A),
        backpack_B=tuple(inv_B),
        delivered_A=tuple(deliv_A),
        delivered_B=tuple(deliv_B),
        remaining=frozenset(remaining_resources),
        player_turn=next_turn
    )  

# 7) Check if the game is over: all resources collected and delivered to bases. //
def game_over(state: State) -> bool:
    if state.remaining:
        return False
    if any(state.backpack_A) or any(state.backpack_B):
        return False
    return True


# 8) Calculate utility //
def utility(state: State) -> int:
    score_A = count_resources_delivered_base(state.delivered_A)
    score_B = count_resources_delivered_base(state.delivered_B)
    return score_A - score_B

# 9) manhattan distance heuristic //
def manhattan(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    x1, y1 = a[0],a[1]
    x2, y2 = b[0],b[1]
    dx, dy = x1-x2, y1-y2
    abs_dx, abs_dy = abs(dx), abs(dy)
    distance = abs_dx + abs_dy
    return distance

# 10) Nearest resource distance heuristic //
def nearest_resource_dist(position: Tuple[int,int], remaining: FrozenSet[Tuple[int,int,str]]) -> int:
    if not remaining:
        return 0
    px, py = position
    best_dist: int | None = None

    for rx, ry, name in remaining:
        d = manhattan((px, py), (rx, ry))
        if best_dist is None or d < best_dist:
            best_dist = d
        if best_dist == 0:  # can't get closer than same tile
            break

    return best_dist if best_dist is not None else 0

# 11) helper to move 1 step toward the active player's base (greedy,makes random have an end) //
def move_greedy_to_the_base(state: State):
    moves = legal_moves(state)
    if not moves:
        return None
    base = BASE_PLAYER_A_LOCATION if state.player_turn == "A" else BASE_PLAYER_B_LOCATION
    best_d = min(manhattan(m, base) for m in moves)
    best_moves = [m for m in moves if manhattan(m, base) == best_d]
    return random.choice(best_moves)


#--------------------------------------------------------------------------------------------------------
#Evaluation and minimax:
#--------------------------------------------------------------------------------------------------------
#1) //
#Heuristic used to evaluate non-terminal states.
#Simply calculates how many resources have been delivered + carried by each player, and returns the difference.
def evaluate_competition(state: State) -> int:
    a_delivered = count_resources_delivered_base(state.delivered_A)
    b_delivered = count_resources_delivered_base(state.delivered_B)
    a_carry = sum(state.backpack_A)
    b_carry = sum(state.backpack_B)

    # Distances: to base if carrying, else to nearest resource
    a_dist = manhattan(state.position_player_A, BASE_PLAYER_A_LOCATION) if a_carry > 1 else nearest_resource_dist(state.position_player_A, state.remaining)
    b_dist = manhattan(state.position_player_B, BASE_PLAYER_B_LOCATION) if b_carry > 1 else nearest_resource_dist(state.position_player_B, state.remaining)

    # 1) Resources delivered dominates (win condition)
    delivered_score = 80.0 * (a_delivered - b_delivered)

    # 2) Resource carried matters, but less
    carried_score = 25.0  * (a_carry    - b_carry)

    # 3) Distance pressure: if carrying, get closer to your base; if empty, get closer to any resource
    distance_score = (-1.35 if a_carry > 0 else -0.75) * a_dist + (+1.35 if b_carry > 0 else +0.75) * b_dist

    return (delivered_score + carried_score + distance_score)


#2) // uses pruning
def minimax_ab_algorithm(state: State, depth: int, alpha: float, beta: float, stats: SearchStats | None = None):

    if stats is None:                      
        stats = SearchStats()            
    stats.nodes_expanded += 1              
    
    over = game_over(state)
    if over:
        return utility(state), None
    if depth == 0:
        return evaluate_competition(state), None

    moves = legal_moves(state)
    if not moves: #If no legal moves, return evaluation.
        return evaluate_competition(state), None
    
    maximizing = (state.player_turn == "A")
    best_score = -math.inf if maximizing else math.inf #if we are maximizing, we start with -inf, else +inf.
    best_move = None

    for move in moves:
        child = apply_move(state, move)
        score, _ = minimax_ab_algorithm(child, depth-1, alpha, beta, stats)
        
        #if the new score is better than previous:
        if maximizing:
            if score > best_score:
                best_score, best_move = score, move
            alpha = max(alpha, best_score)
        else:
            if score < best_score:
                best_score, best_move = score, move
            beta = min(beta, best_score)

        if beta <= alpha: #Alpha-beta pruning
            stats.cutoffs += 1
            break
    
    return best_score, best_move


# 3) Plain (non-pruning) minimax for totals comparison //
def minimax_plain_algorithm(state: State, depth: int, stats: SearchStats):
    # Count this node
    stats.nodes_expanded += 1
    # Terminal state check
    if game_over(state):
        final_score = utility(state)
        return final_score, None
    # Depth limit reached -> use heuristic
    if depth == 0:
        estimate = evaluate_competition(state)
        return estimate, None
    # Generate moves
    moves = legal_moves(state)
    if not moves:
        # No actions available -> use heuristic
        fallback = evaluate_competition(state)
        return fallback, None

    # Determine if we are maximizing or minimizing
    maximizing = (state.player_turn == "A")
    # Initialize best trackers
    if maximizing:
        best_score = -math.inf
    else:
        best_score = math.inf
    best_move = None

    # Evaluate each move one by one
    for move in moves:
        # Apply the move to get the next state
        child = apply_move(state, move)
        # Recurse to the next depth
        child_score, _ = minimax_plain_algorithm(child, depth - 1, stats)
        # Update the best score/move depending on player type
        if maximizing:
            if child_score > best_score:
                best_score = child_score
                best_move = move
        else:
            if child_score < best_score:
                best_score = child_score
                best_move = move
    # Return the best result found
    return best_score, best_move         

    
#3) wrapper function to get the best action for the current player. //
def choose_minimax_action(state: State, depth: int = 3):
    score, move = minimax_ab_algorithm(state, depth, -math.inf, math.inf)
    return move


#--------------------------------------------------------------------------------------------------------
#Agents and play loop
#--------------------------------------------------------------------------------------------------------
#Random agent that chooses any move uniformly at random from the list of legal moves. //
def random_agent(state: State):
    moves = legal_moves(state)
    if not moves:
        return None

    # Figure out whose turn it is 
    turn = state.player_turn
    base = BASE_PLAYER_A_LOCATION if turn == 'A' else BASE_PLAYER_B_LOCATION
    carrying = (sum(state.backpack_A) if turn == 'A' else sum(state.backpack_B))

    # If carrying, bias toward home: scan once to collect all closest moves
    if carrying:
        best_moves = []
        best_d = math.inf
        random.shuffle(moves)
        for m in moves:
            d = manhattan(m, base)
            if d < best_d:
                best_d = d
                best_moves = [m]
            elif d == best_d:
                best_moves.append(m)
        return random.choice(best_moves)

    # Otherwise, act purely at random
    return random.choice(moves)



#Main game play loop //
def play_game(map_name: str, agentA: str = "minimax", agentB: str = "random", depth: int = 3, max_turns: int = 500):

    #terrain selection
    global terrain_grid
    try:
        terrain_grid = TEST_MAPS[map_name]
    except KeyError:
        raise ValueError(f"Unknown map '{map_name}'. Options: {list(TEST_MAPS.keys())}")

    #start state:
    state = make_start_state()

    # Visit times (user for the graph)
    times_A: dict[Tuple[int,int], list[int]] = {}
    times_B: dict[Tuple[int,int], list[int]] = {}
    # initial positions counted as times 0 (A) and 1 (B)
    times_A.setdefault(state.position_player_A, []).append(0)
    times_B.setdefault(state.position_player_B, []).append(1)
    global_time = 2  # next turn uses t=2

    #totals across the whole game
    total_plain_nodes = 0.0        
    total_plain_time  = 0.0        
    total_ab_nodes    = 0.0        
    total_ab_time     = 0.0        
    total_ab_cutoffs  = 0.0      

    #grab the method selected by the player
    def pick_move(method: str, s: State):
        if method == "minimax":
            return choose_minimax_action(s, depth=depth)
        if method == "random":
            return random_agent(s)
        raise ValueError(f"Unknown agent '{method}' (use 'minimax' or 'random')")

    # player_turn loop)
    turns = 0
    while turns < max_turns and not game_over(state):
        turns+=1

        a_delivered = count_resources_delivered_base(state.delivered_A)
        b_delivered = count_resources_delivered_base(state.delivered_B)

        print(f"\nTurn {turns}, Player {state.player_turn} to move, Delivered A={a_delivered}, B={b_delivered}, Carry A={sum(state.backpack_A)}, B={sum(state.backpack_B)}")


        # run BOTH searches once per turn to accumulate TOTALS:-----------------
        # Plain minimax (nodes + time)
        _plain_stats = SearchStats()                          
        _t0 = time.perf_counter()                             
        _ = minimax_plain_algorithm(state, depth, _plain_stats)  
        _dt = time.perf_counter() - _t0                       
        total_plain_nodes += _plain_stats.nodes_expanded      
        total_plain_time  += _dt                              

        # Alpha–beta (nodes + cutoffs + time)
        _ab_stats = SearchStats()                             
        _t0 = time.perf_counter()                             
        _ = minimax_ab_algorithm(state, depth, -math.inf, math.inf, _ab_stats)  
        _dt = time.perf_counter() - _t0                       
        total_ab_nodes   += _ab_stats.nodes_expanded          
        total_ab_cutoffs += _ab_stats.cutoffs                 
        total_ab_time    += _dt                               
        # ----------------------------------------------------------------------------

        #Get the policy yser for the player's current turn
        A_turn = (state.player_turn == "A")
        method = agentA if A_turn else agentB

        #Endgame safety: if no resources remain on the map and the active player is carrying, override any agent and step greedily toward base so the game terminates.
        if not state.remaining:
            carrying = sum(state.backpack_A) if A_turn else sum(state.backpack_B)  
            if carrying > 0:                                                       
                move = move_greedy_to_the_base(state)                                  
            else:                                                                  
                move = pick_move(method, state)                                    
        else:
            move = pick_move(method, state) #choose a action

        #if the function returns no legal moves:
        if move is None:
            print(f"Player {state.player_turn} has no legal moves and must pass his turn.")
        
            state = State(
                position_player_A=state.position_player_A,
                position_player_B=state.position_player_B,
                backpack_A=state.backpack_A,
                backpack_B=state.backpack_B,
                delivered_A=state.delivered_A,
                delivered_B=state.delivered_B,
                remaining=state.remaining,
                player_turn='B' if A_turn else 'A')
            global_time += 1
            continue

        #Apply move if valid
        state = apply_move(state, move)

        #record the time the mover ENTERED the new cell (We use this for the graph) -----
        if A_turn:
            position = state.position_player_A
            times_A.setdefault(position, []).append(global_time)
        else:
            position = state.position_player_B
            times_B.setdefault(position, []).append(global_time)
        global_time += 1
        #------------------------------------------------------------------------------------

        if not A_turn:
            print(f"Player A moves to {state.position_player_A}")
        else:
            print(f"Player B moves to {state.position_player_B}")

        
    #Game over
    finished = game_over(state)
    print("\n" + ("Game over!" if finished else "Max turns reached."))
    a_delivered = count_resources_delivered_base(state.delivered_A)
    b_delivered = count_resources_delivered_base(state.delivered_B)
    utility = a_delivered - b_delivered

    if utility > 0:
        print(f"Player A wins! Delivered A={a_delivered}, B={b_delivered}")
    elif utility < 0:
        print(f"Player B wins! Delivered A={a_delivered}, B={b_delivered}")
    else:
        print(f"Tie. Delivered A={a_delivered}, B={b_delivered}")
    

    #final one-shot summary of TOTALS over the whole game
    print("\nSearch stats over the whole game")                        
    print(f"Minimax (plain): nodes expanded={int(total_plain_nodes)}, runtime={total_plain_time*1000:.2f} ms")  
    print(f"Alpha–beta     : nodes expanded={int(total_ab_nodes)}, runtime={total_ab_time*1000:.2f} ms, cutoffs={int(total_ab_cutoffs)}") 

    return {
        "winner": 'A' if utility > 0 else 'B' if utility < 0 else 'Tie',
        "A": a_delivered,
        "B": b_delivered,
        "turns": turns,
        "final_state": state,
        "visit_times_A": times_A,
        "visit_times_B": times_B,
        #including totals in the result dict
        "search_totals": { 
            "minimax_plain": {"nodes": int(total_plain_nodes), "time_sec": total_plain_time}, 
            "alpha_beta":    {"nodes": int(total_ab_nodes),   "time_sec": total_ab_time, "cutoffs": int(total_ab_cutoffs)}, 
        }, 
    }

        
#--------------------------------------------------------------------------------------------------------------------
#Graph Creation
#--------------------------------------------------------------------------------------------------------------------


def plot_board_with_times_pvp(terrain_grid, times_A: dict[Tuple[int, int], list[int]], times_B: dict[Tuple[int, int], list[int]],
                              title: str = "Board with visit times (A=red, B=blue)",resource_names: Tuple[str, str, str] = ('stones', 'irons', 'crystals')):

    # --- helpers --------------------------------------------------------------
    def to_codes(grid):
        t2c = {'grassland': 0, 'hills': 1, 'swamp': 2, 'mountain': 3}
        return [[t2c[cell] for cell in row] for row in grid]

    def draw_grid(ax, w, h):
        ax.set_xticks(range(w))
        ax.set_yticks(range(h))
        ax.set_xticks([x - 0.5 for x in range(1, w)], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, h)], minor=True)
        ax.grid(which='minor', color='black', linewidth=1)
        ax.set_xticklabels(range(w))
        ax.set_yticklabels(range(h))

    def annotate_times(ax, h, w):
        for i in range(h):
            for j in range(w):
                a_t = times_A.get((i, j))
                b_t = times_B.get((i, j))
                if a_t:
                    ax.text(j, i - 0.18, "A: " + ", ".join(map(str, a_t)),
                            ha='center', va='center', fontsize=6, color='red')
                if b_t:
                    ax.text(j, i + 0.18, "B: " + ", ".join(map(str, b_t)),
                            ha='center', va='center', fontsize=6, color='blue')

    def draw_bases(ax):
        ax.scatter(BASE_PLAYER_A_LOCATION[1], BASE_PLAYER_A_LOCATION[0], s=180, marker='*', color='red', label='A base')
        ax.scatter(BASE_PLAYER_B_LOCATION[1], BASE_PLAYER_B_LOCATION[0], s=180, marker='*', color='blue', label='B base')

    def draw_resources(ax):
        style = {
            'stones':   dict(marker='X', edgecolor='black',     facecolor='none', label='stones'),
            'irons':    dict(marker='D', edgecolor='gray',    facecolor='none', label='irons'),
            'crystals': dict(marker='P', edgecolor='blue', facecolor='none', label='crystals'),
        }
        for rname in resource_names:
            if rname not in resources:
                continue
            st = style[rname]
            for (rx, ry) in resources[rname]:
                ax.scatter(ry, rx, s=150, marker=st['marker'],facecolors=st['facecolor'], edgecolors=st['edgecolor'], linewidths=2)
                ax.text(ry, rx, rname[0].upper(), color=st['edgecolor'],
                        ha='center', va='center', fontsize=9, fontweight='bold')

    def legend_handles():
        return [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor="#a6d96a", markersize=10, label='grassland'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor="#fdae61", markersize=10, label='hills'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor="#74add1", markersize=10, label='swamp'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor="#bdbdbd", markersize=10, label='mountain'),
            plt.Line2D([0], [0], marker='*', color='red',  markersize=10, label='A base'),
            plt.Line2D([0], [0], marker='*', color='blue', markersize=10, label='B base'),
            plt.Line2D([0], [0], color='red',  label='A times'),
            plt.Line2D([0], [0], color='blue', label='B times'),
        ]

    # --- figure setup ---------------------------------------------------------
    H, W = len(terrain_grid), len(terrain_grid[0])
    grid_codes = to_codes(terrain_grid)
    cmap = ListedColormap(["#c9ff8b", "#ffac91",  "#baf8ff", "#b38080", ]) #grassland, hills, swamp, mountains

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(grid_codes, cmap=cmap, origin='upper')
    ax.set_title(title)

    draw_grid(ax, W, H)
    annotate_times(ax, H, W)
    draw_bases(ax)
    draw_resources(ax)

    # legend (terrain + bases + time colors + resource markers)
    handles = legend_handles()
    # add resource markers to legend based on same styling used for plotting
    for rname, edge, marker in [('stones', 'red', 'o'),
                                ('irons', 'blue', 's'),
                                ('crystals', 'magenta', '^')]:
        if rname in resource_names:
            handles.append(plt.Line2D([0], [0], marker=marker, color='w', markerfacecolor='none', markeredgecolor=edge, markersize=8, label=rname))
    ax.legend(handles=handles, loc='upper right', fontsize=8, frameon=True)

    plt.tight_layout()
    plt.show()



#------------------------------------------------------------------------------------------------------------------------------
#Main function
#------------------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    map_name = "Map4"
    result = play_game(
        map_name=map_name,
        agentA="minimax",
        agentB="minimax",
        depth=3,
        max_turns=1000
    )

    # draw the visit timeline for both players
    plot_board_with_times_pvp(
        terrain_grid,
        result["visit_times_A"],
        result["visit_times_B"],
        title=f"{map_name} | A vs B visit times"
    )
