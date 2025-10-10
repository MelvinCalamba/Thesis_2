from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import networkx as nx
import numpy as np
import time
import heapq
from math import sqrt
import random
from typing import List, Tuple, Optional, Dict, Any
import math
from collections import defaultdict, deque
import os

app = Flask(__name__)
CORS(app)

# ====================== Data Loading and Preparation ======================
def load_and_prepare_data(filepath="lb_road_data.csv"):
    try:
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Road data file '{filepath}' not found")
            
        df = pd.read_csv(filepath)
        
        # Drop unnecessary columns if they exist
        columns_to_drop = ['street_name', 'road_class', 'road_type']
        for col in columns_to_drop:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # Convert categorical variables to numerical ones
        if 'road_direction' in df.columns:
            df['road_direction'] = df['road_direction'].map({'oneway': 0, 'twoway': 1})
        
        # Extract latitude and longitude from start_node and end_node
        def extract_coords(coord_str):
            try:
                if isinstance(coord_str, str) and coord_str.startswith("("):
                    lat, lon = coord_str.strip('()').split(',')
                    return float(lat.strip()), float(lon.strip())
                return (0.0, 0.0)
            except:
                return (0.0, 0.0)
        
        df['start_lat'], df['start_lon'] = zip(*df['start_node'].apply(extract_coords))
        df['end_lat'], df['end_lon'] = zip(*df['end_node'].apply(extract_coords))
        
        # Calculate cost - adjust weights as needed
        if 'is_blocked' not in df.columns:
            df['is_blocked'] = 0
        
        # Ensure required columns exist
        required_columns = ['duration_seconds', 'distance_meters']
        for col in required_columns:
            if col not in df.columns:
                df[col] = 1.0  # Default value if column is missing
        
        df['cost'] = df['duration_seconds'] + df['distance_meters'] * 0.1 + (df['is_blocked'] * 1000)
        
        return df
    except Exception as e:
        app.logger.error(f"Error loading data: {str(e)}")
        raise

# ====================== Graph Construction ======================
def build_graph(df, cost_mode="hybrid"):
    """Build graph from dataframe with configurable cost metric"""
    def compute_edge_weight(row, mode="duration"):
        if mode == "duration":
            return row["duration_seconds"]
        elif mode == "distance":
            return row["distance_meters"]
        elif mode == "hybrid":
            return 0.7 * row["duration_seconds"] + 0.3 * row["distance_meters"]
        else:
            raise ValueError("Invalid cost metric. Use 'duration', 'distance', or 'hybrid'.")

    G = nx.DiGraph()
    for _, row in df.iterrows():
        if not row.get("is_blocked", False):
            weight = compute_edge_weight(row, cost_mode)
            u, v = row["start_node"], row["end_node"]

            # Oneway vs. twoway edges
            if 'road_direction' in row and row["road_direction"] == 0:  # oneway
                G.add_edge(u, v,
                           weight=weight,
                           distance=row["distance_meters"],
                           duration=row["duration_seconds"],
                           is_blocked=row["is_blocked"])
            else:  # twoway or default
                G.add_edge(u, v, weight=weight,
                           distance=row["distance_meters"],
                           duration=row["duration_seconds"],
                           is_blocked=row["is_blocked"])
                G.add_edge(v, u, weight=weight,
                           distance=row["distance_meters"],
                           duration=row["duration_seconds"],
                           is_blocked=row["is_blocked"])
    return G

# ====================== Disaster Simulation System ======================
class DisasterSimulator:
    """Simulate different types of disasters and their effects on the road network"""
    
    def __init__(self, graph, df):
        """Initialize the disaster simulator with graph and dataframe"""
        self.graph = graph
        self.df = df
        self.disaster_type = None
        self.origin_node = None
        self.disaster_effects = {}
    
    def simulate_fire(self, origin_node, end_node=None):
        """Simulate fire disaster with clustered, progressive effects and containment"""
        self.disaster_type = "fire"
        self.origin_node = origin_node
        self.disaster_effects = {}
        
        # Store end node for special handling
        self.end_node = end_node
        
        # Start with the edges connected to the origin node
        fire_front = [origin_node]
        visited = set([origin_node])
        blocked_edges_count = 0
        max_blocked_edges = random.randint(2, 3)  # Limit to 2-3 blocked edges
        
        # Process fire spread with containment
        while fire_front and blocked_edges_count < max_blocked_edges:
            current_node = fire_front.pop(0)
            
            # Get all edges from current node
            for neighbor in self.graph.neighbors(current_node):
                edge = (current_node, neighbor)
                
                # If this edge hasn't been processed yet and we haven't reached the limit
                if edge not in self.disaster_effects and blocked_edges_count < max_blocked_edges:
                    
                    # Special handling: if this edge leads to the end node, don't block it completely
                    # Instead add a delay to make it harder but still reachable
                    if neighbor == end_node:
                        # Add 1-2 minute delay (60-120 seconds) instead of blocking
                        delay_seconds = random.randint(60, 120)
                        self.disaster_effects[edge] = {
                            'type': 'fire',
                            'delay': delay_seconds,
                            'completely_blocked': False,
                            'is_end_node_access': True
                        }
                        app.logger.info(f"Fire disaster: Added delay to end node access edge {edge}")
                    else:
                        # The first edge (connected to origin) is always blocked
                        if current_node == origin_node:
                            self.disaster_effects[edge] = {
                                'type': 'fire',
                                'delay': 0,
                                'completely_blocked': True,
                                'is_end_node_access': False
                            }
                            blocked_edges_count += 1
                        else:
                            # For subsequent edges, 50% chance of being blocked
                            if random.random() < 0.5 and blocked_edges_count < max_blocked_edges:
                                self.disaster_effects[edge] = {
                                    'type': 'fire',
                                    'delay': 0,
                                    'completely_blocked': True,
                                    'is_end_node_access': False
                                }
                                blocked_edges_count += 1
                    
                    # Add neighbor to fire front if not visited and we haven't reached the limit
                    if (neighbor not in visited and edge in self.disaster_effects and 
                        blocked_edges_count < max_blocked_edges and neighbor != end_node):
                        visited.add(neighbor)
                        fire_front.append(neighbor)
                    
                    # Break if we've reached the maximum blocked edges
                    if blocked_edges_count >= max_blocked_edges:
                        break
            
            # Break if we've reached the maximum blocked edges
            if blocked_edges_count >= max_blocked_edges:
                break
        
        return self.disaster_effects
    
    def simulate_earthquake(self):
        """Simulate earthquake disaster with scattered, random effects"""
        self.disaster_type = "earthquake"
        self.disaster_effects = {}
        
        # Select 60-70 random edges to be affected
        all_edges = list(self.graph.edges())
        num_affected = random.randint(60, 70)
        affected_edges = random.sample(all_edges, min(num_affected, len(all_edges)))
        
        for u, v in affected_edges:
            # 70% chance of delay, 30% chance of complete blockage
            if random.random() < 0.7:
                # Add 4-5 minutes (240-300 seconds) of additional travel time
                delay_seconds = random.randint(240, 300)
                self.disaster_effects[(u, v)] = {
                    'type': 'earthquake',
                    'delay': delay_seconds,
                    'completely_blocked': False
                }
            else:
                # Completely blocked
                self.disaster_effects[(u, v)] = {
                    'type': 'earthquake',
                    'delay': 0,
                    'completely_blocked': True
                }
        
        return self.disaster_effects
    
    def simulate_flood(self):
        """Simulate flood disaster"""
        self.disaster_type = "flood"
        self.disaster_effects = {}
        
        # Select 40-50 random edges to be affected by flood
        all_edges = list(self.graph.edges())
        num_affected = random.randint(40, 50)
        affected_edges = random.sample(all_edges, min(num_affected, len(all_edges)))
        
        for u, v in affected_edges:
            # Flood causes complete blockage in most cases
            if random.random() < 0.8:
                self.disaster_effects[(u, v)] = {
                    'type': 'flood',
                    'delay': 0,
                    'completely_blocked': True
                }
            else:
                # Some edges just have delays due to water
                delay_seconds = random.randint(180, 420)  # 3-7 minutes delay
                self.disaster_effects[(u, v)] = {
                    'type': 'flood',
                    'delay': delay_seconds,
                    'completely_blocked': False
                }
        
        return self.disaster_effects
    
    def apply_disaster_effects(self, df):
        """Apply disaster effects to the dataframe"""
        df_modified = df.copy()
        
        for (u, v), effect in self.disaster_effects.items():
            # Find the corresponding row in the dataframe
            mask = (df_modified['start_node'] == u) & (df_modified['end_node'] == v)
            
            if mask.any():
                idx = df_modified[mask].index[0]
                
                if effect['completely_blocked']:
                    # Mark as completely blocked
                    df_modified.at[idx, 'is_blocked'] = 1
                else:
                    # Add delay to duration
                    df_modified.at[idx, 'duration_seconds'] += effect['delay']
        
        return df_modified
    
    def get_disaster_info(self):
        """Get information about the current disaster"""
        if not self.disaster_effects:
            return None
            
        blocked_count = sum(1 for effect in self.disaster_effects.values() 
                           if effect['completely_blocked'])
        delayed_count = len(self.disaster_effects) - blocked_count
        
        # Count end node access edges (delayed but not blocked)
        end_node_access_count = sum(1 for effect in self.disaster_effects.values() 
                                   if effect.get('is_end_node_access', False))
        
        return {
            'type': self.disaster_type,
            'affected_edges': len(self.disaster_effects),
            'blocked_edges': blocked_count,
            'delayed_edges': delayed_count,
            'end_node_access_edges': end_node_access_count,
            'origin_node': self.origin_node
        }
    
# ====================== Helper Functions ======================
def parse_coords(node):
    """Parse coordinates from node string"""
    if isinstance(node, str) and node.startswith("("):
        lat, lon = node.strip("()").split(",")
        return float(lat.strip()), float(lon.strip())
    elif isinstance(node, (tuple, list)) and len(node) == 2:
        return node
    return (0.0, 0.0)

def compute_travel_time(G, path):
    if not path or len(path) < 2:
        return float("inf")  # no path found
    try:
        total_seconds = sum(G[u][v]["duration"] for u, v in zip(path[:-1], path[1:]))
        return total_seconds / 60  # convert to minutes
    except KeyError:
        return float("inf")  # invalid/broken path

def heuristic(u, v):
    """Euclidean distance heuristic for A* and Greedy BFS"""
    u_lat, u_lon = parse_coords(u)
    v_lat, v_lon = parse_coords(v)
    return sqrt((u_lat - v_lat) ** 2 + (u_lon - v_lon) ** 2)

# ====================== Pathfinding Algorithms ======================
def run_dijkstra(G, start, end):
    """Optimized Dijkstra's algorithm"""
    start_time = time.perf_counter()
    
    if start == end:
        return [start], 0, time.perf_counter() - start_time
    
    try:
        path = nx.shortest_path(G, source=start, target=end, weight="weight", method="dijkstra")
        total_cost = sum(G[u][v]["weight"] for u, v in zip(path[:-1], path[1:]))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None, float("inf"), time.perf_counter() - start_time
    return path, total_cost, time.perf_counter() - start_time

def run_astar(G, start, end):
    """Optimized A* algorithm"""
    start_time = time.perf_counter()
    
    if start == end:
        return [start], 0, time.perf_counter() - start_time
    
    try:
        path = nx.astar_path(G, start, end, heuristic=heuristic, weight="weight")
        total_cost = sum(G[u][v]["weight"] for u, v in zip(path[:-1], path[1:]))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None, float("inf"), time.perf_counter() - start_time
    return path, total_cost, time.perf_counter() - start_time

def run_greedy_bfs(G, start, end):
    """Optimized Greedy Best-First Search"""
    start_time = time.perf_counter()
    
    if start == end:
        return [start], 0, time.perf_counter() - start_time
    
    try:
        open_set = [(heuristic(start, end), start)]
        came_from = {}
        visited = set()

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == end:
                # reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                # Calculate actual cost
                cost = sum(G[u][v]["weight"] for u, v in zip(path[:-1], path[1:]))
                return path, cost, time.perf_counter() - start_time

            if current in visited:
                continue
            visited.add(current)

            for neighbor in G.neighbors(current):
                if neighbor not in visited:
                    # Check if edge is not blocked before considering it
                    if not G[current][neighbor].get('is_blocked', False):
                        came_from[neighbor] = current
                        heapq.heappush(open_set, (heuristic(neighbor, end), neighbor))

        return None, float("inf"), time.perf_counter() - start_time
        
    except Exception as e:
        app.logger.error(f"Greedy BFS error: {str(e)}")
        return None, float("inf"), time.perf_counter() - start_time

class OptimizedAntColony:
    """Optimized Ant Colony Optimization for pathfinding"""

    def __init__(self, graph, n_ants=20, n_iterations=100, decay=0.5, alpha=1, beta=3, gamma=10,
                 elitist_factor=2, stagnation_limit=15):
        self.graph = graph
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.elitist_factor = elitist_factor
        self.stagnation_limit = stagnation_limit

        self._initialize_pheromones()
        self.best_path = None
        self.best_cost = float('inf')
        self.stagnation_count = 0
        self.dead_end_nodes = set()

    def _initialize_pheromones(self):
        """Initialize pheromones for all edges in the graph"""
        self.pheromone = {}
        for u, v, d in self.graph.edges(data=True):
            if not d.get('is_blocked', False):
                self.pheromone[(u, v)] = 1 / max(0.1, d['weight'])
            else:
                self.pheromone[(u, v)] = 1e-10

    def _is_edge_blocked(self, u, v):
        """Check if an edge is blocked"""
        try:
            return self.graph[u][v].get('is_blocked', False)
        except KeyError:
            return True

    def _find_dead_end_nodes(self, end):
        """Efficient dead-end detection using BFS"""
        reachable = set()
        queue = [end]

        while queue:
            node = queue.pop(0)
            if node in reachable:
                continue
            reachable.add(node)

            for pred in self.graph.predecessors(node):
                if (not self._is_edge_blocked(pred, node) and
                    pred not in reachable):
                    queue.append(pred)

        self.dead_end_nodes = set(self.graph.nodes()) - reachable

        # Also mark nodes that only lead to dead ends
        changed = True
        while changed:
            changed = False
            for node in set(self.graph.nodes()) - self.dead_end_nodes:
                has_valid_exit = False
                for neighbor in self.graph.neighbors(node):
                    if (not self._is_edge_blocked(node, neighbor) and
                        neighbor not in self.dead_end_nodes):
                        has_valid_exit = True
                        break

                if not has_valid_exit and node != end:
                    self.dead_end_nodes.add(node)
                    changed = True

    def run(self, start, end, max_retries=3):
        start_time = time.time()

        self._find_dead_end_nodes(end)

        if self.best_path and not self._is_path_valid(self.best_path):
            self.best_path = None
            self.best_cost = float('inf')

        for retry in range(max_retries):
            found_valid_path = False

            for iteration in range(self.n_iterations):
                paths, costs = self._explore(start, end)

                if paths:
                    found_valid_path = True
                    self._update_best_solution(paths, costs)

                    if self.stagnation_count >= self.stagnation_limit:
                        break

                self._update_pheromones(paths, costs)

            if found_valid_path and self.best_path:
                exec_time = time.time() - start_time
                return self.best_path, self.best_cost, exec_time

            self.alpha = max(1, self.alpha - 0.5)
            self.beta = min(20, self.beta + 1)

        self._find_dead_end_nodes(end)
        paths, costs = self._explore(start, end)
        self._update_best_solution(paths, costs)

        exec_time = time.time() - start_time
        return self.best_path, self.best_cost, exec_time

    def _is_path_valid(self, path):
        """Check if a path is still valid"""
        if not path or len(path) < 2:
            return False

        for u, v in zip(path[:-1], path[1:]):
            if self._is_edge_blocked(u, v):
                return False
        return True

    def _explore(self, start, end):
        paths, costs = [], []
        for _ in range(self.n_ants):
            path = self._construct_path(start, end)
            if path and path[-1] == end:
                cost = sum(self.graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
                paths.append(path)
                costs.append(cost)
        return paths, costs

    def _construct_path(self, start, end):
        path = [start]
        current = start
        visited = set([start])
        max_steps = 100

        for step in range(max_steps):
            if current == end:
                return path

            neighbors = []
            for neighbor in self.graph.neighbors(current):
                if (not self._is_edge_blocked(current, neighbor) and
                    neighbor not in visited and
                    neighbor not in self.dead_end_nodes):
                    neighbors.append(neighbor)

            if not neighbors:
                if len(path) > 1:
                    path.pop()
                    current = path[-1]
                    continue
                else:
                    return None

            next_node = self._select_next_node(current, neighbors, end)
            if next_node is None:
                return None

            path.append(next_node)
            visited.add(next_node)
            current = next_node

        return None

    def _select_next_node(self, current, unvisited, end):
        if not unvisited:
            return None

        epsilon = 1e-10
        probabilities = []

        for neighbor in unvisited:
            if self._is_edge_blocked(current, neighbor):
                continue

            pheromone_level = self.pheromone.get((current, neighbor), 1e-10)
            dist_to_goal = heuristic(neighbor, end)
            heuristic_value = 1.0 / (dist_to_goal + epsilon)
            scaled_heuristic = heuristic_value ** self.gamma

            prob = (pheromone_level ** self.alpha) * (scaled_heuristic ** self.beta)
            probabilities.append((neighbor, prob))

        if not probabilities:
            return None

        nodes, probs = zip(*probabilities)
        total_prob = sum(probs)

        if total_prob <= 0:
            return random.choice(nodes)

        normalized_probs = [p / total_prob for p in probs]
        return np.random.choice(nodes, p=normalized_probs)

    def _update_best_solution(self, paths, costs):
        if costs and min(costs) < self.best_cost:
            idx = np.argmin(costs)
            self.best_path = paths[idx]
            self.best_cost = costs[idx]
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1

    def _update_pheromones(self, paths, costs):
        for edge in self.pheromone:
            self.pheromone[edge] *= self.decay
        for path, cost in zip(paths, costs):
            deposit = 1 / max(0.1, cost)
            for u, v in zip(path[:-1], path[1:]):
                self.pheromone[(u, v)] += deposit
                if not isinstance(self.graph, nx.DiGraph):
                    self.pheromone[(v, u)] += deposit
        if self.best_path and self.best_cost < float('inf'):
            elite_deposit = self.elitist_factor / max(0.1, self.best_cost)
            for u, v in zip(self.best_path[:-1], self.best_path[1:]):
                self.pheromone[(u, v)] += elite_deposit
                if not isinstance(self.graph, nx.DiGraph):
                    self.pheromone[(v, u)] += elite_deposit

# ====================== Algorithm Evaluation ======================
def evaluate_algorithms(G, start, end):
    """Evaluate all pathfinding algorithms"""
    algorithms = {
        'Dijkstra': run_dijkstra,
        'A*': run_astar,
        'Greedy BFS': run_greedy_bfs,
        'Ant Colony': lambda G, s, e: OptimizedAntColony(G).run(s, e)
    }
    
    results = []
    
    for name, algo in algorithms.items():
        try:
            path, cost, exec_time = algo(G, start, end)
            
            actual_cost = float('inf')
            travel_time = float('inf')
            path_length = 0
            
            if path is not None:
                try:
                    actual_cost = 0
                    travel_seconds = 0
                    for u, v in zip(path[:-1], path[1:]):
                        actual_cost += G[u][v]['weight']
                        travel_seconds += G[u][v]['duration']
                    travel_time = travel_seconds / 60  # Convert to minutes
                    path_length = len(path)
                except:
                    actual_cost = float('inf')
                    travel_time = float('inf')
            
            results.append({
                'algorithm': name,
                'path_length': path_length,
                'cost': actual_cost if actual_cost != float('inf') else -1,
                'execution_time': exec_time,
                'path_found': path is not None and actual_cost != float('inf'),
                'path': path if path else [],
                'travel_time': travel_time if travel_time != float('inf') else -1
            })
        except Exception as e:
            app.logger.error(f"Error in {name} algorithm: {str(e)}")
            results.append({
                'algorithm': name,
                'path_length': 0,
                'cost': -1,
                'execution_time': 0,
                'path_found': False,
                'path': [],
                'travel_time': -1
            })
    
    return results

# ====================== API Endpoints ======================
@app.route('/get_road_data', methods=['GET'])
def get_road_data():
    """Return road data for the frontend"""
    try:
        df = load_and_prepare_data("lb_road_data.csv")
        road_data = df[['start_node', 'end_node', 'is_blocked']].to_dict('records')
        return jsonify(road_data)
    except Exception as e:
        app.logger.error(f"Error in get_road_data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/simulate_disaster', methods=['POST'])
def simulate_disaster():
    """Simulate a disaster and return affected edges"""
    try:
        data = request.json
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        disaster_type = data.get('disaster_type')
        origin_node = data.get('origin_node', '').strip()
        end_node = data.get('end_node', '').strip()  # Get end node for fire disaster handling
        
        if not disaster_type:
            return jsonify({'error': 'Missing disaster type'}), 400
            
        # Load data
        df = load_and_prepare_data("lb_road_data.csv")
        G = build_graph(df, cost_mode="hybrid")
        
        # Create disaster simulator
        simulator = DisasterSimulator(G, df)
        
        # Simulate the disaster
        if disaster_type == 'flood':
            effects = simulator.simulate_flood()
        elif disaster_type == 'fire':
            if not origin_node:
                return jsonify({'error': 'Fire disaster requires an origin node'}), 400
            # Pass end node to fire simulation for special handling
            effects = simulator.simulate_fire(origin_node, end_node)
        elif disaster_type == 'earthquake':
            effects = simulator.simulate_earthquake()
        else:
            return jsonify({'error': 'Invalid disaster type'}), 400
        
        # Get disaster info
        disaster_info = simulator.get_disaster_info()
        
        # Convert effects to a format suitable for the frontend
        affected_edges = []
        for (u, v), effect in effects.items():
            affected_edges.append({
                'start_node': u,
                'end_node': v,
                'effect_type': effect['type'],
                'completely_blocked': effect['completely_blocked'],
                'delay_seconds': effect['delay'],
                'is_end_node_access': effect.get('is_end_node_access', False)
            })
        
        return jsonify({
            'disaster_info': disaster_info,
            'affected_edges': affected_edges
        })
        
    except Exception as e:
        app.logger.error(f"Error in disaster simulation: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    """Run pathfinding simulation with given parameters"""
    try:
        data = request.json
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        start_node = data.get('start_node', '').strip()
        end_node = data.get('end_node', '').strip()
        blocked_edges = data.get('blocked_edges', [])
        disaster_edges = data.get('disaster_edges', [])
        
        if not start_node or not end_node:
            return jsonify({'error': 'Missing start or end node'}), 400

        # Load data
        df = load_and_prepare_data("lb_road_data.csv")
        
        # Apply manual blockages
        for edge in blocked_edges:
            if not isinstance(edge, dict):
                continue
                
            blocked_start = edge.get('start_node', '').strip()
            blocked_end = edge.get('end_node', '').strip()
            
            if not blocked_start or not blocked_end:
                continue

            # Mark edges as blocked in both directions
            for i, row in df.iterrows():
                row_start = row['start_node'].strip()
                row_end = row['end_node'].strip()
                
                if ((blocked_start == row_start and blocked_end == row_end) or
                    (blocked_start == row_end and blocked_end == row_start)):
                    df.at[i, 'is_blocked'] = 1
        
        # Apply disaster effects
        for edge in disaster_edges:
            if not isinstance(edge, dict):
                continue
                
            disaster_start = edge.get('start_node', '').strip()
            disaster_end = edge.get('end_node', '').strip()
            completely_blocked = edge.get('completely_blocked', False)
            delay_seconds = edge.get('delay_seconds', 0)
            
            if not disaster_start or not disaster_end:
                continue

            # Apply disaster effects to edges
            for i, row in df.iterrows():
                row_start = row['start_node'].strip()
                row_end = row['end_node'].strip()
                
                if ((disaster_start == row_start and disaster_end == row_end) or
                    (disaster_start == row_end and disaster_end == row_start)):
                    if completely_blocked:
                        df.at[i, 'is_blocked'] = 1
                    else:
                        df.at[i, 'duration_seconds'] += delay_seconds

        # Rebuild graph
        G = build_graph(df, cost_mode="hybrid")
        
        # Check if start and end nodes exist in graph
        if start_node not in G.nodes:
            return jsonify({'error': f'Start node {start_node} not found in road network'}), 400
        if end_node not in G.nodes:
            return jsonify({'error': f'End node {end_node} not found in road network'}), 400
            
        results = evaluate_algorithms(G, start_node, end_node)
        
        return jsonify({'results': results})

    except Exception as e:
        app.logger.error(f"Error in simulation: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    try:
        return send_file('index.html')
    except Exception as e:
        return f"Error loading index.html: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000)