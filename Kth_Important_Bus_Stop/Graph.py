import csv
import os
import heapq
import json
import pickle
import multiprocessing
import os
import numpy as np
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
from pyproj import Transformer
import time

class Graph:
    transformer = Transformer.from_crs("EPSG:3405", "EPSG:4326")
    def __init__(self):
        self.G = nx.DiGraph()
        self.stop_id_to_node = defaultdict(set)
        self.node_ids = set()

    def build_bus_graph(self):
        print("Starting to build the bus graph...")
        csv_files = ['type12.csv', 'type34.csv']
        for csv_file_path in csv_files:
            print(f"Processing file: {csv_file_path}")
            
            with open(csv_file_path, mode='r') as file:
                total_rows = sum(1 for row in file) 

            with open(csv_file_path, mode='r') as file:
                reader = csv.reader(file)
                
                for row in tqdm(reader, total=total_rows, desc=f"Building Graph from {csv_file_path}", unit="row"):
                    stop_id1, route_id1, var_id1 = map(int, row[:3])
                    stop_id2, route_id2, var_id2 = map(int, row[4:7])
                    timestamp1, timestamp2, time_diff = map(lambda x: int(float(x)), (row[3], row[7], row[8]))
                    latx1, lngy1, latx2, lngy2 = map(float, row[9:13])
                    node_type1, node_type2, node_pos1,node_pos2,edge_pos,edge_type = map(int, row[15:21])
                    
                    node1 = (stop_id1, route_id1, var_id1, timestamp1, node_type1, latx1, lngy1)
                    node2 = (stop_id2, route_id2, var_id2, timestamp2, node_type2, latx2, lngy2)
                    
                    self.stop_id_to_node[stop_id1].add(node1)
                    self.stop_id_to_node[stop_id2].add(node2)

                    self.node_ids.add(stop_id1)
                    self.node_ids.add(stop_id2)

                    num_transfers = 0 if edge_type in {1, 2} else 1
                    weight = (num_transfers, time_diff)
                    self.G.add_edge(node1, node2, weight=weight)
                    
        print(f"Graph building completed. {len(self.G.nodes)} nodes and {len(self.G.edges)} edges have been added.")

    def dijkstra(self, source_id):
        target_node = {}
        source_node = {}
        cost = {node: (float('inf'), float('inf')) for node in self.G.nodes} 
        predecessors = {node: None for node in self.G.nodes}
        priority_queue = []
        
        # Initialize cost for all source nodes
        for source in self.stop_id_to_node[source_id]:
            cost[source] = (0, 0)  # (transfers, time)
            heapq.heappush(priority_queue, (0, 0, source, source))  # (transfers, time, source, target)

        while priority_queue:
            current_transfers, current_time, source, u = heapq.heappop(priority_queue)
            
            if (current_transfers, current_time) > cost[u]:
                continue
            if u[0] != source_id:
                source_node[u] = source
                node_id = u[0]
                if node_id not in target_node:
                    target_node[node_id] = u
                else:
                    target = target_node[node_id]
                    if cost[target] == cost[u]:
                        current = (source_node[u][3], u[3])
                        best = (source_node[target][3], target[3])
                        if current < best:
                            target_node[node_id] = u
            
            for v, edge_data in self.G[u].items():
                new_transfers = current_transfers + edge_data['weight'][0]
                new_time = current_time + edge_data['weight'][1]
                
                if (new_transfers, new_time) < cost[v]:
                    cost[v] = (new_transfers, new_time)
                    predecessors[v] = u
                    heapq.heappush(priority_queue, (new_transfers, new_time, source, v))
        
        return predecessors, target_node  

    def get_k_important_stops(self, k=10):
        print("Calculating vertex importance...")
        count = {v: 0 for v in self.node_ids}
        for source_id in tqdm(self.node_ids, desc="Processing nodes", unit='node'):
            predecessors, target_node = self.dijkstra(source_id)
            for target_id in self.node_ids:
                print(source_id, target_id)
                if source_id == target_id or target_id not in target_node:
                    continue
                current_node = target_node[target_id]
                count[current_node[0]] += 1
                
                while current_node[0] != source_id:
                    prev_node = current_node
                    current_node = predecessors[current_node]
                    if current_node[0] != prev_node[0]:
                        count[current_node[0]] += 1
        sorted_count = dict(sorted(count.items(), key=lambda item: item[1], reverse=True))
        top_stops = list(sorted_count.keys())[:k]
        print(f"Top {k} most important bus stops found.")
        with open('important_bus_stops.json', 'w') as file:
            json.dump(sorted_count, file, indent=4)
        return top_stops
    
    def get_k_important_stops_parallel(self, num_processes=10, k=10):
        # Set environment variable to control Intel's thread affinity
        os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
        self.node_ids = list(self.node_ids)
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.imap(self.process_source_chunk, np.array_split(self.node_ids, num_processes))

        count = defaultdict(int)
        for chunk_counts in results:
            for node, node_count in chunk_counts.items():
                count[node] += node_count

        sorted_count = dict(sorted(count.items(), key=lambda item: item[1], reverse=True))
        top_stops = list(sorted_count.keys())[:k]
        print(f"Top {k} most important bus stops found.")
        with open('important_bus_stops.json', 'w') as file:
            json.dump(sorted_count, file, indent=4)
        return top_stops

    def process_source_chunk(self, source_chunk):
        chunk_count = defaultdict(int)
        for source_id in source_chunk:
            predecessors, target_node = self.dijkstra(source_id)
            
            for target_id in self.node_ids:
                if source_id == target_id or target_id not in target_node:
                    continue
                
                current_node = target_node[target_id]
                chunk_count[current_node[0]] += 1
                
                while current_node[0] != source_id:
                    prev_node = current_node
                    current_node = predecessors[current_node]
                    if current_node[0] != prev_node[0]:
                        chunk_count[current_node[0]] += 1
        return chunk_count