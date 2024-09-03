from Graph import *
import csv
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
import time
import itertools
import multiprocessing
import psutil
import numpy as np

def square(x):
    time.sleep(1)  # Simulate a time-consuming task
    res = []
    for i in x:
        print(i)
        res.append(i*i)
    return res


if __name__ == "__main__":
    graph = Graph()
    graph.build_bus_graph()
    # graph.save_graph()
    # graph.load_graph()
    # print(len(list(graph.G.nodes)), len(list(graph.G.edges)))
    t1 = time.time()
    # graph.calculate_stress_centrality_parallel(3)
    t2 = time.time()
    print(f"Executed time: {t2-t1}")

    # data = [1, 2, 3, 4]
    # d = np.array_split(data, 4)
    # print(d)
    # with multiprocessing.Pool(4) as pool:
    #     results = list(pool.imap(square, d))

    #     # Process results as they are available
    #     # for result in results:
    #     print("Result with imap:", results)

# 3:57:32
