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
    t1 = time.time()
    graph.get_k_important_stops_parallel()
    t2 = time.time()
    print(f"Executed time: {t2-t1}")
