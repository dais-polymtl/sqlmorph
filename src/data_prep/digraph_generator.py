import itertools
import networkx as nx
import os
from multiprocessing import Pool, cpu_count
from functools import partial
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash

def get_direction_options(unordered_pairs):
    return [[(a, b), (b, a), None] for a, b in unordered_pairs]

def batched(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def process_batch(batch, nodes):
    local_graphs = []
    for combination in batch:
        edges = [edge for edge in combination if edge is not None]
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)

        if not nx.is_weakly_connected(G):
            continue

        local_graphs.append(G)
    return local_graphs

def filter_unique_by_hash(graphs):
    seen_hashes = set()
    unique_graphs = []
    for g in graphs:
        h = weisfeiler_lehman_graph_hash(g)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        unique_graphs.append(g)
    return unique_graphs

def generate_graphs_parallel_lazy(node_count, batch_size=10000):
    nodes = [chr(97 + i) for i in range(node_count)]
    unordered_pairs = list(itertools.combinations(nodes, 2))
    direction_options = get_direction_options(unordered_pairs)

    all_graphs = []
    combo_iterator = itertools.product(*direction_options)
    with Pool(cpu_count()) as pool:
        for i, batch in enumerate(batched(combo_iterator, batch_size)):
            print(f"Processing batch {i + 1}...")
            results = pool.map(partial(process_batch, nodes=nodes), [batch])
            flat_graphs = [g for sublist in results for g in sublist]
            all_graphs.extend(flat_graphs)

    print(f"Deduplicating {len(all_graphs)} graphs...")
    return filter_unique_by_hash(all_graphs)

def is_undirected_acyclic(g):
    return g.to_undirected().number_of_edges() < g.number_of_nodes()

def format_graph(graph):
    return ",".join([f"{u}->{v}" for u, v in graph.edges()])

def write_all_graphs_to_file(file_path, all_graphs_dict):
    with open(file_path, "w") as f:
        for node_count in sorted(all_graphs_dict.keys()):
            graphs = all_graphs_dict[node_count]
            acyclic_lines = []
            cyclic_lines = []

            for g in graphs:
                line = format_graph(g)
                if is_undirected_acyclic(g):
                    acyclic_lines.append(line)
                else:
                    cyclic_lines.append(line)

            print(f"\nWriting graphs for {node_count} nodes")
            print(f"Cyclic: {len(cyclic_lines)}")
            print(f"Acyclic: {len(acyclic_lines)}")

            f.write(f"Number of nodes: {node_count}\n")
            f.write("cyclic\n")
            f.write("\n".join(cyclic_lines) + "\n")
            f.write("acyclic\n")
            f.write("\n".join(acyclic_lines) + "\n\n")

# === MAIN ===
if __name__ == "__main__":
    os.makedirs("output_graphs", exist_ok=True)
    output_path = "output_graphs/all_graphs.txt"
    all_graphs = {}

    for n in range(3, 7):
        print(f"\n=== Generating graphs for {n} nodes ===")
        graphs = generate_graphs_parallel_lazy(n)
        all_graphs[n] = graphs
        write_all_graphs_to_file(output_path, all_graphs)
