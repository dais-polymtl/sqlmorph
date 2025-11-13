import pickle
from pathlib import Path
import random
from collections import defaultdict
import numpy as np

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def retrieve_jqgs(folder_path):
    """
    Retrieve all subgraphs from the pickle files in the folder.

    Args:
        folder_path (str): The path to the folder containing the .pkl files.

    Returns:
        list: A list of all subgraphs from the folder.
    """
    folder = Path(folder_path)
    all_subgraphs = []

    for file_name in folder.glob("jq_graphs_*_tables.pkl"):
        try:
            with file_name.open("rb") as f:
                join_patterns = pickle.load(f)
                if isinstance(join_patterns, list):
                    all_subgraphs.extend(join_patterns)
        except (ValueError, KeyError, pickle.UnpicklingError) as e:
            print(f"Skipping file {file_name.name}: {e}")
    return all_subgraphs


def compute_node_distribution_per_db(jqgs):
    db_node_counts = defaultdict(lambda: defaultdict(int))
    for jqg in jqgs:
        db = jqg["db_id"]
        n = jqg["jq_graph"].number_of_nodes()
        db_node_counts[db][n] += 1

    db_node_dist = {}
    for db, counts in db_node_counts.items():
        total = sum(counts.values())
        db_node_dist[db] = {k: v / total for k, v in counts.items()}
    return db_node_dist


def cluster_db_ids_by_node_dist(db_node_dist, n_clusters=2):
    # 1. Sort node labels to ensure consistent vector dimension order
    all_nodes = sorted({n for dist in db_node_dist.values() for n in dist})

    # 2. Sort database keys to ensure fixed order
    db_keys = sorted(db_node_dist.keys())

    # 3. Build feature matrix in consistent order
    db_features = []
    for db in db_keys:
        dist = db_node_dist[db]
        vector = [dist.get(n, 0.0) for n in all_nodes]
        db_features.append(vector)

    X = np.array(db_features)

    # 4. Run KMeans with fixed seed *and* fixed n_init
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X)

    # 5. Group db_ids by cluster label
    clustered = defaultdict(list)
    for label, db in zip(kmeans.labels_, db_keys):
        clustered[label].append(db)

    return clustered


def split_cluster_jqgs(jqgs, cluster_db_ids, test_size=100, dev_ratio=0.2, seed=42):
    # Get queries for the cluster
    cluster_jqgs = [q for q in jqgs if q["db_id"] in cluster_db_ids]

    # Group queries by db_id
    db_to_queries = defaultdict(list)
    for q in cluster_jqgs:
        db_to_queries[q["db_id"]].append(q)

    # Get sorted list of db_ids for stability
    db_ids = sorted(db_to_queries.keys())

    # Set seed for reproducibility just before shuffling
    random.seed(seed)
    random.shuffle(db_ids)

    # Utility to accumulate DBs until reaching target queries count
    def accumulate_until(target_queries, db_list):
        selected_dbs = []
        total_queries = 0
        for db in db_list:
            selected_dbs.append(db)
            total_queries += len(db_to_queries[db])
            if total_queries >= target_queries:
                break
        remaining_dbs = [db for db in db_list if db not in selected_dbs]
        return selected_dbs, remaining_dbs

    # 1. Select test DBs (~100 queries)
    test_dbs, remaining_dbs = accumulate_until(test_size, db_ids)

    # 2. Compute dev size based on remaining
    remaining_jqgs_count = sum(len(db_to_queries[db]) for db in remaining_dbs)
    dev_size = int(remaining_jqgs_count * dev_ratio)

    # 3. Select dev DBs
    dev_dbs, train_dbs = accumulate_until(dev_size, remaining_dbs)

    # 4. Build splits
    test_set = [q for db in test_dbs for q in db_to_queries[db]]
    dev_set = [q for db in dev_dbs for q in db_to_queries[db]]
    train_set = [q for db in train_dbs for q in db_to_queries[db]]

    # 5. Function to compute node distribution
    def node_dist(jqgs):
        counts = defaultdict(int)
        for q in jqgs:
            n = q["jq_graph"].number_of_nodes()
            counts[n] += 1
        total = len(jqgs)
        return {k: f"{v/total*100:.2f}%" for k, v in counts.items()}

    # Print info if the split matches desired sizes
    # if len(train_set) == 345 and len(dev_set) == 100 and len(train_dbs) == 20 and len(dev_dbs) == 12:
    print(
        f"Train queries: {len(train_set)}, DBs: {len(train_dbs)}, Node dist: {node_dist(train_set)}"
    )
    print(
        f"Dev queries: {len(dev_set)}, DBs: {len(dev_dbs)}, Node dist: {node_dist(dev_set)}"
    )
    print(
        f"Test queries: {len(test_set)}, DBs: {len(test_dbs)}, Node dist: {node_dist(test_set)}"
    )

    return train_set, dev_set, test_set


def visualize_clusters(db_node_dist, clustered, output_path, method="pca"):
    # Prepare features
    all_nodes = sorted({n for dist in db_node_dist.values() for n in dist})
    db_keys = sorted(db_node_dist.keys())
    db_features = [[db_node_dist[db].get(n, 0.0) for n in all_nodes] for db in db_keys]
    X = np.array(db_features)

    # PCA for 2D projection
    reducer = PCA(n_components=2)
    X_2d = reducer.fit_transform(X)

    # Build db → cluster label map
    db_to_label = {db: label for label, dbs in clustered.items() for db in dbs}
    labels = [db_to_label[db] for db in db_keys]

    # Color map with fixed number of clusters
    n_clusters = len(clustered)
    cmap = plt.get_cmap("tab10" if n_clusters <= 10 else "tab20")

    # Plot
    plt.figure(figsize=(8, 6))
    for i in range(n_clusters):
        cluster_indices = [j for j, lbl in enumerate(labels) if lbl == i]
        plt.scatter(
            X_2d[cluster_indices, 0],
            X_2d[cluster_indices, 1],
            color=cmap(i),
            label=f"Cluster {i}",
            s=80,
        )

    plt.title("PCA Projection of DB Clusters")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
