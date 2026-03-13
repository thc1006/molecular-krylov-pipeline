"""Debug: find which configs have different connection counts between
sequential get_connections() and vectorized get_connections_vectorized_batch().

LiH: seq=2684, vec=2676 (diff=8)
N2 500 configs: seq=63011, vec=63010 (diff=1)
"""
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hamiltonians.molecular import create_lih_hamiltonian
from pipeline import FlowGuidedKrylovPipeline, PipelineConfig


def compare_connections(hamiltonian, configs):
    """Compare sequential and vectorized connections per config."""
    n = len(configs)

    # Sequential
    seq_counts = []
    seq_elements = {}
    for i in range(n):
        connected, elements = hamiltonian.get_connections(configs[i])
        seq_counts.append(len(connected))
        # Store as set of (config_hash, element_value) for comparison
        if len(connected) > 0:
            hashes = set()
            for j in range(len(connected)):
                cfg_tuple = tuple(connected[j].cpu().numpy().astype(int))
                hashes.add((cfg_tuple, round(elements[j].item(), 10)))
            seq_elements[i] = hashes

    # Vectorized
    all_connected, all_elements, batch_indices = \
        hamiltonian.get_connections_vectorized_batch(configs)

    vec_counts = [0] * n
    vec_elements = {}
    for i in range(n):
        vec_elements[i] = set()

    for j in range(len(all_connected)):
        idx = batch_indices[j].item()
        vec_counts[idx] += 1
        cfg_tuple = tuple(all_connected[j].cpu().numpy().astype(int))
        vec_elements[idx].add((cfg_tuple, round(all_elements[j].item(), 10)))

    # Find differences
    total_diff = 0
    for i in range(n):
        if seq_counts[i] != vec_counts[i]:
            diff = seq_counts[i] - vec_counts[i]
            total_diff += abs(diff)
            print(f"\nConfig {i}: seq={seq_counts[i]}, vec={vec_counts[i]}, diff={diff}")
            print(f"  Config: {configs[i].cpu().numpy().astype(int)}")

            # Find missing/extra connections
            if i in seq_elements and i in vec_elements:
                only_seq = seq_elements[i] - vec_elements[i]
                only_vec = vec_elements[i] - seq_elements[i]
                if only_seq:
                    print(f"  Only in sequential ({len(only_seq)}):")
                    for cfg, val in list(only_seq)[:5]:
                        print(f"    config={list(cfg)}, H={val}")
                if only_vec:
                    print(f"  Only in vectorized ({len(only_vec)}):")
                    for cfg, val in list(only_vec)[:5]:
                        print(f"    config={list(cfg)}, H={val}")

    print(f"\nTotal configs with differences: {sum(1 for i in range(n) if seq_counts[i] != vec_counts[i])}")
    print(f"Total connection difference: {total_diff}")
    return total_diff


if __name__ == "__main__":
    lih = create_lih_hamiltonian()
    pipe_config = PipelineConfig(skip_nf_training=True, device="cpu")
    pipeline = FlowGuidedKrylovPipeline(lih, config=pipe_config)
    configs = pipeline._generate_essential_configs()
    print(f"LiH: {len(configs)} essential configs")
    compare_connections(lih, configs)
