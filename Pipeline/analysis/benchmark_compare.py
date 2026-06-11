"""
benchmark_compare.py — Unified comparison of rPPG + FER benchmark results.

Pattern: [Utility / Reporter] — aggregates Docker container and host fusion
CSVs into formatted comparison tables for the report; read-only, no side effects.

Reads CSV outputs from Docker containers and host fusion script,
produces formatted comparison tables for the report.

Usage:
    python benchmark_compare.py [--docker-dir ../Docker/output] [--host-dir logs]
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def load_rppg_results(docker_dir):
    """Load rPPG benchmark CSVs from Docker output directories."""
    results = []
    rppg_tools = ["pyvhr", "rppg_toolbox"]

    for tool in rppg_tools:
        tool_dir = os.path.join(docker_dir, tool)
        if not os.path.isdir(tool_dir):
            continue

        for csv_file in Path(tool_dir).glob("*_results.csv"):
            try:
                df = pd.read_csv(csv_file)
                if "bpm" in df.columns:
                    for _, row in df.iterrows():
                        results.append(dict(row))
            except Exception as e:
                print(f"  Warning: Could not read {csv_file}: {e}")

    return results


def load_host_rppg(host_dir):
    """Load rPPG results from host fusion script (_rppg.csv files)."""
    results = []

    for rppg_file in Path(host_dir).rglob("*_rppg.csv"):
        try:
            df = pd.read_csv(rppg_file)
            for _, row in df.iterrows():
                r = dict(row)
                r["tool"] = "standalone"
                results.append(r)
        except Exception as e:
            print(f"  Warning: Could not read {rppg_file}: {e}")

    return results


def load_fer_results(docker_dir):
    """Load FER benchmark CSVs from Docker output directories."""
    results = {}
    fer_tools = ["deepface", "pyfeat", "mp_hs"]

    for tool in fer_tools:
        tool_dir = os.path.join(docker_dir, tool)
        if not os.path.isdir(tool_dir):
            continue

        meta_files = list(Path(tool_dir).glob("*_meta.json"))
        for meta_file in meta_files:
            try:
                with open(meta_file) as f:
                    meta = json.load(f)
                results[tool] = meta
            except Exception as e:
                print(f"  Warning: Could not read {meta_file}: {e}")

    return results


def print_rppg_table(docker_results, host_results):
    """Print formatted rPPG comparison table."""
    all_results = docker_results + host_results
    if not all_results:
        print("\n  No rPPG results found.\n")
        return

    df = pd.DataFrame(all_results)
    if "tool" not in df.columns or "algorithm" not in df.columns:
        print("\n  rPPG CSVs missing 'tool' or 'algorithm' columns.\n")
        return

    print("\n" + "=" * 80)
    print("rPPG ALGORITHM COMPARISON")
    print("=" * 80)
    print(f"{'Tool':<15} {'Algorithm':<10} {'Mean BPM':>9} {'Std':>6} "
          f"{'Plaus%':>7} {'Windows':>8}")
    print("-" * 80)

    for (tool, algo), group in df.groupby(["tool", "algorithm"]):
        bpms = group["bpm"].values
        plausible = group["bpm_plausible"].sum() if "bpm_plausible" in group else 0
        n = len(bpms)
        plaus_pct = f"{plausible/n*100:.0f}%" if n > 0 else "N/A"

        print(f"{str(tool):<15} {str(algo):<10} {np.mean(bpms):>9.1f} "
              f"{np.std(bpms):>6.1f} {plaus_pct:>7} {n:>8}")

    print("=" * 80)

    # Cross-tool agreement: POS from different tools on same video
    if "video" in df.columns:
        pos_df = df[df["algorithm"].str.upper() == "POS"]
        if len(pos_df["tool"].unique()) > 1:
            print("\nPOS cross-tool agreement:")
            for video, vgroup in pos_df.groupby("video"):
                tools = vgroup.groupby("tool")["bpm"].mean()
                if len(tools) > 1:
                    pairs = list(tools.items())
                    for i in range(len(pairs)):
                        for j in range(i + 1, len(pairs)):
                            diff = abs(pairs[i][1] - pairs[j][1])
                            print(f"  {video}: {pairs[i][0]} vs {pairs[j][0]} "
                                  f"= {diff:.1f} BPM difference")


def print_fer_table(fer_results):
    """Print formatted FER comparison table."""
    if not fer_results:
        print("\n  No FER results found.\n")
        return

    print("\n" + "=" * 80)
    print("FER TOOL COMPARISON")
    print("=" * 80)
    print(f"{'Tool':<15} {'Env':<8} {'Mean Lat':>9} {'FPS':>7} {'GPU':>5}")
    print("-" * 80)

    for tool, meta in fer_results.items():
        env = "Docker"
        lat = meta.get("avg_latency_ms", "N/A")
        fps = meta.get("fps", "N/A")
        gpu = meta.get("gpu_available", "N/A")

        lat_str = f"{lat:.1f}ms" if isinstance(lat, (int, float)) else str(lat)
        fps_str = f"{fps:.1f}" if isinstance(fps, (int, float)) else str(fps)
        gpu_str = "Yes" if gpu else "No"

        print(f"{tool:<15} {env:<8} {lat_str:>9} {fps_str:>7} {gpu_str:>5}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Compare rPPG + FER benchmark results")
    parser.add_argument("--docker-dir", default="../Docker/output",
                        help="Docker output directory")
    parser.add_argument("--host-dir", default="logs",
                        help="Host logs directory")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    docker_dir = os.path.join(script_dir, args.docker_dir)
    host_dir = os.path.join(script_dir, args.host_dir)

    print("La Facade Fissuree — Benchmark Comparison")
    print(f"Docker results: {docker_dir}")
    print(f"Host results:   {host_dir}")

    # rPPG
    docker_rppg = load_rppg_results(docker_dir)
    host_rppg = load_host_rppg(host_dir)
    print(f"\nLoaded {len(docker_rppg)} Docker rPPG rows, "
          f"{len(host_rppg)} host rPPG rows")
    print_rppg_table(docker_rppg, host_rppg)

    # FER
    fer_results = load_fer_results(docker_dir)
    print(f"\nLoaded FER meta from {len(fer_results)} tools")
    print_fer_table(fer_results)


if __name__ == "__main__":
    main()
