# -*- coding: utf-8 -*-
"""
Print metrics file created by the training scripts.
"""
import argparse, json

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics_path", default="models/metrics.json")
    args = p.parse_args()
    with open(args.metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    print("Metrics:")
    for k, v in metrics.items():
        print(f"- {k}: {v}")

if __name__ == "__main__":
    main()