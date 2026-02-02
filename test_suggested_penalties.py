"""
Test script to compare current vs suggested ACCURACY_PENALTIES.

This temporarily modifies the penalties in multi_model_dev.py and runs experiments.
"""

import os
import subprocess
import pickle
import shutil

# Suggested penalties (more reasonable diminishing returns)
SUGGESTED_PENALTIES = """
ACCURACY_PENALTIES = {
    "ResNet50":  {"layer1": -1.0, "layer2": -2.5, "layer3": -3.5, "final": -4.0},
    "ResNet101": {"layer1": -1.0, "layer2": -3.0, "layer3": -5.0, "final": -6.0},
    "ResNet152": {"layer1": -1.0, "layer2": -4.0, "layer3": -6.5, "final": -8.0},
}
"""

ORIGINAL_PENALTIES = """
ACCURACY_PENALTIES = {
    "ResNet50":  {"layer1": -0.2, "layer2": -1.5, "layer3": -3.5, "final": -4.5},
    "ResNet101": {"layer1": -0.2, "layer2": -2, "layer3": -7, "final": -8},
    "ResNet152": {"layer1": -0.2, "layer2": -3, "layer3": -10, "final": -11},
}
"""

def backup_file(filepath):
    """Create backup of file."""
    backup = filepath + ".backup"
    shutil.copy2(filepath, backup)
    print(f"Backed up {filepath} to {backup}")
    return backup

def restore_file(filepath, backup):
    """Restore file from backup."""
    shutil.copy2(backup, filepath)
    os.remove(backup)
    print(f"Restored {filepath} from backup")

def modify_penalties(filepath, new_penalties):
    """Replace ACCURACY_PENALTIES in the file."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Find and replace the ACCURACY_PENALTIES block
    start = content.find("ACCURACY_PENALTIES = {")
    if start == -1:
        raise ValueError("Could not find ACCURACY_PENALTIES in file")

    # Find the end of the dict (next line that starts with "}")
    end = content.find("\n}", start) + 2

    # Replace
    new_content = content[:start] + new_penalties.strip() + content[end:]

    with open(filepath, 'w') as f:
        f.write(new_content)

    print(f"Modified ACCURACY_PENALTIES in {filepath}")

def run_experiment(W_ACC_N, tag):
    """Run experiment with given W_ACC_N."""
    env = os.environ.copy()
    env["W_ACC_N"] = str(W_ACC_N)
    env["W_SLO_N"] = "0.3"
    env["SLO_PENALTY_TARGET_N"] = "0.03"

    cmd = ["python", "run_multi_model_intensity_dev.py", "--run-seconds", "15"]

    print(f"\n{'='*80}")
    print(f"Running: {tag}, W_ACC_N={W_ACC_N}")
    print(f"{'='*80}")

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    # Parse result from output
    for line in result.stdout.split('\n'):
        if '[RESULT]' in line:
            print(line)
            # Extract metrics
            if 'p95=' in line:
                p95 = line.split('p95=')[1].split(' ms')[0]
                violate = line.split('violate_ratio=')[1].split(',')[0]
                completed = line.split('completed=')[1].split(',')[0]
                avg_exit = line.split('AvgExit    Completed')[1].strip().split()[2] if 'AvgExit' in line else "N/A"

                return {
                    'tag': tag,
                    'W_ACC_N': W_ACC_N,
                    'p95_ms': float(p95),
                    'violate_ratio': float(violate),
                    'completed': int(completed),
                }

    return None

def main():
    script_path = "multi_model_dev.py"

    # Backup original
    backup = backup_file(script_path)

    results = []

    try:
        # Test 1: Current penalties with W_ACC_N=1, 3, 5
        print("\n" + "="*80)
        print("TEST 1: CURRENT PENALTIES")
        print("="*80)

        # Already using current penalties, no need to modify
        for W_ACC_N in [1, 3]:
            result = run_experiment(W_ACC_N, f"Current_W{W_ACC_N}")
            if result:
                results.append(result)

        # Test 2: Suggested penalties with W_ACC_N=1, 2, 3
        print("\n" + "="*80)
        print("TEST 2: SUGGESTED PENALTIES (with diminishing returns)")
        print("="*80)

        modify_penalties(script_path, SUGGESTED_PENALTIES)

        for W_ACC_N in [1, 2, 3]:
            result = run_experiment(W_ACC_N, f"Suggested_W{W_ACC_N}")
            if result:
                results.append(result)

    finally:
        # Restore original
        restore_file(script_path, backup)

    # Print comparison
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    print(f"{'Config':<20} {'W_ACC_N':<10} {'Avg Exit':<12} {'P95 (ms)':<12} {'SLO Viol%':<12} {'Throughput'}")
    print("-"*80)

    for r in results:
        print(f"{r['tag']:<20} {r['W_ACC_N']:<10} {'TBD':<12} {r['p95_ms']:<12.2f} "
              f"{r['violate_ratio']*100:<12.2f} {r['completed']}")

    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print("""
If suggested penalties work better:
- Should see reasonable exit distribution with W_ACC_N=1
- Should need less aggressive W_ACC_N values
- Should have more predictable behavior

If current penalties work better:
- Keep current approach
- Document W_ACC_N as tunable parameter
- Note that penalties are empirically tuned
""")

if __name__ == "__main__":
    main()
