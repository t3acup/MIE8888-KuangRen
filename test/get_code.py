import os
import shutil
from collections import defaultdict
def get_code(root_dir="code", output_dir="separated_by_language"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for cluster_name in os.listdir(root_dir):
        cluster_path = os.path.join(root_dir, cluster_name)
        if not os.path.isdir(cluster_path):
            continue

        for language in os.listdir(cluster_path):
            language_path = os.path.join(cluster_path, language)
            if not os.path.isdir(language_path):
                continue

            # Destination: output_dir/language/cluster_name/
            target_dir = os.path.join(output_dir, language, cluster_name)
            os.makedirs(target_dir, exist_ok=True)

            for file_name in os.listdir(language_path):
                src_file = os.path.join(language_path, file_name)
                dst_file = os.path.join(target_dir, file_name)
                try:
                    shutil.copy2(src_file, dst_file)
                except Exception as e:
                    print(f"Error copying {src_file} → {dst_file}: {e}")

    print(f"Code has been separated into '{output_dir}' by language.")

def count_code_samples(base_dir):
    """
    Counts the number of code samples in a folder structured as:
    base_dir/
        algorithm/
            language/
                file1
                file2
                ...
    Returns a nested dictionary {algorithm: {language: count}} and prints a table.
    """
    counts = defaultdict(lambda: defaultdict(int))

    for algorithm in sorted(os.listdir(base_dir)):
        algo_path = os.path.join(base_dir, algorithm)
        if not os.path.isdir(algo_path):
            continue

        for language in sorted(os.listdir(algo_path)):
            lang_path = os.path.join(algo_path, language)
            if not os.path.isdir(lang_path):
                continue

            # Count only files in the folder
            file_count = sum(
                1 for f in os.listdir(lang_path)
                if os.path.isfile(os.path.join(lang_path, f))
            )
            counts[algorithm][language] = file_count

    # Print table
    print(f"{'Algorithm':<15} {'Language':<10} {'Count':<6}")
    print("-" * 35)
    for algo in counts:
        for lang, count in counts[algo].items():
            print(f"{algo:<15} {lang:<10} {count:<6}")

    return counts


# Example usage
dataset_path = "code"  # your dataset root folder
counts = count_code_samples(dataset_path)
