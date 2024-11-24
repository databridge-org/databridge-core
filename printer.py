import os
import argparse
from typing import Set
from datetime import datetime


class DirectoryTree:
    """Class to generate and display directory tree structure"""

    def __init__(self):
        self.tree = []
        self.files_found = 0

    def add_path(self, path: str, is_file: bool = True):
        """Add a path to the tree"""
        parts = path.split(os.sep)
        if is_file:
            self.files_found += 1

        # Build indentation for each level
        for i in range(len(parts)):
            prefix = "    " * i
            if i == len(parts) - 1:
                marker = "📄 " if is_file else "📁 "
            else:
                marker = "📁 "
            self.tree.append(f"{prefix}{marker}{parts[i]}")

    def get_tree_string(self) -> str:
        """Get the tree structure as a string"""
        output = ["Directory Structure:", "=" * 50]
        output.extend(self.tree)
        output.extend([
            "=" * 50,
            f"Total files found: {self.files_found}",
            ""
        ])
        return "\n".join(output)

    def display(self):
        """Display the tree structure"""
        print("\n" + self.get_tree_string())


def should_ignore_directory(dirname: str) -> bool:
    """Check if directory should be ignored."""
    ignore_dirs = {
        'venv',
        'env',
        '.venv',
        'virtualenv',
        '__pycache__',
        '.pytest_cache',
        '.mypy_cache',
        '.tox',
        '.git',
        'build',
        'dist',
        'node_modules'
    }
    return dirname in ignore_dirs


def get_target_directories(mode: str, root_dir: str) -> Set[str]:
    """Get target directories based on mode"""
    if mode == "all":
        return {root_dir}

    mode_dirs = {
        "core": ["core"],
        "sdk": ["sdks"],
        "test": ["core/tests", "sdks/python/tests"]
    }

    return {os.path.join(root_dir, d) for d in mode_dirs.get(mode, [])}


def aggregate_python_files(
    root_dir: str,
    output_file: str,
    script_name: str,
    mode: str = "all"
) -> None:
    """
    Recursively search through directories and aggregate Python files.

    Args:
        root_dir: Root directory to start search
        output_file: Output file name
        script_name: Name of this script to ignore
        mode: One of "all", "core", "sdk", or "test"
    """
    root_dir = os.path.abspath(root_dir)
    tree = DirectoryTree()
    target_dirs = get_target_directories(mode, root_dir)

    print(f"\nProcessing Python files in {mode} mode...")
    print(f"Target directories: {', '.join(target_dirs)}")

    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Write header information
        header = f"""DataBridge Code Aggregation
{'=' * 80}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Mode: {mode}
Root Directory: {root_dir}

"""
        outfile.write(header)

        for target_dir in target_dirs:
            if not os.path.exists(target_dir):
                print(f"Warning: Directory not found - {target_dir}")
                continue

            tree.add_path(os.path.relpath(target_dir, root_dir), is_file=False)

            for dirpath, dirnames, filenames in os.walk(target_dir, topdown=True):
                # Skip ignored directories
                dirnames[:] = [
                    d for d in dirnames if not should_ignore_directory(d)
                ]

                # Add directories to tree
                for d in dirnames:
                    rel_path = os.path.relpath(
                        os.path.join(dirpath, d), root_dir
                    )
                    tree.add_path(rel_path, is_file=False)

                # Process Python files
                python_files = [
                    f for f in filenames
                    if f.endswith('.py')
                    and f != '__init__.py'
                    and f != script_name
                    and f != output_file
                ]

                for py_file in python_files:
                    file_path = os.path.join(dirpath, py_file)
                    rel_path = os.path.relpath(file_path, root_dir)

                    # Add file to tree
                    tree.add_path(rel_path)

        # Write tree structure to file
        outfile.write(tree.get_tree_string())
        outfile.write("\n" + "=" * 80 + "\n\n")
        outfile.write("FILE CONTENTS\n")
        outfile.write("=" * 80 + "\n\n")

        # Now write the actual file contents
        for target_dir in target_dirs:
            if not os.path.exists(target_dir):
                continue

            for dirpath, dirnames, filenames in os.walk(target_dir, topdown=True):
                dirnames[:] = [
                    d for d in dirnames if not should_ignore_directory(d)
                ]

                python_files = [
                    f for f in filenames
                    if f.endswith('.py')
                    and f != '__init__.py'
                    and f != script_name
                    and f != output_file
                ]

                for py_file in python_files:
                    file_path = os.path.join(dirpath, py_file)
                    rel_path = os.path.relpath(file_path, root_dir)

                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            content = infile.read()

                        # Write file content with header
                        outfile.write(f"\n{rel_path}\n")
                        outfile.write("=" * len(rel_path) + "\n")
                        outfile.write(content)
                        outfile.write("\n\n" + "=" * 80 + "\n")

                    except Exception as e:
                        outfile.write(
                            f"Error reading {rel_path}: {str(e)}\n\n"
                        )

    # Display tree structure in console
    tree.display()


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate Python files with directory structure"
    )
    parser.add_argument(
        '--mode',
        choices=['all', 'core', 'sdk', 'test'],
        default='all',
        help='Which directories to process'
    )
    parser.add_argument(
        '--output',
        default='aggregated_code.txt',
        help='Output file name'
    )
    args = parser.parse_args()

    script_name = os.path.basename(__file__)
    current_dir = os.getcwd()

    print("\nStarting code aggregation:")
    print(f"Mode: {args.mode}")
    print(f"Output: {args.output}")
    print(f"Root directory: {current_dir}")

    aggregate_python_files(
        root_dir=current_dir,
        output_file=args.output,
        script_name=script_name,
        mode=args.mode
    )

    print(f"\nFinished! Results written to: {args.output}")


if __name__ == "__main__":
    main()
