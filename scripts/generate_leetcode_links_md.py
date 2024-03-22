import re
from pathlib import Path
from typing import List, Tuple

# Compile the regular expression once and use it globally
leetcode_url_pattern = re.compile(r"https://leetcode\.\w+/problems/[\w-]+/")


def extract_leetcode_links(file_path: Path) -> Tuple[str, str]:
    """
    Extracts the first LeetCode URL from the initial docstring of a Python file and
    generates a tuple containing the URL and a descriptive name based on the file naming
    convention.

    Args:
        file_path (Path): The path to the Python file to be scanned.

    Returns:
        Tuple[str, str]: A tuple containing the LeetCode URL and a descriptive name
        derived from the filename, or an empty tuple if no URL is found.
    """
    # Extract problem number and title from filename
    file_name_parts = file_path.stem.split("_")  # p, <problem number>, <problem title parts>...
    if len(file_name_parts) < 3 or file_name_parts[0] != "p":
        return ("", "")  # Return empty if file doesn't match the naming convention

    problem_number = file_name_parts[1]
    problem_title = " ".join(file_name_parts[2:]).title()

    with open(file_path, encoding="utf-8") as file:
        content = file.read()
        match = leetcode_url_pattern.search(content)
        if match:
            url = match.group(0)
            descriptive_name = f"Problem {problem_number}: {problem_title}"
            return (url, descriptive_name)

    return ("", "")  # Return empty if no URL is found


def scan_directory_for_links(directory: Path) -> List[Tuple[str, str]]:
    """
    Scans a directory recursively for Python files, extracting the first LeetCode URL
    from their initial docstrings, along with generating a descriptive name based on the
    file's naming convention.

    Args:
        directory (Path): The root directory to start scanning from.

    Returns:
        List[Tuple[str, str]]: A sorted list of tuples containing LeetCode URLs and
        their descriptive names. Empty tuples are filtered out.
    """
    links = []
    for file in directory.rglob("p_*.py"):  # Only include files matching the naming convention
        link = extract_leetcode_links(file)
        if link[0]:  # Add link if it's not empty
            links.append(link)
    return sorted(links, key=lambda x: x[1])  # Sort by descriptive name


def write_links_to_markdown(links: List[Tuple[str, str]], output_file: Path) -> None:
    """
    Writes a list of LeetCode URLs and their descriptive names to a Markdown file,
    formatted as a list.

    Args:
        links (List[Tuple[str, str]]): The list of LeetCode URLs and their descriptive names.
        output_file (Path): The path to the Markdown file to write the links to.
    """
    with open(output_file, "w", encoding="utf-8") as md_file:
        md_file.write("# LeetCode Links\n\n")
        for url, name in links:
            md_file.write(f"- [{name}]({url})\n")


if __name__ == "__main__":
    subdirectory = Path("src/leetcode")  # Adjust this path
    output_file = Path("LeetCode_Links.md")  # Output Markdown file
    links = scan_directory_for_links(subdirectory)
    write_links_to_markdown(links, output_file)
