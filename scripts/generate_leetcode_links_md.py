import re
from pathlib import Path
from typing import List, Set

# Compile the regular expression once and use it globally
leetcode_url_pattern = re.compile(r"https://leetcode\.\w+/problems/[\w-]+/")


def extract_leetcode_links(file_path: Path) -> Set[str]:
    """
    Extracts LeetCode URLs from the initial docstring of a Python file.

    This function searches for URLs that match the LeetCode problem URL pattern
    within the initial docstring of the specified Python file, assuming the
    docstring appears before any imports.

    Args:
        file_path (Path): The path to the Python file to be scanned.

    Returns:
        Set[str]: A set of unique LeetCode URLs found in the file's initial docstring.
    """
    links = set()

    with open(file_path, encoding="utf-8") as file:
        content = []
        for line in file:
            if line.startswith('"""'):  # Start of docstring
                content.append(line)
                break
        for line in file:
            content.append(line)
            if line.startswith('"""'):  # End of docstring
                break
        docstring = "\n".join(content)
        links.update(leetcode_url_pattern.findall(docstring))

    return links


def scan_directory_for_links(directory: Path) -> List[str]:
    """
    Scans a directory recursively for Python files and extracts LeetCode URLs from
    their initial docstrings.

    Args:
        directory (Path): The root directory to start scanning from.

    Returns:
        List[str]: A sorted list of unique LeetCode URLs found.
    """
    links = set()
    for file in directory.rglob("*.py"):
        links.update(extract_leetcode_links(file))
    return sorted(links)


def write_links_to_markdown(links: List[str], output_file: Path) -> None:
    """
    Writes a list of LeetCode URLs to a Markdown file, formatted as a list.

    Args:
        links (List[str]): The list of LeetCode URLs to write.
        output_file (Path): The path to the Markdown file to write the URLs to.
    """
    with open(output_file, "w", encoding="utf-8") as md_file:
        md_file.write("# LeetCode Links\n\n")
        for link in links:
            md_file.write(f"- [{link}]({link})\n")


if __name__ == "__main__":
    subdirectory = Path("src/leetcode")  # Adjust this path
    output_file = Path("LeetCode_Links.md")  # Output Markdown file
    links = scan_directory_for_links(subdirectory)
    write_links_to_markdown(links, output_file)
