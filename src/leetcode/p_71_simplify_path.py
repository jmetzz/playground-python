"""

71. Simplify Path

My solution:
https://leetcode.com/problems/simplify-path/solutions/5714660/simple-and-efficient-o-n-stack-based-solution-beats-98-65/

Medium

Given an absolute path for a Unix-style file system,
which begins with a slash '/', transform this path into
its simplified canonical path.

In Unix-style file system context, a single period '.' signifies
the current directory, a double period ".." denotes moving up
one directory level, and multiple slashes such as "//" are interpreted
as a single slash. In this problem, treat sequences of periods
not covered by the previous rules (like "...") as valid names for
files or directories.

The simplified canonical path should adhere to the following rules:

- It must start with a single slash '/'.
- Directories within the path should be separated by only one slash '/'.
- It should not end with a slash '/', unless it's the root directory.
- It should exclude any single or double periods used to denote current
or parent directories.

Return the new path.

Example 1:
Input: path = "/home/"
Output: "/home"
Explanation:
The trailing slash should be removed.

Example 2:
Input: path = "/home//foo/"
Output: "/home/foo"
Explanation:
Multiple consecutive slashes are replaced by a single one.

Example 3:
Input: path = "/home/user/Documents/../Pictures"
Output: "/home/user/Pictures"
Explanation:
A double period ".." refers to the directory up a level.

Example 4:
Input: path = "/../"
Output: "/"
Explanation:
Going one level up from the root directory is not possible.

Example 5:
Input: path = "/.../a/../b/c/../d/./"
Output: "/.../b/d"
Explanation:
"..." is a valid name for a directory in this problem.


Constraints:
1 <= path.length <= 3000
path consists of English letters, digits, period '.', slash '/' or '_'.
path is a valid absolute Unix path.
"""


def simplify_path(path_str: str) -> str:
    """
    Simplifies an absolute Unix-style file path.

    This function takes an absolute path string and returns its canonical form
    by resolving special directory names such as '.' (current directory) and
    '..' (parent directory), and by removing redundant slashes.

    Args:
        path_str (str): The absolute path string to be simplified.

    Returns:
        str: The simplified canonical path.

    Examples:
        >>> simplify_path("/home/")
        '/home'

        >>> simplify_path("/../")
        '/'

        >>> simplify_path("/home//foo/")
        '/home/foo'

        >>> simplify_path("/a/./b/../../c/")
        '/c'
    """
    stack = []
    parts = path_str.split("/")
    for part in parts:
        if part == "" or part == ".":
            continue
        if part == "..":
            if stack:
                stack.pop()
        else:
            stack.append(part)
    return "/" + "/".join(stack)


def simplify_path_match_case(path_str: str) -> str:
    stack = []
    parts = path_str.split("/")
    for part in parts:
        match part:
            case "":
                continue
            case ".":
                continue
            case "..":
                if stack:
                    stack.pop()
            case _:
                stack.append(part)
    return "/" + "/".join(stack)
