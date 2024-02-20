from heapq import heappop, heappush
from typing import List


class FindPolygonWithLargestPerimeter:
    def solve_with_loop(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        perimeter = sorted(nums)
        for i in range(1, n):
            perimeter[i] += perimeter[i - 1]

        for i in reversed(range(2, n)):
            if nums[i] < perimeter[i - 1]:
                return perimeter[i]

        return -1

    def solve_with_heap(self, nums: List[int]) -> int:
        perimeter = sorted(nums)
        heap = [1]
        for i in range(1, len(nums)):
            perimeter[i] += perimeter[i - 1]
            if i >= 2 and (perimeter[i] - perimeter[i - 1]) < perimeter[i - 1]:
                heappush(heap, -perimeter[i])  # max heap
        return -heappop(heap)

    def solve_reversed(self, nums: List[int]) -> int:
        nums.sort()
        perimeter = sum(nums)
        for index in range(len(nums) - 1, 1, -1):
            perimeter -= nums[index]  # take nums[i] as the largest side
            if nums[index] < perimeter:
                return perimeter + nums[index]
        return -1


if __name__ == "__main__":
    print(FindPolygonWithLargestPerimeter().solve_with_loop([1, 12, 1, 2, 5, 3, 50]))
    print(FindPolygonWithLargestPerimeter().solve_with_heap([1, 12, 1, 2, 5, 3, 50]))
    print(FindPolygonWithLargestPerimeter().solve_reversed([1, 12, 1, 2, 5, 3, 50]))
    print("---")

    print(FindPolygonWithLargestPerimeter().solve_with_loop([5, 5, 5]))
    print(FindPolygonWithLargestPerimeter().solve_with_heap([5, 5, 5]))
    print(FindPolygonWithLargestPerimeter().solve_reversed([5, 5, 5]))
    print("---")

    print(FindPolygonWithLargestPerimeter().solve_with_loop([5, 5, 50]))
    print(FindPolygonWithLargestPerimeter().solve_with_heap([5, 5, 50]))
    print(FindPolygonWithLargestPerimeter().solve_reversed([5, 5, 50]))
