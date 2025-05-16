# Find the maximum number of fish the fisher can catch by starting at the best water cell
class Solution:
   def findMaxFish(self, grid):
       m, n = len(grid), len(grid[0])
       visited = set()
       max_fish = 0

       def bfs(r, c):
           queue = deque([(r, c)])
           visited.add((r, c))
           total_fish = grid[r][c]

           while queue:
               curr_r, curr_c = queue.popleft()

               for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                   new_r, new_c = curr_r + dr, curr_c + dc

                   if (0 <= new_r < m and 0 <= new_c < n and
                           grid[new_r][new_c] > 0 and
                           (new_r, new_c) not in visited):
                       queue.append((new_r, new_c))
                       visited.add((new_r, new_c))
                       total_fish += grid[new_r][new_c]

           return total_fish

       for r in range(m):
           for c in range(n):
               if grid[r][c] > 0 and (r, c) not in visited:
                   max_fish = max(max_fish, bfs(r, c))

       return max_fish
   # You have a grid (a 2D matrix) where:
   # A "0" represents a land cell.
   # A positive number represents a water cell with fish, where the number tells how many fish are in that cell.
   # A fisher can start at any water cell and do two things:
   # Catch all the fish in the current cell.
   # Move to any adjacent water cell (up, down, left, or right).
   # The goal is to find the maximum number of fish the fisher can catch by starting at the best water cell.
   # If no water cells exist, return 0.
sys.exit()