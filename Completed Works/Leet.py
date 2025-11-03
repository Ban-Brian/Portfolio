class Solution:
    def longestBalanced(self, s: str) -> int:
        dif_map = {(0, 0): 0}
        num_a = 0
        num_b = 0
        num_c = 0
        max_len = 0

        for j, char in enumerate(s):
            if char == 'a':
                num_a += 1
            elif char == 'b':
                num_b += 1
            elif char == 'c':
                num_c += 1

            dif_ab = num_a - num_b
            dif_bc = num_b - num_c
            current_c = (dif_ab, dif_bc)

            if current_c in dif_map:
                first_i = dif_map[current_c]
                max_len = max(max_len, (j + 1) - first_i)
            else:
                dif_map[current_c] = j + 1

        return max_len