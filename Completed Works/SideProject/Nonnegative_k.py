import sys
from collections import deque

# Find non-negative integer of k
class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        encoded_word = []
        vowel_map = {'a': 0, 'e': 1, 'i': 2, 'o': 3, 'u': 4}

        for char in word:
            if char in vowel_map:
                encoded_word.append(vowel_map[char])
            else:
                encoded_word.append(5)

        total_count = 0
        n = len(word)
        frequencies = [0, 0, 0, 0, 0, 0]
        left = 0
        window_start = 0

        for right in range(n):
            char_code = encoded_word[right]
            frequencies[char_code] += 1

            while frequencies[5] > k and left <= right:
                frequencies[encoded_word[left]] -= 1
                left += 1
                window_start = left

            while frequencies[5] == k and left < right:
                char_code = encoded_word[left]
                if char_code < 5 and frequencies[char_code] > 1:
                    frequencies[char_code] -= 1
                    left += 1
                else:
                    break

            if frequencies[5] == k and all(frequencies[i] > 0 for i in range(5)):
                total_count += (left - window_start + 1)

        return total_count #
    #  You are given a string word and a non-negative integer k.
    # Return the total number of substrings of word that contain every vowel ('a', 'e', 'i', 'o', and 'u')
    # at least once and exactly k consonants.
sys.exit()

