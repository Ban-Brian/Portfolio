#ways you can write a number `n` as the sum of consecutive positive integers.
class Solution:
    def consecutiveNumbersSum(self, n: int) -> int:
        count = 0
        k = 1
        while k * (k - 1) // 2 < n:
            remainder = n - k * (k - 1) // 2
            if remainder % k == 0:
                m = remainder // k
                if m > 0:
                    count += 1
            k += 1
        return count

if __name__ == "__main__":
    print(Solution().consecutiveNumbersSum(15))

