class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows == 1 or numRows >= len(s):
            return s 
        
        row = [""] * numRows
        curRow = 0
        goingDown = False
        
        for c in s:
            row[curRow] += c
            if curRow == 0 or curRow == numRows - 1:
                goingDown = not goingDown
            curRow += 1 if goingDown else -1
        
        return "".join(row)
        