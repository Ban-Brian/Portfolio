class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9  # 3x3 board represented as a list
        self.current_player = 'X'

    def display_board(self):
        """Display the current state of the board."""
        print("\n")
        print(f" {self.board[0]} | {self.board[1]} | {self.board[2]} ")
        print("---+---+---")
        print(f" {self.board[3]} | {self.board[4]} | {self.board[5]} ")
        print("---+---+---")
        print(f" {self.board[6]} | {self.board[7]} | {self.board[8]} ")
        print("\n")

    def make_move(self, position):
        """Make a move at the specified position (1-9)."""
        if self.board[position - 1] == ' ':
            self.board[position - 1] = self.current_player
            return True
        else:
            print("Invalid move. The position is already taken.")
            return False

    def switch_player(self):
        """Switch the current player."""
        self.current_player = 'O' if self.current_player == 'X' else 'X'

    def check_winner(self):
        """Check if there's a winner."""
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]  # Diagonals
        ]
        for combo in winning_combinations:
            if self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]] != ' ':
                return True
        return False

    def is_draw(self):
        """Check if the game is a draw."""
        return ' ' not in self.board

    def play(self):
        """Main game loop."""
        print("Welcome to Tic Tac Toe!")
        while True:
            self.display_board()
            try:
                position = int(input(f"Player {self.current_player}, enter your move (1-9): "))
                if position < 1 or position > 9:
                    print("Invalid input. Please choose a number between 1 and 9.")
                    continue
            except ValueError:
                print("Invalid input. Please enter a number.")
                continue

            if self.make_move(position):
                if self.check_winner():
                    self.display_board()
                    print(f"Player {self.current_player} wins!")
                    break
                elif self.is_draw():
                    self.display_board()
                    print("It's a draw!")
                    break
                else:
                    self.switch_player()


if __name__ == "__main__":
    game = TicTacToe()
    game.play()
