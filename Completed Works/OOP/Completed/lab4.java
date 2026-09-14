package Completed;

import java.util.Scanner;
import java.util.Random;

// Lab 4 - Implementation of Connect Four game with Human and AI players using Minimax with Alpha-Beta pruning
public class lab4 {

    static final int ROWS = 6;
    static final int COLS = 7;

    interface Player {
        void init(boolean color);
        String name();
        int move();
        void inform(int col);
    }

    static class Board {
        int[][] grid;

        Board() {
            grid = new int[ROWS][COLS];
        }

        int drop(int col, int token) {
            for (int r = ROWS - 1; r >= 0; r--) {
                if (grid[r][col] == 0) {
                    grid[r][col] = token;
                    return r;
                }
            }
            return -1;
        }

        void undrop(int col) {
            for (int r = 0; r < ROWS; r++) {
                if (grid[r][col] != 0) {
                    grid[r][col] = 0;
                    return;
                }
            }
        }

        boolean isColumnFull(int col) {
            return grid[0][col] != 0;
        }

        boolean isFull() {
            for (int c = 0; c < COLS; c++) {
                if (!isColumnFull(c)) return false;
            }
            return true;
        }

        boolean checkWin(int token) {
            for (int r = 0; r < ROWS; r++) {
                for (int c = 0; c < COLS; c++) {
                    if (grid[r][c] == token) {
                        if (countDirection(r, c, 0, 1, token) >= 4) return true;
                        if (countDirection(r, c, 1, 0, token) >= 4) return true;
                        if (countDirection(r, c, 1, 1, token) >= 4) return true;
                        if (countDirection(r, c, 1, -1, token) >= 4) return true;
                    }
                }
            }
            return false;
        }

        int countDirection(int row, int col, int dRow, int dCol, int token) {
            int count = 0;
            int r = row;
            int c = col;
            while (r >= 0 && r < ROWS && c >= 0 && c < COLS && grid[r][c] == token) {
                count++;
                r += dRow;
                c += dCol;
            }
            return count;
        }

        void display() {
            System.out.println();
            System.out.println("  0   1   2   3   4   5   6");
            System.out.println("+---+---+---+---+---+---+---+");
            for (int r = 0; r < ROWS; r++) {
                System.out.print("|");
                for (int c = 0; c < COLS; c++) {
                    if (grid[r][c] == 0) {
                        System.out.print("   |");
                    } else if (grid[r][c] == 1) {
                        System.out.print(" 1 |");
                    } else {
                        System.out.print(" 0 |");
                    }
                }
                System.out.println();
                System.out.println("+---+---+---+---+---+---+---+");
            }
            System.out.println();
        }
    }

    static class HumanPlayer implements Player {
        boolean color;
        String playerName;
        Scanner scanner;

        HumanPlayer(String name, Scanner scanner) {
            this.playerName = name;
            this.scanner = scanner;
        }

        public void init(boolean color) {
            this.color = color;
        }

        public String name() {
            return playerName;
        }

        public int move() {
            System.out.print(playerName + " (" + (color ? "1/Red" : "0/Yellow") + "), pick a column (0-6): ");
            return scanner.nextInt();
        }

        public void inform(int col) {
        }
    }

    static class ComputerPlayer implements Player {
        boolean color;
        int myToken;
        int oppToken;
        Board board;
        Random rand;
        static final int MAX_DEPTH = 8;

        ComputerPlayer() {
            board = new Board();
            rand = new Random();
        }

        public void init(boolean color) {
            this.color = color;
            myToken = color ? 1 : 2;
            oppToken = color ? 2 : 1;
        }

        public String name() {
            return "Computer";
        }

        public void inform(int col) {
            board.drop(col, oppToken);
        }

        public int move() {
            int bestCol = -1;
            int bestScore = Integer.MIN_VALUE;

            for (int c = 0; c < COLS; c++) {
                if (board.isColumnFull(c)) continue;

                board.drop(c, myToken);

                if (board.checkWin(myToken)) {
                    board.undrop(c);
                    board.drop(c, myToken);
                    System.out.println("Computer plays column " + c);
                    return c;
                }

                int score = minimax(MAX_DEPTH - 1, false, Integer.MIN_VALUE, Integer.MAX_VALUE);
                board.undrop(c);

                if (score > bestScore || (score == bestScore && rand.nextBoolean())) {
                    bestScore = score;
                    bestCol = c;
                }
            }

            board.drop(bestCol, myToken);
            System.out.println("Computer plays column " + bestCol);
            return bestCol;
        }

        int minimax(int depth, boolean isMaximizing, int alpha, int beta) {
            if (board.checkWin(myToken)) return 100000 + depth;
            if (board.checkWin(oppToken)) return -100000 - depth;
            if (board.isFull() || depth == 0) return evaluate();

            if (isMaximizing) {
                int maxEval = Integer.MIN_VALUE;
                for (int c = 0; c < COLS; c++) {
                    if (board.isColumnFull(c)) continue;
                    board.drop(c, myToken);
                    int eval = minimax(depth - 1, false, alpha, beta);
                    board.undrop(c);
                    maxEval = Math.max(maxEval, eval);
                    alpha = Math.max(alpha, eval);
                    if (beta <= alpha) break;
                }
                return maxEval;
            } else {
                int minEval = Integer.MAX_VALUE;
                for (int c = 0; c < COLS; c++) {
                    if (board.isColumnFull(c)) continue;
                    board.drop(c, oppToken);
                    int eval = minimax(depth - 1, true, alpha, beta);
                    board.undrop(c);
                    minEval = Math.min(minEval, eval);
                    beta = Math.min(beta, eval);
                    if (beta <= alpha) break;
                }
                return minEval;
            }
        }

        int evaluate() {
            int score = 0;
            for (int r = 0; r < ROWS; r++) {
                if (board.grid[r][3] == myToken) score += 3;
                if (board.grid[r][3] == oppToken) score -= 3;
            }
            score += scoreAllWindows();
            return score;
        }

        int scoreAllWindows() {
            int total = 0;

            for (int r = 0; r < ROWS; r++) {
                for (int c = 0; c <= COLS - 4; c++) {
                    total += scoreWindow(
                        board.grid[r][c], board.grid[r][c+1],
                        board.grid[r][c+2], board.grid[r][c+3]);
                }
            }

            for (int c = 0; c < COLS; c++) {
                for (int r = 0; r <= ROWS - 4; r++) {
                    total += scoreWindow(
                        board.grid[r][c], board.grid[r+1][c],
                        board.grid[r+2][c], board.grid[r+3][c]);
                }
            }

            for (int r = 0; r <= ROWS - 4; r++) {
                for (int c = 0; c <= COLS - 4; c++) {
                    total += scoreWindow(
                        board.grid[r][c], board.grid[r+1][c+1],
                        board.grid[r+2][c+2], board.grid[r+3][c+3]);
                }
            }

            for (int r = 0; r <= ROWS - 4; r++) {
                for (int c = 3; c < COLS; c++) {
                    total += scoreWindow(
                        board.grid[r][c], board.grid[r+1][c-1],
                        board.grid[r+2][c-2], board.grid[r+3][c-3]);
                }
            }

            return total;
        }

        int scoreWindow(int a, int b, int c, int d) {
            int mine = 0, opp = 0, empty = 0;
            int[] cells = {a, b, c, d};

            for (int cell : cells) {
                if (cell == myToken) mine++;
                else if (cell == oppToken) opp++;
                else empty++;
            }

            if (mine == 4) return 100;
            if (mine == 3 && empty == 1) return 10;
            if (mine == 2 && empty == 2) return 3;
            if (opp == 3 && empty == 1) return -80;
            if (opp == 2 && empty == 2) return -2;
            return 0;
        }
    }

    static class Game {
        Board board;
        Player player1;
        Player player2;

        Game(Player p1, Player p2) {
            board = new Board();
            player1 = p1;
            player2 = p2;
            player1.init(true);
            player2.init(false);
        }

        void play() {
            System.out.println("\n" + player1.name() + " (1/Red) vs " + player2.name() + " (0/Yellow)");
            System.out.println("Red goes first.\n");
            board.display();

            boolean redTurn = true;

            while (true) {
                Player current = redTurn ? player1 : player2;
                int token = redTurn ? 1 : 2;
                int col;

                while (true) {
                    col = current.move();
                    if (col < 0 || col >= COLS) {
                        System.out.println("Invalid column. Pick 0-6.");
                    } else if (board.isColumnFull(col)) {
                        System.out.println("Column " + col + " is full. Pick another.");
                    } else {
                        break;
                    }
                }

                board.drop(col, token);
                board.display();

                if (board.checkWin(token)) {
                    System.out.println(current.name() + " wins!");
                    return;
                }

                if (board.isFull()) {
                    System.out.println("It's a tie!");
                    return;
                }

                Player other = redTurn ? player2 : player1;
                other.inform(col);

                redTurn = !redTurn;
            }
        }
    }

    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        System.out.println("=== Connect Four ===\n");
        System.out.println("Select game mode:");
        System.out.println("1 - Human vs Human");
        System.out.println("2 - Human vs Computer");
        System.out.println("3 - Computer vs Human");
        System.out.println("4 - Computer vs Computer");
        int mode = in.nextInt();

        Player p1;
        Player p2;

        if (mode == 1) {
            p1 = new HumanPlayer("Player 1", in);
            p2 = new HumanPlayer("Player 2", in);
        } else if (mode == 2) {
            p1 = new HumanPlayer("Player 1", in);
            p2 = new ComputerPlayer();
        } else if (mode == 3) {
            p1 = new ComputerPlayer();
            p2 = new HumanPlayer("Player 2", in);
        } else if (mode == 4) {
            p1 = new ComputerPlayer();
            p2 = new ComputerPlayer();
        } else {
            System.out.println("Invalid choice.");
            return;
        }

        Game game = new Game(p1, p2);
        game.play();
    }
}
