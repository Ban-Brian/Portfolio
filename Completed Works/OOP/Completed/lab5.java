package Completed;

import java.util.Scanner;

/**
 * 2-Player 3D Tic Tac Toe (4x4x4 Qubic) game.
 * Two players take turns placing X and O on a 4x4x4 board.
 * First to get four in a row wins.
 */
public class lab5 {

    static final int SIZE = 4;

    /**
     * Represents the 4x4x4 game board as 4 layers of 4x4 grids.
     */
    static class Board {
        char[][][] grid;

        /**
         * Creates an empty 4x4x4 board filled with spaces.
         */
        Board() {
            grid = new char[SIZE][SIZE][SIZE];
            for (int layer = 0; layer < SIZE; layer++)
                for (int row = 0; row < SIZE; row++)
                    for (int col = 0; col < SIZE; col++)
                        grid[layer][row][col] = ' ';
        }

        /**
         * Places a marker on the board if the cell is empty.
         * @param layer the layer index (0-3)
         * @param row the row index (0-3)
         * @param col the column index (0-3)
         * @param marker the player's marker ('X' or 'O')
         * @return true if the move was made, false if the cell was occupied
         */
        boolean makeMove(int layer, int row, int col, char marker) {
            if (grid[layer][row][col] != ' ')
                return false;
            grid[layer][row][col] = marker;
            return true;
        }

        /**
         * Displays the 4 layers side-by-side in the console.
         */
        void display() {
            System.out.println();

            System.out.print("  ");
            for (int layer = 0; layer < SIZE; layer++) {
                System.out.printf("   Layer %-2d       ", layer + 1);
            }
            System.out.println();

            System.out.print("  ");
            for (int layer = 0; layer < SIZE; layer++) {
                System.out.print("  0   1   2   3   ");
            }
            System.out.println();

            for (int row = 0; row < SIZE; row++) {
                System.out.print("  ");
                for (int layer = 0; layer < SIZE; layer++) {
                    System.out.print("+---+---+---+---+ ");
                }
                System.out.println();

                System.out.print(row + " ");
                for (int layer = 0; layer < SIZE; layer++) {
                    for (int col = 0; col < SIZE; col++) {
                        System.out.print("| " + grid[layer][row][col] + " ");
                    }
                    System.out.print("| ");
                }
                System.out.println();
            }

            System.out.print("  ");
            for (int layer = 0; layer < SIZE; layer++) {
                System.out.print("+---+---+---+---+ ");
            }
            System.out.println();
            System.out.println();
        }

        /**
         * Checks if every cell on the board is occupied.
         * @return true if the board is full
         */
        boolean isFull() {
            for (int l = 0; l < SIZE; l++)
                for (int r = 0; r < SIZE; r++)
                    for (int c = 0; c < SIZE; c++)
                        if (grid[l][r][c] == ' ')
                            return false;
            return true;
        }

        /**
         * Checks all 76 possible winning lines for the given marker.
         * @param marker the marker to check ('X' or 'O')
         * @return true if the marker has four in a row
         */
        boolean checkWin(char marker) {
            for (int l = 0; l < SIZE; l++) {
                for (int r = 0; r < SIZE; r++)
                    if (line(grid[l][r][0], grid[l][r][1], grid[l][r][2], grid[l][r][3], marker))
                        return true;

                for (int c = 0; c < SIZE; c++)
                    if (line(grid[l][0][c], grid[l][1][c], grid[l][2][c], grid[l][3][c], marker))
                        return true;

                if (line(grid[l][0][0], grid[l][1][1], grid[l][2][2], grid[l][3][3], marker))
                    return true;
                if (line(grid[l][0][3], grid[l][1][2], grid[l][2][1], grid[l][3][0], marker))
                    return true;
            }

            for (int r = 0; r < SIZE; r++)
                for (int c = 0; c < SIZE; c++)
                    if (line(grid[0][r][c], grid[1][r][c], grid[2][r][c], grid[3][r][c], marker))
                        return true;

            for (int r = 0; r < SIZE; r++) {
                if (line(grid[0][r][0], grid[1][r][1], grid[2][r][2], grid[3][r][3], marker))
                    return true;
                if (line(grid[0][r][3], grid[1][r][2], grid[2][r][1], grid[3][r][0], marker))
                    return true;
            }

            for (int c = 0; c < SIZE; c++) {
                if (line(grid[0][0][c], grid[1][1][c], grid[2][2][c], grid[3][3][c], marker))
                    return true;
                if (line(grid[0][3][c], grid[1][2][c], grid[2][1][c], grid[3][0][c], marker))
                    return true;
            }

            if (line(grid[0][0][0], grid[1][1][1], grid[2][2][2], grid[3][3][3], marker))
                return true;
            if (line(grid[0][0][3], grid[1][1][2], grid[2][2][1], grid[3][3][0], marker))
                return true;
            if (line(grid[0][3][0], grid[1][2][1], grid[2][1][2], grid[3][0][3], marker))
                return true;
            if (line(grid[0][3][3], grid[1][2][2], grid[2][1][1], grid[3][0][0], marker))
                return true;

            return false;
        }

        /**
         * Checks if four cells all match the given marker.
         * @param a first cell
         * @param b second cell
         * @param c third cell
         * @param d fourth cell
         * @param marker the marker to match
         * @return true if all four cells equal the marker
         */
        boolean line(char a, char b, char c, char d, char marker) {
            return (a == marker && b == marker && c == marker && d == marker);
        }
    }

    /**
     * Manages the game loop between two human players.
     */
    static class Game {
        Board board;
        Scanner scanner;

        /**
         * Creates a new game with an empty board.
         * @param scanner the Scanner for player input
         */
        Game(Scanner scanner) {
            board = new Board();
            this.scanner = scanner;
        }

        /**
         * Runs the game loop, alternating turns until someone wins or the board is full.
         */
        void play() {
            System.out.println("\n=== 3-D Tic Tac Toe (4x4x4) ===");
            System.out.println("Player 1: X    Player 2: O");
            System.out.println("Enter moves as: layer row col  (each 0-3)\n");
            board.display();

            char currentMarker = 'X';
            int playerNum = 1;

            while (true) {
                int layer, row, col;

                while (true) {
                    System.out.print("Player " + playerNum + " (" + currentMarker + "), enter move (layer row col): ");
                    if (!scanner.hasNextInt()) { scanner.next(); System.out.println("Invalid input. Enter three numbers 0-3."); continue; }
                    layer = scanner.nextInt();
                    if (!scanner.hasNextInt()) { scanner.next(); System.out.println("Invalid input. Enter three numbers 0-3."); continue; }
                    row = scanner.nextInt();
                    if (!scanner.hasNextInt()) { scanner.next(); System.out.println("Invalid input. Enter three numbers 0-3."); continue; }
                    col = scanner.nextInt();

                    if (layer < 0 || layer >= SIZE || row < 0 || row >= SIZE || col < 0 || col >= SIZE) {
                        System.out.println("Out of range. Each value must be 0-3.");
                        continue;
                    }

                    if (!board.makeMove(layer, row, col, currentMarker)) {
                        System.out.println("That cell is already taken. Try again.");
                        continue;
                    }

                    break;
                }

                board.display();

                if (board.checkWin(currentMarker)) {
                    System.out.println("Player " + playerNum + " (" + currentMarker + ") wins!");
                    return;
                }

                if (board.isFull()) {
                    System.out.println("The board is full — it's a tie!");
                    return;
                }

                if (currentMarker == 'X') {
                    currentMarker = 'O';
                    playerNum = 2;
                } else {
                    currentMarker = 'X';
                    playerNum = 1;
                }
            }
        }
    }

    /**
     * Entry point — starts a new 3D Tic Tac Toe game.
     * @param args command line arguments (not used)
     */
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        Game game = new Game(in);
        game.play();
    }
}
