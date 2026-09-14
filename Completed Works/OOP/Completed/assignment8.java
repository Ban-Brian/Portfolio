package Completed;

import java.util.ArrayList;
import java.util.Scanner;

public class assignment8 {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        System.out.println("Select a program:");
        System.out.println("1 - Grid (E10.14)");
        System.out.println("2 - PrimeSequence (P10.2)");
        System.out.println("3 - FirstDigitDistribution (P10.4)");
        int choice = in.nextInt();

        if (choice == 1) {
            testGrid();
        } else if (choice == 2) {
            testPrimeSequence();
        } else if (choice == 3) {
            testFirstDigitDistribution();
        } else {
            System.out.println("Invalid choice.");
        }
    }

    public static void testGrid() {
        System.out.println("=== Grid Test (E10.14) ===");
        Grid grid = new Grid(3, 4);

        grid.add(0, 0, "Top-left corner");
        grid.add(0, 3, "Top-right corner");
        grid.add(1, 2, "Center area");
        grid.add(2, 0, "Bottom-left corner");
        grid.add(2, 3, "Bottom-right corner");

        System.out.println("Location (0,0): " + grid.getDescription(0, 0));
        System.out.println("Location (1,1): " + grid.getDescription(1, 1));
        System.out.println("Location (2,3): " + grid.getDescription(2, 3));

        System.out.println("\nAll described locations:");
        ArrayList<Grid.Location> locations = grid.getDescribedLocations();
        for (Grid.Location loc : locations) {
            System.out.println("  Row " + loc.getRow() + ", Column " + loc.getColumn()
                    + " -> " + grid.getDescription(loc.getRow(), loc.getColumn()));
        }
    }

    public static void testPrimeSequence() {
        System.out.println("=== PrimeSequence Test (P10.2) ===");
        PrimeSequence primes = new PrimeSequence();

        System.out.println("First 20 prime numbers:");
        for (int i = 0; i < 20; i++) {
            System.out.print(primes.next() + " ");
        }
        System.out.println();
    }

    public static void testFirstDigitDistribution() {
        System.out.println("=== FirstDigitDistribution Test (P10.4) ===");
        Sequence primes = new PrimeSequence();
        FirstDigitDistribution dist = new FirstDigitDistribution();
        dist.process(primes, 1000);
        dist.display();
    }

    /** Stores descriptions in a rectangular grid. */
    static class Grid {
        private String[][] descriptions;
        private int numRows;
        private int numColumns;

        public Grid(int numRows, int numColumns) {
            this.numRows = numRows;
            this.numColumns = numColumns;
            descriptions = new String[numRows][numColumns];
        }

        public void add(int row, int column, String description) {
            descriptions[row][column] = description;
        }

        public String getDescription(int row, int column) {
            return descriptions[row][column];
        }

        public ArrayList<Location> getDescribedLocations() {
            ArrayList<Location> result = new ArrayList<>();
            for (int r = 0; r < numRows; r++) {
                for (int c = 0; c < numColumns; c++) {
                    if (descriptions[r][c] != null) {
                        result.add(new Location(r, c));
                    }
                }
            }
            return result;
        }

        /** Encapsulates a row and column position in the grid. */
        public static class Location {
            private int row;
            private int column;

            public Location(int row, int column) {
                this.row = row;
                this.column = column;
            }

            public int getRow() {
                return row;
            }

            public int getColumn() {
                return column;
            }
        }
    }

    /** Produces a sequence of values one at a time. */
    interface Sequence {

        int next();
    }

    /** Produces the sequence of prime numbers. */
    static class PrimeSequence implements Sequence {
        private int current = 1;

        public int next() {
            current++;
            while (!isPrime(current)) {
                current++;
            }
            return current;
        }

        private static boolean isPrime(int n) {
            if (n < 2) {
                return false;
            }
            for (int i = 2; i <= Math.sqrt(n); i++) {
                if (n % i == 0) {
                    return false;
                }
            }
            return true;
        }
    }

    /**
     * Counts the distribution of first digits (1-9) across a sequence of values.
     */
    static class FirstDigitDistribution {
        private int[] counters;

        public FirstDigitDistribution() {
            counters = new int[10];
        }

        public void process(Sequence seq, int valuesToProcess) {
            for (int i = 0; i < valuesToProcess; i++) {
                int value = seq.next();
                int firstDigit = getFirstDigit(value);
                if (firstDigit >= 1 && firstDigit <= 9) {
                    counters[firstDigit]++;
                }
            }
        }

        public void display() {
            System.out.println("First Digit Distribution:");
            for (int i = 1; i <= 9; i++) {
                System.out.printf("%d: %s (%d)%n", i, buildBar(counters[i]), counters[i]);
            }
        }

        private int getFirstDigit(int value) {
            value = Math.abs(value);
            while (value >= 10) {
                value /= 10;
            }
            return value;
        }

        private String buildBar(int count) {
            StringBuilder bar = new StringBuilder();
            for (int i = 0; i < count / 5; i++) {
                bar.append("*");
            }
            return bar.toString();
        }
    }
}
