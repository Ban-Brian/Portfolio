package Completed;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Scanner;

public class assignment9 {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        System.out.println("Select a program:");
        System.out.println("1 - Reverse Lines (E11.10)");
        System.out.println("2 - Coin File Reader (P11.7)");
        System.out.println("3 - Bond Energy Lookup (P11.14)");
        int choice = in.nextInt();
        in.nextLine();

        if (choice == 1) {
            testReverse(in);
        } else if (choice == 2) {
            testCoins(in);
        } else if (choice == 3) {
            testBonds(in);
        } else {
            System.out.println("Invalid choice.");
        }
    }

    /** Reads a file, reverses each line, and writes the result back. */
    public static void testReverse(Scanner in) {
        System.out.println("=== Reverse Lines (E11.10) ===");
        System.out.print("Enter the filename: ");
        String filename = in.nextLine();

        try {
            ArrayList<String> lines = new ArrayList<>();
            Scanner fileIn = new Scanner(new File(filename));
            while (fileIn.hasNextLine()) {
                lines.add(fileIn.nextLine());
            }
            fileIn.close();

            System.out.println("Original contents:");
            for (String line : lines) {
                System.out.println("  " + line);
            }

            PrintWriter out = new PrintWriter(filename);
            for (String line : lines) {
                String reversed = new StringBuilder(line).reverse().toString();
                out.println(reversed);
            }
            out.close();

            System.out.println("\nReversed contents written to file:");
            Scanner verify = new Scanner(new File(filename));
            while (verify.hasNextLine()) {
                System.out.println("  " + verify.nextLine());
            }
            verify.close();

        } catch (FileNotFoundException e) {
            System.out.println("Error: File not found - " + e.getMessage());
        }
    }

    /** Reads coins from a file, retries on error, then prints total value. */
    public static void testCoins(Scanner in) {
        System.out.println("=== Coin File Reader (P11.7) ===");

        boolean done = false;
        while (!done) {
            System.out.print("Enter coin filename: ");
            String filename = in.nextLine();

            try {
                ArrayList<Coin> coins = Coin.readFile(filename);
                double total = 0;
                System.out.println("Coins read:");
                for (Coin c : coins) {
                    System.out.println("  " + c.getName() + " = " + c.getValue() + " cents");
                    total += c.getValue();
                }
                System.out.printf("Total value: %.0f cents ($%.2f)%n", total, total / 100);
                done = true;
            } catch (FileNotFoundException e) {
                System.out.println("File not found: " + e.getMessage());
                System.out.println("Please try another file.\n");
            } catch (IllegalArgumentException e) {
                System.out.println("Bad format: " + e.getMessage());
                System.out.println("Please try another file.\n");
            }
        }
    }

    /** Represents a coin with a name and value in cents. */
    static class Coin {
        private String name;
        private double value;

        public Coin() {
            name = "";
            value = 0;
        }

        public Coin(String name, double value) {
            this.name = name;
            this.value = value;
        }

        public String getName() {
            return name;
        }

        public double getValue() {
            return value;
        }

        /** Reads a single coin from the scanner. Throws if the line is badly formatted. */
        public void read(Scanner in) throws FileNotFoundException {
            if (!in.hasNextLine()) {
                throw new FileNotFoundException("No more data in file");
            }
            String line = in.nextLine().trim();
            String[] parts = line.split("\\s+");

            if (parts.length != 2) {
                throw new IllegalArgumentException(
                        "Expected 'name value', got: " + line);
            }

            name = parts[0];
            try {
                value = Double.parseDouble(parts[1]);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException(
                        "Invalid coin value on line: " + line);
            }
        }

        /** Reads all coins from a file and returns them as a list. */
        public static ArrayList<Coin> readFile(String filename) throws FileNotFoundException {
            ArrayList<Coin> coins = new ArrayList<>();
            Scanner fileIn = new Scanner(new File(filename));
            while (fileIn.hasNextLine()) {
                Coin c = new Coin();
                c.read(fileIn);
                coins.add(c);
            }
            fileIn.close();
            return coins;
        }
    }

    /** Looks up bond data by any column value (bond type, energy, or length). */
    public static void testBonds(Scanner in) {
        System.out.println("=== Bond Energy Lookup (P11.14) ===");

        try {
            ArrayList<Bond> bonds = Bond.readFile("OOP/Completed/bonds.txt");

            System.out.println("Bond data loaded (" + bonds.size() + " entries).");
            System.out.println("Enter a bond name, energy (kJ/mol), or length (nm) to search.");
            System.out.println("Type 'quit' to exit.\n");

            while (true) {
                System.out.print("Search: ");
                String query = in.nextLine().trim();
                if (query.equalsIgnoreCase("quit")) {
                    break;
                }

                ArrayList<Bond> matches = new ArrayList<>();
                for (Bond b : bonds) {
                    if (b.matches(query)) {
                        matches.add(b);
                    }
                }

                if (matches.isEmpty()) {
                    System.out.println("No matching bonds found.\n");
                } else {
                    System.out.println("Matches:");
                    for (Bond b : matches) {
                        System.out.printf("  Bond: %-5s  Energy: %4d kJ/mol  Length: %.3f nm%n",
                                b.getName(), b.getEnergy(), b.getLength());
                    }
                    System.out.println();
                }
            }

        } catch (FileNotFoundException e) {
            System.out.println("Error: Bond file not found - " + e.getMessage());
        }
    }

    /** Represents a covalent bond with its energy and length. */
    static class Bond {
        private String name;
        private int energy;
        private double length;

        public Bond(String name, int energy, double length) {
            this.name = name;
            this.energy = energy;
            this.length = length;
        }

        public String getName() {
            return name;
        }

        public int getEnergy() {
            return energy;
        }

        public double getLength() {
            return length;
        }

        /** Returns true if the query matches any column of this bond. */
        public boolean matches(String query) {
            if (name.equalsIgnoreCase(query)) {
                return true;
            }
            if (String.valueOf(energy).equals(query)) {
                return true;
            }
            if (String.valueOf(length).equals(query)) {
                return true;
            }
            try {
                double val = Double.parseDouble(query);
                if (Math.abs(val - length) < 0.0001) {
                    return true;
                }
                if (Math.abs(val - energy) < 0.0001) {
                    return true;
                }
            } catch (NumberFormatException e) {
                // Not a number, already checked name
            }
            return false;
        }

        /** Reads all bonds from a file and returns them as a list. */
        public static ArrayList<Bond> readFile(String filename) throws FileNotFoundException {
            ArrayList<Bond> bonds = new ArrayList<>();
            Scanner fileIn = new Scanner(new File(filename));
            while (fileIn.hasNextLine()) {
                String line = fileIn.nextLine().trim();
                if (line.isEmpty()) {
                    continue;
                }
                String[] parts = line.split("\\s+");
                if (parts.length != 3) {
                    throw new IllegalArgumentException("Bad bond line: " + line);
                }
                String name = parts[0];
                int energy = Integer.parseInt(parts[1]);
                double length = Double.parseDouble(parts[2]);
                bonds.add(new Bond(name, energy, length));
            }
            fileIn.close();
            return bonds;
        }
    }
}
