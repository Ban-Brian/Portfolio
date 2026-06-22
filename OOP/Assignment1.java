/**
 * Assignment 1 — Exercises E1.1, E1.4, and P1.3.
 * E1.1: Prints a greeting in Samoan.
 * E1.4: Computes account balance over 3 years at 5% interest.
 * P1.3: Approximates Pi using the Leibniz series to six significant digits.
 *
 * @author Brian Butler
 */
public class Assignment1 {

    /**
     * Runs all three exercises and prints the results.
     */
    public static void main(String[] args) {

        // E1.1
        System.out.println("=== E1.1 ===");
        System.out.println("Talfoa!!!");
        System.out.println();

        // E1.4
        System.out.println("=== E1.4 ===");

        double balance = 1000.00;
        double interestRate = 0.05;

        for (int year = 1; year <= 3; year++) {
            balance = balance + balance * interestRate;
            System.out.printf("After year %d: $%.2f%n", year, balance);
        }

        System.out.println();

        // P1.3
        System.out.println("=== P1.3 ===");
        double pi = 0.0;
        int terms = 0;
        int sign = 1;
        final double TARGET = 3.14159;

        while (true) {
            pi += sign / (2.0 * terms + 1);
            terms++;
            sign = -sign;

            double currentPi = pi * 4.0;
            if (Math.abs(currentPi - Math.PI) < 0.000005) {
                System.out.printf("Pi ≈ %.6f%n", currentPi);
                System.out.printf("Terms used: %,d%n", terms);
                break;
            }
        }
    }
}
