package Completed;

import java.util.Scanner;

/**
 * Assignment 3 — Exercises E4.4/E4.5, E4.24, P4.9, P4.13
 *
 * @author Brian Butler
 */
public class Assignment3 {

    // E4.24 — Spherical balloon being filled with air
    static class Balloon {

        private double volume;

        // Start with an empty balloon
        public Balloon() {
            volume = 0;
        }

        // Add air to the balloon (in cm³)
        public void addAir(double amount) {
            volume = volume + amount;
        }

        // Get current volume in cm³
        public double getVolume() {
            return volume;
        }

        // Radius from volume: r = (3V / 4π)^(1/3)
        public double getRadius() {
            return Math.pow((3.0 * volume) / (4.0 * Math.PI), 1.0 / 3.0);
        }

        // Surface area of a sphere: A = 4πr²
        public double getSurfaceArea() {
            double r = getRadius();
            return 4.0 * Math.PI * r * r;
        }
    }

    public static void main(String[] args) {

        Scanner in = new Scanner(System.in);

        // ---- E4.4 & E4.5: Two-integer calculator with aligned output ----
        System.out.println("=== E4.4 / E4.5: Integer Calculator ===");

        System.out.print("Enter first integer: ");
        int a = in.nextInt();
        System.out.print("Enter second integer: ");
        int b = in.nextInt();

        // Compute all requested values
        int sum = a + b;
        int difference = a - b;
        int product = a * b;
        double average = (a + b) / 2.0;
        int distance = Math.abs(a - b);
        int maximum = Math.max(a, b);
        int minimum = Math.min(a, b);

        // Print with aligned formatting (E4.5)
        System.out.printf("Sum:        %d%n", sum);
        System.out.printf("Difference: %d%n", difference);
        System.out.printf("Product:    %d%n", product);
        System.out.printf("Average:    %.2f%n", average);
        System.out.printf("Distance:   %d%n", distance);
        System.out.printf("Maximum:    %d%n", maximum);
        System.out.printf("Minimum:    %d%n", minimum);
        System.out.println();

        // ---- E4.24: Balloon Tester ----
        System.out.println("=== E4.24: Balloon Tester ===");

        Balloon balloon = new Balloon();

        // Add 100 cm³ of air and check measurements
        balloon.addAir(100);
        System.out.println("After adding 100 cm³ of air:");
        System.out.printf("  Volume:       %.2f cm³%n", balloon.getVolume());
        System.out.printf("  Radius:       %.2f cm%n", balloon.getRadius());
        System.out.printf("  Surface Area: %.2f cm²%n", balloon.getSurfaceArea());

        // Add another 100 cm³ and check again
        balloon.addAir(100);
        System.out.println("After adding another 100 cm³ of air:");
        System.out.printf("  Volume:       %.2f cm³%n", balloon.getVolume());
        System.out.printf("  Radius:       %.2f cm%n", balloon.getRadius());
        System.out.printf("  Surface Area: %.2f cm²%n", balloon.getSurfaceArea());
        System.out.println();

        // ---- P4.9: Giving Change ----
        System.out.println("=== P4.9: Giving Change ===");

        System.out.print("Enter amount due (in pennies): ");
        int due = in.nextInt();
        System.out.print("Enter amount received (in pennies): ");
        int received = in.nextInt();

        int change = received - due;
        System.out.println("Change to give back: " + change + " cents");

        // Break change into bills and coins
        int dollars = change / 100;
        change = change % 100;

        int quarters = change / 25;
        change = change % 25;

        int dimes = change / 10;
        change = change % 10;

        int nickels = change / 5;
        int pennies = change % 5;

        System.out.println("  Dollars:  " + dollars);
        System.out.println("  Quarters: " + quarters);
        System.out.println("  Dimes:    " + dimes);
        System.out.println("  Nickels:  " + nickels);
        System.out.println("  Pennies:  " + pennies);
        System.out.println();

        // ---- P4.13: Dew Point Calculator ----
        System.out.println("=== P4.13: Dew Point Calculator ===");

        System.out.print("Enter relative humidity (0 to 1): ");
        double rh = in.nextDouble();
        System.out.print("Enter temperature (°C): ");
        double temp = in.nextDouble();

        // Constants for the dew point formula
        double constA = 17.27;
        double constB = 237.7;

        // f(T, RH) = (a * T) / (b + T) + ln(RH)
        double f = (constA * temp) / (constB + temp) + Math.log(rh);

        // Td = (b * f) / (a - f)
        double dewPoint = (constB * f) / (constA - f);

        System.out.printf("Dew point: %.2f °C%n", dewPoint);

        in.close();
    }
}
