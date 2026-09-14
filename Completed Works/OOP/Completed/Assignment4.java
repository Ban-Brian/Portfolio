package Completed;

import java.util.Scanner;

/**
 * Assignment 4 — Exercises E5.9, E5.18, P5.13, P5.18
 *
 * @author Brian Butler
 */
public class Assignment4 {

    public static void main(String[] args) {

        Scanner in = new Scanner(System.in);

        // ---- E5.9: Compass Direction ----
        System.out.println("=== E5.9: Compass Direction ===");
        System.out.print("Enter angle (degrees from North, clockwise): ");
        double angle = in.nextDouble();

        angle = ((angle % 360) + 360) % 360;

        String direction;
        if (angle <= 22.5 || angle > 337.5) {
            direction = "N";
        } else if (angle <= 67.5) {
            direction = "NE";
        } else if (angle <= 112.5) {
            direction = "E";
        } else if (angle <= 157.5) {
            direction = "SE";
        } else if (angle <= 202.5) {
            direction = "S";
        } else if (angle <= 247.5) {
            direction = "SW";
        } else if (angle <= 292.5) {
            direction = "W";
        } else {
            direction = "NW";
        }

        System.out.println("Direction: " + direction);
        System.out.println();

        // ---- E5.18: 1913 Income Tax ----
        System.out.println("=== E5.18: 1913 Income Tax ===");
        System.out.print("Enter income: ");
        double income = in.nextDouble();

        double tax = 0;

        if (income > 500000) {
            tax += (income - 500000) * 0.06;
            income = 500000;
        }
        if (income > 250000) {
            tax += (income - 250000) * 0.05;
            income = 250000;
        }
        if (income > 100000) {
            tax += (income - 100000) * 0.04;
            income = 100000;
        }
        if (income > 75000) {
            tax += (income - 75000) * 0.03;
            income = 75000;
        }
        if (income > 50000) {
            tax += (income - 50000) * 0.02;
            income = 50000;
        }
        tax += income * 0.01;

        System.out.printf("Tax: $%.2f%n", tax);
        System.out.println();

        // ---- P5.13: Restaurant Tip ----
        System.out.println("=== P5.13: Restaurant Tip ===");
        System.out.print("Enter meal cost: ");
        double meal = in.nextDouble();
        System.out.print("Satisfaction (1 = Totally satisfied, 2 = Satisfied, 3 = Dissatisfied): ");
        int satisfaction = in.nextInt();

        double tipPercent;
        String level;

        if (satisfaction == 1) {
            tipPercent = 20;
            level = "Totally satisfied";
        } else if (satisfaction == 2) {
            tipPercent = 15;
            level = "Satisfied";
        } else {
            tipPercent = 10;
            level = "Dissatisfied";
        }

        double tip = meal * tipPercent / 100;

        System.out.println("Satisfaction level: " + level);
        System.out.printf("Tip (%.0f%%): $%.2f%n", tipPercent, tip);
        System.out.println();

        // ---- P5.18: Sound Level Description ----
        System.out.println("=== P5.18: Sound Level ===");
        System.out.print("Enter value: ");
        double value = in.nextDouble();
        System.out.print("Enter unit (dB or Pa): ");
        String unit = in.next();

        double dB;
        double p0 = 20e-6;

        if (unit.equals("Pa")) {
            dB = 20 * Math.log10(value / p0);
        } else {
            dB = value;
        }

        int[] levels = { 0, 30, 60, 90, 100, 120, 130 };
        String[] descriptions = {
                "Light leaf rustling",
                "Calm library",
                "Normal conversation",
                "Traffic on a busy roadway at 10 m",
                "Jack hammer at 1 m",
                "Possible hearing damage",
                "Threshold of pain"
        };

        int closest = 0;
        double smallestDiff = Math.abs(dB - levels[0]);

        for (int i = 1; i < levels.length; i++) {
            double diff = Math.abs(dB - levels[i]);
            if (diff < smallestDiff) {
                smallestDiff = diff;
                closest = i;
            }
        }

        System.out.printf("Sound level: %.1f dB%n", dB);
        System.out.println("Description: " + descriptions[closest]);

        in.close();
    }
}
