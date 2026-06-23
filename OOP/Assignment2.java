/**
 * Assignment 2 — Lab 1, Question P2.8
 *
 * @author Brian Butler
 */
public class Assignment2 {

    public static void main(String[] args) {

        // Starting year
        int startYear = 2026;

        // Loop through this year and the next 3 years
        for (int i = 0; i < 4; i++) {

            int year = startYear + i;

            // Create a Day object for Feb 28 of this year
            Day date = new Day(year, 2, 28);

            // Move forward one day
            Day nextDay = date.addDays(1);

            // Print the result
            System.out.println("Feb 28, " + year + " + 1 day = " + nextDay);

            // Print what we expect:
            // Leap years give Feb 29
            // Non-leap years give Mar 1
            if (year % 4 == 0) {
                System.out.println("  Expected: Feb 29 (leap year)");
            } else {
                System.out.println("  Expected: Mar 1 (not a leap year)");
            }

            System.out.println();
        }
    }
}
