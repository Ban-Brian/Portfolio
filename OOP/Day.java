import java.time.LocalDate;
import java.time.temporal.ChronoUnit;

/**
 * Day class based on Worked Example 2.1.
 * Handles calendar dates including leap years.
 */
public class Day {

    private LocalDate date;

    // Create a Day with a given year, month, and day
    public Day(int year, int month, int day) {
        date = LocalDate.of(year, month, day);
    }

    // Create a Day set to today's date
    public Day() {
        date = LocalDate.now();
    }

    // Private constructor used internally
    private Day(LocalDate d) {
        date = d;
    }

    // Returns a new Day that is n days later
    public Day addDays(int n) {
        return new Day(date.plusDays(n));
    }

    // Returns how many days this day is from another day
    // Positive if this day comes after other
    public int daysFrom(Day other) {
        return (int) ChronoUnit.DAYS.between(other.date, this.date);
    }

    // Get the year
    public int getYear() {
        return date.getYear();
    }

    // Get the month (1-12)
    public int getMonth() {
        return date.getMonthValue();
    }

    // Get the day of the month
    public int getDate() {
        return date.getDayOfMonth();
    }

    // Returns the date as a readable string
    public String toString() {
        return date.getMonthValue() + "/" + date.getDayOfMonth() + "/" + date.getYear();
    }
}
