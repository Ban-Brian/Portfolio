package Completed;

import java.util.Scanner;

public class assignment7 {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        System.out.println("Select a program:");
        System.out.println("1 - Person, Student, Instructor");
        System.out.println("2 - Clock and WorldClock");
        System.out.println("3 - Appointment Book");
        int choice = in.nextInt();

        if (choice == 1) {
            testPerson();
        } else if (choice == 2) {
            testClock();
        } else if (choice == 3) {
            testAppointments(in);
        } else {
            System.out.println("Invalid choice.");
        }
    }

    /** Test program for Person, Student, and Instructor. */
    public static void testPerson() {
        System.out.println("=== Person, Student, Instructor Test ===");
        Person p = new Person("John Doe", 1990);
        Student s = new Student("Alice Smith", 2002, "Computer Science");
        Instructor i = new Instructor("Dr. Jones", 1975, 95000);

        System.out.println(p);
        System.out.println(s);
        System.out.println(i);
    }

    /** Test program for Clock and WorldClock. */
    public static void testClock() {
        System.out.println("=== Clock and WorldClock Test ===");
        Clock c = new Clock();
        System.out.println("Local Time (Hours): " + c.getHours());
        System.out.println("Local Time (Minutes): " + c.getMinutes());
        System.out.println("Local Time (Full): " + c.getTime());

        WorldClock wc = new WorldClock(3);
        System.out.println("WorldClock (+3 hours) Time: " + wc.getTime());

        System.out.println("\nMethod Overriding Explanation:");
        System.out.println("In WorldClock, we override the getHours() method to apply the time zone offset.");
        System.out.println("We do not override getTime(), because getTime() calls getHours() dynamically, "
                + "allowing it to use the overridden method through polymorphism.");
    }

    /** Test program for Appointments. */
    public static void testAppointments(Scanner in) {
        System.out.println("=== Appointment Book Test ===");
        Appointment[] list = {
                new Onetime("See the dentist", 2026, 7, 25),
                new Daily("Brush teeth"),
                new Monthly("Pay rent", 1),
                new Monthly("Check car oil", 15),
                new Onetime("Doctor appointment", 2026, 8, 10),
                new Daily("Read a book")
        };
        System.out.print("Enter year: ");
        int year = in.nextInt();
        System.out.print("Enter month (1-12): ");
        int month = in.nextInt();
        System.out.print("Enter day (1-31): ");
        int day = in.nextInt();

        System.out.println("\nAppointments on " + year + "-" + month + "-" + day + ":");
        boolean found = false;
        for (Appointment app : list) {
            if (app.occursOn(year, month, day)) {
                System.out.println("- " + app.getDescription() + " (" + app.getClass().getSimpleName() + ")");
                found = true;
            }
        }
        if (!found) {
            System.out.println("No appointments on this date.");
        }
    }
}

// Problem 1: Person, Student, and Instructor classes

/** Represents a person with a name and birth year. */
class Person {
    private String name;
    private int yearOfBirth;

    public Person(String name, int yearOfBirth) {
        this.name = name;
        this.yearOfBirth = yearOfBirth;
    }

    public String getName() {
        return name;
    }

    public int getYearOfBirth() {
        return yearOfBirth;
    }

    @Override
    public String toString() {
        return "Person[name=" + name + ",yearOfBirth=" + yearOfBirth + "]";
    }
}

/** Represents a student inheriting from Person. */
class Student extends Person {
    private String major;

    public Student(String name, int yearOfBirth, String major) {
        super(name, yearOfBirth);
        this.major = major;
    }

    public String getMajor() {
        return major;
    }

    @Override
    public String toString() {
        return "Student[super=" + super.toString() + ",major=" + major + "]";
    }
}

/** Represents an instructor inheriting from Person. */
class Instructor extends Person {
    private double salary;

    public Instructor(String name, int yearOfBirth, double salary) {
        super(name, yearOfBirth);
        this.salary = salary;
    }

    public double getSalary() {
        return salary;
    }

    @Override
    public String toString() {
        return "Instructor[super=" + super.toString() + ",salary=" + salary + "]";
    }
}

// Problem 2: Clock and WorldClock classes

/** Represents a clock returning local hours and minutes. */
class Clock {
    public String getHours() {
        String time = java.time.LocalTime.now().toString();
        return time.split(":")[0];
    }

    public String getMinutes() {
        String time = java.time.LocalTime.now().toString();
        return time.split(":")[1];
    }

    public String getTime() {
        return getHours() + ":" + getMinutes();
    }
}

/** Represents a clock with a time zone offset. */
class WorldClock extends Clock {
    private int offset;

    public WorldClock(int offset) {
        this.offset = offset;
    }

    @Override
    public String getHours() {
        int localHours = Integer.parseInt(super.getHours());
        int offsetHours = (localHours + offset) % 24;
        if (offsetHours < 0) {
            offsetHours += 24;
        }
        return String.format("%02d", offsetHours);
    }
}

// Problem 3: Appointment and subclasses

/** Represents a general appointment. */
class Appointment {
    private String description;

    public Appointment(String description) {
        this.description = description;
    }

    public String getDescription() {
        return description;
    }

    public boolean occursOn(int year, int month, int day) {
        return false;
    }

    @Override
    public String toString() {
        return description;
    }
}

/** Represents a single occurrence appointment. */
class Onetime extends Appointment {
    private int year;
    private int month;
    private int day;

    public Onetime(String description, int year, int month, int day) {
        super(description);
        this.year = year;
        this.month = month;
        this.day = day;
    }

    @Override
    public boolean occursOn(int year, int month, int day) {
        return this.year == year && this.month == month && this.day == day;
    }
}

/** Represents a daily appointment. */
class Daily extends Appointment {
    public Daily(String description) {
        super(description);
    }

    @Override
    public boolean occursOn(int year, int month, int day) {
        return true;
    }
}

/** Represents a monthly appointment. */
class Monthly extends Appointment {
    private int day;

    public Monthly(String description, int day) {
        super(description);
        this.day = day;
    }

    @Override
    public boolean occursOn(int year, int month, int day) {
        return this.day == day;
    }
}
