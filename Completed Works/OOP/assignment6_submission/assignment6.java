import java.util.Scanner;
import java.util.ArrayList;
import java.util.Random;
import java.awt.Point;

/**
 * Assignment 6 - E8.26 Robot, P8.1 ComboLock, P8.3 Car Sharing Simulation.
 * Menu-driven program that lets the user select which exercise to run.
 */
public class assignment6 {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        System.out.println("Select a program:");
        System.out.println("1 - E8.26  Robot");
        System.out.println("2 - P8.1   ComboLock");
        System.out.println("3 - P8.3   Car Sharing Simulation");
        int choice = in.nextInt();

        if (choice == 1) {
            testRobot();
        } else if (choice == 2) {
            testComboLock();
        } else if (choice == 3) {
            testCarSharing();
        } else {
            System.out.println("Invalid choice.");
        }
    }

    /** E8.26 - Demonstrates the Robot class by moving and turning. */
    public static void testRobot() {
        Robot robot = new Robot();
        System.out.println("=== E8.26 Robot Test ===\n");

        System.out.println("Start:  " + robot.getDirection() + " at " + robot.getLocation());

        robot.move();
        System.out.println("Move:   " + robot.getDirection() + " at " + robot.getLocation());

        robot.turnRight();
        System.out.println("Right:  " + robot.getDirection() + " at " + robot.getLocation());

        robot.move();
        System.out.println("Move:   " + robot.getDirection() + " at " + robot.getLocation());

        robot.move();
        System.out.println("Move:   " + robot.getDirection() + " at " + robot.getLocation());

        robot.turnLeft();
        System.out.println("Left:   " + robot.getDirection() + " at " + robot.getLocation());

        robot.move();
        System.out.println("Move:   " + robot.getDirection() + " at " + robot.getLocation());

        // Walk a full square to return to origin
        System.out.println("\n--- Walking a square ---");
        Robot r2 = new Robot();
        for (int side = 0; side < 4; side++) {
            for (int step = 0; step < 3; step++) {
                r2.move();
            }
            r2.turnRight();
        }
        System.out.println("After square: " + r2.getDirection() + " at " + r2.getLocation());
    }

    /** P8.1 - Demonstrates the ComboLock class with correct and wrong combos. */
    public static void testComboLock() {
        System.out.println("=== P8.1 ComboLock Test ===\n");

        // Test correct combo (10, 20, 30): right to 10, left to 20, right to 30
        ComboLock lock = new ComboLock(10, 20, 30);
        lock.reset();
        lock.turnRight(10); // dial lands on 10
        lock.turnLeft(30); // dial lands on (10 - 30 + 40) % 40 = 20
        lock.turnRight(10); // dial lands on (20 + 10) % 40 = 30
        System.out.println("Correct combo (10,20,30): " + lock.open());

        // Test wrong combo
        lock.reset();
        lock.turnRight(10);
        lock.turnLeft(5);
        lock.turnRight(10);
        System.out.println("Wrong combo:              " + lock.open());

        // Test combo (0, 0, 0)
        ComboLock lock2 = new ComboLock(0, 0, 0);
        lock2.reset();
        lock2.turnRight(0);
        lock2.turnLeft(0);
        lock2.turnRight(0);
        System.out.println("Combo (0,0,0) correct:    " + lock2.open());
    }

    /** P8.3 - Runs the car sharing simulation. */
    public static void testCarSharing() {
        System.out.println("=== P8.3 Car Sharing Simulation ===\n");
        Simulation sim = new Simulation();
        sim.run();
    }
}

/**
 * E8.26 - Simulates a robot wandering on an infinite plane.
 * The robot starts at the origin facing north and can turn or move forward.
 */
class Robot {
    private int x;
    private int y;
    private int direction; // 0=N, 1=E, 2=S, 3=W

    /** Creates a robot at the origin facing north. */
    public Robot() {
        x = 0;
        y = 0;
        direction = 0;
    }

    /** Turns the robot 90 degrees to the left. */
    public void turnLeft() {
        direction = (direction + 3) % 4;
    }

    /** Turns the robot 90 degrees to the right. */
    public void turnRight() {
        direction = (direction + 1) % 4;
    }

    /** Moves the robot one unit in the direction it faces. */
    public void move() {
        if (direction == 0) {
            y++;
        } else if (direction == 1) {
            x++;
        } else if (direction == 2) {
            y--;
        } else {
            x--;
        }
    }

    /**
     * Returns the robot's current position.
     * 
     * @return a Point with the robot's x and y coordinates
     */
    public Point getLocation() {
        return new Point(x, y);
    }

    /**
     * Returns the direction as "N", "E", "S", or "W".
     * 
     * @return the current direction string
     */
    public String getDirection() {
        String[] dirs = { "N", "E", "S", "W" };
        return dirs[direction];
    }
}

/**
 * P8.1 - Simulates a gym locker combination lock with 40 positions (0-39).
 * The user must turn right to the first number, left to the second, and right
 * to the third.
 */
class ComboLock {
    private int secret1;
    private int secret2;
    private int secret3;
    private int dialPosition;
    private int turnCount;
    private boolean failed;

    /**
     * Creates a lock with the given three-number combination.
     * 
     * @param secret1 the first number (0-39)
     * @param secret2 the second number (0-39)
     * @param secret3 the third number (0-39)
     */
    public ComboLock(int secret1, int secret2, int secret3) {
        this.secret1 = secret1;
        this.secret2 = secret2;
        this.secret3 = secret3;
        reset();
    }

    /** Resets the dial to 0 and clears all turn history. */
    public void reset() {
        dialPosition = 0;
        turnCount = 0;
        failed = false;
    }

    /**
     * Turns the dial left (counterclockwise) by the given number of ticks.
     * 
     * @param ticks the number of positions to turn left
     */
    public void turnLeft(int ticks) {
        dialPosition = (dialPosition - ticks % 40 + 40) % 40;
        turnCount++;
        // The second turn should be left to the second number
        if (turnCount == 2) {
            if (dialPosition != secret2) {
                failed = true;
            }
        } else {
            failed = true;
        }
    }

    /**
     * Turns the dial right (clockwise) by the given number of ticks.
     * 
     * @param ticks the number of positions to turn right
     */
    public void turnRight(int ticks) {
        dialPosition = (dialPosition + ticks) % 40;
        turnCount++;
        // The first turn should be right to the first number
        if (turnCount == 1) {
            if (dialPosition != secret1) {
                failed = true;
            }
        }
        // The third turn should be right to the third number
        else if (turnCount == 3) {
            if (dialPosition != secret3) {
                failed = true;
            }
        } else {
            failed = true;
        }
    }

    /**
     * Attempts to open the lock.
     * 
     * @return true if the correct right-left-right sequence was entered
     */
    public boolean open() {
        return !failed && turnCount == 3;
    }
}

/**
 * P8.3 - Represents a passenger with a target station.
 */
class Passenger {
    private int targetStation;

    /**
     * Creates a passenger heading to the given station.
     * 
     * @param target the destination station number
     */
    public Passenger(int target) {
        targetStation = target;
    }

    /**
     * Returns the passenger's destination station.
     * 
     * @return the target station number
     */
    public int getTarget() {
        return targetStation;
    }
}

/**
 * P8.3 - Represents a car with a destination that picks up and drops off
 * passengers.
 */
class Car {
    private int currentStation;
    private int targetStation;
    private ArrayList<Passenger> passengers;
    private double revenue;

    /**
     * Creates a car at the given station heading to the given destination.
     * 
     * @param currentStation the starting station
     * @param targetStation  the destination station
     */
    public Car(int currentStation, int targetStation) {
        this.currentStation = currentStation;
        this.targetStation = targetStation;
        passengers = new ArrayList<Passenger>();
        revenue = 0;
    }

    /**
     * Picks up a passenger if their destination is on the way and car has room (max
     * 3).
     * 
     * @param p the passenger to pick up
     * @return true if the passenger was picked up
     */
    public boolean pickUp(Passenger p) {
        if (passengers.size() >= 3) {
            return false;
        }
        if (isOnTheWay(p.getTarget())) {
            passengers.add(p);
            return true;
        }
        return false;
    }

    // Checks if a station is between the car's current location and destination
    private boolean isOnTheWay(int station) {
        if (currentStation < targetStation) {
            return station > currentStation && station <= targetStation;
        } else {
            return station < currentStation && station >= targetStation;
        }
    }

    /**
     * Drives the car to its destination, dropping off and picking up at each
     * station.
     * 
     * @param stations the array of all stations along the route
     */
    public void drive(Station[] stations) {
        int step = (targetStation > currentStation) ? 1 : -1;

        while (currentStation != targetStation) {
            currentStation += step;

            // Drop off passengers at this station
            ArrayList<Passenger> staying = new ArrayList<Passenger>();
            for (Passenger p : passengers) {
                if (p.getTarget() == currentStation) {
                    // Passenger exits here
                } else {
                    staying.add(p);
                }
            }
            // Count dropped passengers — they also rode this mile segment
            int dropped = passengers.size() - staying.size();
            passengers = staying;

            // Revenue: each passenger earns $1 per mile for this segment
            revenue += passengers.size();
            revenue += dropped;

            // Try to pick up new passengers at this station if there's room
            if (currentStation >= 0 && currentStation < stations.length) {
                ArrayList<Passenger> waiting = stations[currentStation].getWaitingPassengers();
                ArrayList<Passenger> stillWaiting = new ArrayList<Passenger>();
                for (Passenger p : waiting) {
                    if (passengers.size() < 3 && isOnTheWay(p.getTarget())) {
                        passengers.add(p);
                    } else {
                        stillWaiting.add(p);
                    }
                }
                stations[currentStation].setWaitingPassengers(stillWaiting);
            }
        }
    }

    /**
     * Returns the total revenue earned by this car.
     * 
     * @return the revenue in dollars
     */
    public double getRevenue() {
        return revenue;
    }

    /**
     * Returns the total miles this car travels.
     * 
     * @return the distance in miles
     */
    public int getTotalMiles() {
        return Math.abs(targetStation - currentStation);
    }

    /** @return the car's current station */
    public int getCurrentStation() {
        return currentStation;
    }

    /** @return the car's target station */
    public int getTargetStation() {
        return targetStation;
    }
}

/**
 * P8.3 - Represents a station with waiting passengers.
 */
class Station {
    private ArrayList<Passenger> waitingPassengers;

    /** Creates an empty station. */
    public Station() {
        waitingPassengers = new ArrayList<Passenger>();
    }

    /**
     * Adds a passenger to the station's waiting list.
     * 
     * @param p the passenger to add
     */
    public void addPassenger(Passenger p) {
        waitingPassengers.add(p);
    }

    /**
     * Returns the list of passengers waiting at this station.
     * 
     * @return the waiting passengers list
     */
    public ArrayList<Passenger> getWaitingPassengers() {
        return waitingPassengers;
    }

    /**
     * Replaces the waiting list after some passengers are picked up.
     * 
     * @param list the updated passenger list
     */
    public void setWaitingPassengers(ArrayList<Passenger> list) {
        waitingPassengers = list;
    }
}

/**
 * P8.3 - Runs 1000 car sharing simulations and reports average revenue per
 * mile.
 */
class Simulation {
    private static final int NUM_STATIONS = 30;
    private static final int NUM_RUNS = 1000;
    private Random rand;

    /** Creates a new simulation. */
    public Simulation() {
        rand = new Random();
    }

    /** Runs the full simulation and prints the average revenue per mile. */
    public void run() {
        double totalRevenue = 0;
        double totalMiles = 0;

        for (int run = 0; run < NUM_RUNS; run++) {
            // Create 30 stations
            Station[] stations = new Station[NUM_STATIONS];
            for (int i = 0; i < NUM_STATIONS; i++) {
                stations[i] = new Station();
            }

            // Generate random passengers at each station (0 to 5 per station)
            for (int i = 0; i < NUM_STATIONS; i++) {
                int numPassengers = rand.nextInt(6);
                for (int j = 0; j < numPassengers; j++) {
                    int target;
                    do {
                        target = rand.nextInt(NUM_STATIONS);
                    } while (target == i);
                    stations[i].addPassenger(new Passenger(target));
                }
            }

            // Generate random cars at each station (0 to 3 per station)
            ArrayList<Car> cars = new ArrayList<Car>();
            for (int i = 0; i < NUM_STATIONS; i++) {
                int numCars = rand.nextInt(4);
                for (int j = 0; j < numCars; j++) {
                    int target;
                    do {
                        target = rand.nextInt(NUM_STATIONS);
                    } while (target == i);
                    Car car = new Car(i, target);

                    // Pick up initial passengers at this station
                    ArrayList<Passenger> waiting = stations[i].getWaitingPassengers();
                    ArrayList<Passenger> stillWaiting = new ArrayList<Passenger>();
                    for (Passenger p : waiting) {
                        if (!car.pickUp(p)) {
                            stillWaiting.add(p);
                        }
                    }
                    stations[i].setWaitingPassengers(stillWaiting);

                    cars.add(car);
                }
            }

            // Drive all cars to their destinations
            double runRevenue = 0;
            double runMiles = 0;
            for (Car car : cars) {
                int miles = Math.abs(car.getTargetStation() - car.getCurrentStation());
                car.drive(stations);
                runRevenue += car.getRevenue();
                runMiles += miles;
            }

            totalRevenue += runRevenue;
            totalMiles += runMiles;
        }

        double avgRevenuePerMile = totalRevenue / totalMiles;
        System.out.println("Simulations run:          " + NUM_RUNS);
        System.out.println("Total revenue:            $" + String.format("%.2f", totalRevenue));
        System.out.println("Total miles driven:       " + String.format("%.0f", totalMiles));
        System.out.println("Average revenue per mile: $" + String.format("%.2f", avgRevenuePerMile));
    }
}
