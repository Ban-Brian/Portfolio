import java.util.Scanner;
import java.util.Random;
import java.util.ArrayList;
import java.util.Collections;

public class Lab3 {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        System.out.println("Select a program:");
        System.out.println("1 - P6.6  Game of Nim");
        System.out.println("2 - P6.15 Projectile Flight");
        System.out.println("3 - P7.9  Video Poker");
        int choice = in.nextInt();

        if (choice == 1) {
            nim(in);
        } else if (choice == 2) {
            projectile(in);
        } else if (choice == 3) {
            videoPoker(in);
        } else {
            System.out.println("Invalid choice.");
        }
    }

    // P6.6 - Game of Nim: two players take turn removing 1 to n/2 marbles; last
    // marble loses
    public static void nim(Scanner in) {
        Random rand = new Random();
        int pile = rand.nextInt(91) + 10; // 10-100
        boolean computerTurn = rand.nextInt(2) == 0;
        boolean smart = rand.nextInt(2) == 0;

        System.out.println("Pile size: " + pile);
        System.out.println("Computer plays " + (smart ? "smart" : "stupid") + " mode.");
        System.out.println((computerTurn ? "Computer" : "You") + " go first.\n");

        while (pile > 0) {
            if (computerTurn) {
                int take;
                if (smart) {
                    // Try to leave pile at (power of 2) - 1
                    take = smartMove(pile, rand);
                } else {
                    // Stupid mode: random legal move
                    take = rand.nextInt(Math.max(1, pile / 2)) + 1;
                }
                pile -= take;
                System.out.println("Computer takes " + take + ". Pile: " + pile);
                if (pile == 0) {
                    System.out.println("Computer took the last marble. You win!");
                    return;
                }
            } else {
                int maxTake = Math.max(1, pile / 2);
                System.out.println("Pile has " + pile + " marbles. You may take 1 to " + maxTake + ".");
                System.out.print("How many do you take? ");
                int take = in.nextInt();
                while (take < 1 || take > maxTake) {
                    System.out.print("Invalid. Take 1 to " + maxTake + ": ");
                    take = in.nextInt();
                }
                pile -= take;
                System.out.println("You take " + take + ". Pile: " + pile);
                if (pile == 0) {
                    System.out.println("You took the last marble. Computer wins!");
                    return;
                }
            }
            computerTurn = !computerTurn;
        }
    }

    // Smart mode: leave pile at (2^k - 1); if already there, take random
    private static int smartMove(int pile, Random rand) {
        int[] targets = { 63, 31, 15, 7, 3, 1 };
        for (int t : targets) {
            int take = pile - t;
            if (take >= 1 && take <= pile / 2) {
                return take;
            }
        }
        // No good move, take random legal amount
        return rand.nextInt(Math.max(1, pile / 2)) + 1;
    }

    // P6.15 - Projectile Flight: simulate cannonball and compare with exact formula
    public static void projectile(Scanner in) {
        final double G = 9.81;
        final double DELTA_T = 0.01;

        System.out.print("Enter initial velocity (m/s): ");
        double v0 = in.nextDouble();

        double s = 0;
        double v = v0;
        double t = 0;
        int nextSecond = 1;

        System.out.printf("%-10s %-20s %-20s%n", "Time(s)", "Simulated(m)", "Exact(m)");
        System.out.printf("%-10d %-20.4f %-20.4f%n", 0, 0.0, 0.0);

        // Run simulation until ball returns to ground
        while (s >= 0 || t < 0.1) {
            s = s + v * DELTA_T;
            v = v - G * DELTA_T;
            t = t + DELTA_T;

            // Print at each full second
            if (t >= nextSecond - DELTA_T / 2 && t < nextSecond + DELTA_T / 2) {
                double exact = -0.5 * G * nextSecond * nextSecond + v0 * nextSecond;
                System.out.printf("%-10d %-20.4f %-20.4f%n", nextSecond, s, exact);
                nextSecond++;
            }

            if (s < 0) {
                break;
            }
        }
        System.out.println("The ball has hit the ground.");
    }

    // P7.9 - Video Poker: deal 5 cards, allow replacement, score hand
    public static void videoPoker(Scanner in) {
        int tokens = 10;
        System.out.println("Welcome to Video Poker! You start with " + tokens + " tokens.");

        while (tokens > 0) {
            System.out.println("\nTokens: " + tokens);
            System.out.print("Play a round? (yes/no): ");
            String answer = in.next();
            if (answer.equalsIgnoreCase("no")) {
                break;
            }

            tokens--; // pay to play

            // Build and shuffle deck
            ArrayList<String> deck = buildDeck();
            Collections.shuffle(deck);
            int deckIndex = 0;

            // Deal 5 cards
            String[] hand = new String[5];
            for (int i = 0; i < 5; i++) {
                hand[i] = deck.get(deckIndex++);
            }

            System.out.println("Your hand:");
            for (int i = 0; i < 5; i++) {
                System.out.println("  " + (i + 1) + ": " + hand[i]);
            }

            // Let player reject cards
            System.out.println("Enter card numbers to replace (e.g. 1 3 5), or 0 to keep all:");
            String line = in.nextLine(); // consume leftover newline
            line = in.nextLine();

            if (!line.trim().equals("0") && !line.trim().isEmpty()) {
                String[] parts = line.trim().split("\\s+");
                for (String p : parts) {
                    int idx = Integer.parseInt(p) - 1;
                    if (idx >= 0 && idx < 5) {
                        hand[idx] = deck.get(deckIndex++);
                    }
                }
            }

            System.out.println("Final hand:");
            for (int i = 0; i < 5; i++) {
                System.out.println("  " + (i + 1) + ": " + hand[i]);
            }

            // Score the hand
            int payout = scoreHand(hand);
            tokens += payout;
        }

        System.out.println("Game over. Final tokens: " + tokens);
    }

    // Build a standard 52-card deck
    private static ArrayList<String> buildDeck() {
        String[] suits = { "Hearts", "Diamonds", "Clubs", "Spades" };
        String[] ranks = { "2", "3", "4", "5", "6", "7", "8", "9", "10",
                "Jack", "Queen", "King", "Ace" };
        ArrayList<String> deck = new ArrayList<>();
        for (String s : suits) {
            for (String r : ranks) {
                deck.add(r + " of " + s);
            }
        }
        return deck;
    }

    // Get numeric value of a card rank (2-14, Ace=14)
    private static int rankValue(String card) {
        String rank = card.split(" of ")[0];
        switch (rank) {
            case "Jack":
                return 11;
            case "Queen":
                return 12;
            case "King":
                return 13;
            case "Ace":
                return 14;
            default:
                return Integer.parseInt(rank);
        }
    }

    // Get suit of a card
    private static String suit(String card) {
        return card.split(" of ")[1];
    }

    // Score a 5-card poker hand and print the result
    private static int scoreHand(String[] hand) {
        int[] values = new int[5];
        String[] suits = new String[5];
        for (int i = 0; i < 5; i++) {
            values[i] = rankValue(hand[i]);
            suits[i] = suit(hand[i]);
        }

        // Sort values
        java.util.Arrays.sort(values);

        // Check flush
        boolean flush = true;
        for (int i = 1; i < 5; i++) {
            if (!suits[i].equals(suits[0])) {
                flush = false;
                break;
            }
        }

        // Check straight (including ace-low: A,2,3,4,5)
        boolean straight = true;
        for (int i = 1; i < 5; i++) {
            if (values[i] != values[i - 1] + 1) {
                straight = false;
                break;
            }
        }
        // Ace-low straight: A,2,3,4,5
        boolean aceLow = (values[0] == 2 && values[1] == 3 && values[2] == 4
                && values[3] == 5 && values[4] == 14);
        if (aceLow) {
            straight = true;
        }

        // Count matching ranks
        int[] rankCount = new int[15]; // index 2-14
        for (int v : values) {
            rankCount[v]++;
        }

        int pairs = 0, threes = 0, fours = 0;
        for (int c : rankCount) {
            if (c == 2)
                pairs++;
            if (c == 3)
                threes++;
            if (c == 4)
                fours++;
        }

        // Royal flush: 10,J,Q,K,A all same suit
        if (flush && straight && values[0] == 10) {
            System.out.println("*** ROYAL FLUSH! ***");
            return 250;
        }
        if (flush && straight) {
            System.out.println("*** Straight Flush! ***");
            return 50;
        }
        if (fours == 1) {
            System.out.println("Four of a Kind!");
            return 25;
        }
        if (threes == 1 && pairs == 1) {
            System.out.println("Full House!");
            return 6;
        }
        if (flush) {
            System.out.println("Flush!");
            return 5;
        }
        if (straight) {
            System.out.println("Straight!");
            return 4;
        }
        if (threes == 1) {
            System.out.println("Three of a Kind!");
            return 3;
        }
        if (pairs == 2) {
            System.out.println("Two Pairs!");
            return 2;
        }
        if (pairs == 1) {
            System.out.println("One Pair!");
            return 1;
        }

        System.out.println("No pair. Better luck next time.");
        return 0;
    }
}
