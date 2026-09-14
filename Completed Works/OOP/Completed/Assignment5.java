package Completed;

import java.util.Scanner;
import java.util.ArrayList;

public class Assignment5 {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        System.out.println("Select a program:");
        System.out.println("1 - E6.10  Reverse a Word");
        System.out.println("2 - E6.14  Binary Digits");
        System.out.println("3 - E7.10  Array Methods");
        System.out.println("4 - E7.22  Sequence Append");
        int choice = in.nextInt();

        if (choice == 1) {
            reverseWord(in);
        } else if (choice == 2) {
            binaryDigits(in);
        } else if (choice == 3) {
            testArrayMethods();
        } else if (choice == 4) {
            testSequenceAppend();
        } else {
            System.out.println("Invalid choice.");
        }
    }

    // E6.10 - Reads a word and prints it reversed
    public static void reverseWord(Scanner in) {
        System.out.print("Enter a word: ");
        String word = in.next();
        String reversed = "";
        for (int i = word.length() - 1; i >= 0; i--) {
            reversed += word.charAt(i);
        }
        System.out.println(reversed);
    }

    // E6.14 - Reads a number and prints its binary digits
    public static void binaryDigits(Scanner in) {
        System.out.print("Enter a number: ");
        int number = in.nextInt();
        while (number > 0) {
            System.out.println(number % 2);
            number = number / 2;
        }
    }

    // E7.10 - Test all array methods (a through j)
    public static void testArrayMethods() {
        System.out.println("=== E7.10 Array Methods Tests ===\n");

        ArrayMethods am = new ArrayMethods(new int[] { 1, 4, 9, 16, 25 });
        am.swapFirstAndLast();
        System.out.println("a. swapFirstAndLast:          " + am);

        am = new ArrayMethods(new int[] { 1, 4, 9, 16, 25 });
        am.shiftRight();
        System.out.println("b. shiftRight:                " + am);

        am = new ArrayMethods(new int[] { 1, 4, 9, 16, 25 });
        am.replaceEvenWithZero();
        System.out.println("c. replaceEvenWithZero:       " + am);

        am = new ArrayMethods(new int[] { 1, 4, 9, 16, 25 });
        am.replaceWithLargerNeighbor();
        System.out.println("d. replaceWithLargerNeighbor: " + am);

        am = new ArrayMethods(new int[] { 1, 4, 9, 16, 25 });
        am.removeMiddle();
        System.out.println("e. removeMiddle (odd):        " + am);
        am = new ArrayMethods(new int[] { 1, 4, 9, 16 });
        am.removeMiddle();
        System.out.println("   removeMiddle (even):       " + am);

        am = new ArrayMethods(new int[] { 1, 4, 9, 16, 25 });
        am.moveEvensToFront();
        System.out.println("f. moveEvensToFront:          " + am);
        am = new ArrayMethods(new int[] { 1, 4, 9, 16, 25 });
        System.out.println("g. secondLargest:             " + am.secondLargest());
        am = new ArrayMethods(new int[] { 1, 4, 9, 16, 25 });
        System.out.println("h. isSorted (sorted):         " + am.isSorted());
        am = new ArrayMethods(new int[] { 1, 9, 4, 16, 25 });
        System.out.println("   isSorted (unsorted):       " + am.isSorted());
        am = new ArrayMethods(new int[] { 1, 4, 4, 16, 25 });
        System.out.println("i. hasAdjacentDup (yes):      " + am.hasAdjacentDuplicate());
        am = new ArrayMethods(new int[] { 1, 4, 9, 16, 25 });
        System.out.println("   hasAdjacentDup (no):       " + am.hasAdjacentDuplicate());
        am = new ArrayMethods(new int[] { 1, 4, 9, 1, 25 });
        System.out.println("j. hasDuplicate (yes):        " + am.hasDuplicate());
        am = new ArrayMethods(new int[] { 1, 4, 9, 16, 25 });
        System.out.println("   hasDuplicate (no):         " + am.hasDuplicate());
    }

    // E7.22 - Test Sequence append
    public static void testSequenceAppend() {
        System.out.println("=== E7.22 Sequence Append Test ===\n");

        Sequence a = new Sequence();
        a.add(1);
        a.add(4);
        a.add(9);
        a.add(16);

        Sequence b = new Sequence();
        b.add(9);
        b.add(7);
        b.add(4);
        b.add(9);
        b.add(11);

        Sequence result = a.append(b);

        System.out.println("a:           " + a);
        System.out.println("b:           " + b);
        System.out.println("a.append(b): " + result);
    }
}

// E7.10 - Class with array utility methods
class ArrayMethods {
    private int[] values;

    public ArrayMethods(int[] initialValues) {
        values = initialValues;
    }

    public void swapFirstAndLast() {
        if (values.length < 2)
            return;
        int temp = values[0];
        values[0] = values[values.length - 1];
        values[values.length - 1] = temp;
    }

    public void shiftRight() {
        if (values.length < 2)
            return;
        int last = values[values.length - 1];
        for (int i = values.length - 1; i > 0; i--) {
            values[i] = values[i - 1];
        }
        values[0] = last;
    }

    public void replaceEvenWithZero() {
        for (int i = 0; i < values.length; i++) {
            if (values[i] % 2 == 0) {
                values[i] = 0;
            }
        }
    }

    public void replaceWithLargerNeighbor() {
        if (values.length < 3)
            return;
        int[] copy = java.util.Arrays.copyOf(values, values.length);
        for (int i = 1; i < values.length - 1; i++) {
            values[i] = Math.max(copy[i - 1], copy[i + 1]);
        }
    }

    public void removeMiddle() {
        int len = values.length;
        if (len == 0)
            return;
        int[] result;
        if (len % 2 == 1) {
            result = new int[len - 1];
            int mid = len / 2;
            int idx = 0;
            for (int i = 0; i < len; i++) {
                if (i != mid) {
                    result[idx++] = values[i];
                }
            }
        } else {
            // Even length: remove two middle elements
            result = new int[len - 2];
            int mid1 = len / 2 - 1;
            int mid2 = len / 2;
            int idx = 0;
            for (int i = 0; i < len; i++) {
                if (i != mid1 && i != mid2) {
                    result[idx++] = values[i];
                }
            }
        }
        values = result;
    }

    // f. Move all even elements to the front, preserving order
    public void moveEvensToFront() {
        int[] result = new int[values.length];
        int idx = 0;
        // First pass: collect evens
        for (int v : values) {
            if (v % 2 == 0) {
                result[idx++] = v;
            }
        }
        // Second pass: collect odds
        for (int v : values) {
            if (v % 2 != 0) {
                result[idx++] = v;
            }
        }
        values = result;
    }

    // g. Return the second-largest element
    public int secondLargest() {
        int max = Integer.MIN_VALUE;
        int secondMax = Integer.MIN_VALUE;
        for (int v : values) {
            if (v > max) {
                secondMax = max;
                max = v;
            } else if (v > secondMax) {
                secondMax = v;
            }
        }
        return secondMax;
    }

    // h. Return true if the array is sorted in increasing order
    public boolean isSorted() {
        for (int i = 1; i < values.length; i++) {
            if (values[i] < values[i - 1]) {
                return false;
            }
        }
        return true;
    }

    // i. Return true if the array has two adjacent duplicate elements
    public boolean hasAdjacentDuplicate() {
        for (int i = 1; i < values.length; i++) {
            if (values[i] == values[i - 1]) {
                return true;
            }
        }
        return false;
    }

    // j. Return true if the array has any duplicate elements
    public boolean hasDuplicate() {
        for (int i = 0; i < values.length; i++) {
            for (int j = i + 1; j < values.length; j++) {
                if (values[i] == values[j]) {
                    return true;
                }
            }
        }
        return false;
    }

    // String representation for printing
    public String toString() {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < values.length; i++) {
            if (i > 0)
                sb.append(", ");
            sb.append(values[i]);
        }
        sb.append("]");
        return sb.toString();
    }
}

// E7.22 - Sequence class with append method
class Sequence {
    private ArrayList<Integer> values;

    public Sequence() {
        values = new ArrayList<Integer>();
    }

    // Add a number to the sequence
    public void add(int n) {
        values.add(n);
    }

    // Create a new sequence by appending other to this, without modifying either
    public Sequence append(Sequence other) {
        Sequence result = new Sequence();
        for (int v : this.values) {
            result.add(v);
        }
        for (int v : other.values) {
            result.add(v);
        }
        return result;
    }

    public String toString() {
        return values.toString();
    }
}
