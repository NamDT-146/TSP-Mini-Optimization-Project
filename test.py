import unittest
import time
from io import StringIO
import sys
from utils import read_input, parse_sample_answer, validate_constraints


class TestOptimizationProgram(unittest.TestCase):
    def run_program(self, input_file):
        """
        Simulate running the program with a given input file.
        Replace this function with your program's entry point.
        """
        original_stdin = sys.stdin
        sys.stdin = open(input_file, 'r')  # Redirect input from the file
        original_stdout = sys.stdout
        output_buffer = StringIO()
        sys.stdout = output_buffer  # Redirect output to a string buffer
        try:
            from tsp_ortool_2 import main  # Replace with your program's entry point
            main(input_file)
        finally:
            sys.stdin = original_stdin
            sys.stdout = original_stdout
        return output_buffer.getvalue()

    def test_with_sample(self):
        # Define test cases and expected results
        test_cases = [
            ("test_case_10.txt", "sample_answer10.txt", 1),  # 1-second time limit
            ("valid_solvable_test_case_20.txt", "sample_answer20.txt", 10)  # 2-second time limit
            
        ]

        for test_case, sample_answer, time_limit in test_cases:
            with self.subTest(test_case=test_case, sample_answer=sample_answer):
                # Parse test case and sample answer
                N, delivery_constraints, travel_times = read_input(from_file=True, file_path=test_case)
                _, expected_route, expected_time = parse_sample_answer(sample_answer)

                # Run the program and measure execution time
                start_time = time.time()
                program_output = self.run_program(test_case)
                execution_time = time.time() - start_time

                # Parse program output
                output_lines = program_output.strip().split("\n")
                program_route = list(map(int, output_lines[1].strip().split()))
                program_time = int(output_lines[2].strip())

                # Validate results
                constraints_satisfied = validate_constraints(
                    program_route, delivery_constraints, travel_times
                )
                travel_time_valid = (program_time == expected_time)

                # Check execution time
                execution_time_valid = execution_time <= time_limit

                # Scoring
                if constraints_satisfied and travel_time_valid and execution_time_valid:
                    score = 100
                else:
                    score = 0

                # Print Results
                print(f"Test Case: {test_case}")
                print(f"Program Route: {program_route}, Expected Route: {expected_route}")
                print(f"Program Time: {program_time}, Expected Time: {expected_time}")
                print(f"Execution Time: {execution_time:.2f}s (Limit: {time_limit}s)")
                print(f"Score: {score}/100\n")

                # Assertions
                self.assertTrue(constraints_satisfied, "Constraints not satisfied.")
                self.assertTrue(travel_time_valid, "Travel time not optimal.")
                self.assertTrue(execution_time_valid, f"Execution time exceeded {time_limit}s.")


if __name__ == "__main__":
    unittest.main()
