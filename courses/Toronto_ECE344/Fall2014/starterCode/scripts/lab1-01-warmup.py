#!/usr/bin/python

import tester
import sys

def main():
    test = tester.Core('warmup test', 30)
    test.start_program('./hello')
    test.lookA('Hello world', 1)

    test.start_program('./words how many words')
    test.lookA('how', 1)
    test.lookA('many', 1)
    test.lookA('words', 1)

    test.start_program('./fact 9')
    test.lookA('362880', 1)

    test.start_program('./fact 12')
    test.lookA('479001600', 1)

    test.start_program('./fact 0')
    test.lookA('Huh\?', 1)

    test.start_program('./fact hello')
    test.lookA('Huh\?', 1)

    test.start_program('./fact 20')
    test.lookA('Overflow', 1)

    test.start_program('./fact 1.2')
    test.lookA('Huh\?', 1)

    test.start_program('./test_point')
    test.lookA('OK', 2)

    test.start_program('./test_sorted_points', 20)
    test.look('OK\r\n', 8)

    test.start_program('./run_test_wc')
    test.lookA('OK', 10)

if __name__ == '__main__':
	main()
    
