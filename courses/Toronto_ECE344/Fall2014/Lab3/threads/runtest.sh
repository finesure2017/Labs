# bash runtest.sh
make clean
make

# Lab 2 tests
../../starterCode/scripts/lab2-01-ucontext.py 
../../starterCode/scripts/lab2-02-basic.py
# Lab 3 tests
# Note: Somehow calling test wrong, preemptive test passes
# but I think it just keeps calling itself and looping
# You can see its progress by opening another terminal
# and continuously executing: cat tester.log
../../starterCode/scripts/lab3-01-preemptive.py 
../../starterCode/scripts/lab3-02-wakeup.py 
../../starterCode/scripts/lab3-03-wakeupall.py 
../../starterCode/scripts/lab3-04-lock.py 
../../starterCode/scripts/lab3-05-cv.py 

make clean
