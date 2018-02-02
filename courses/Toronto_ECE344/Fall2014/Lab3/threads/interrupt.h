#ifndef _INTERRUPT_H_
#define _INTERRUPT_H_

#include <stdio.h>
#include <signal.h>  // for timer interrupt

// To output temporary message if nothing is being done 
#define TBD() do {							\
		printf("%s:%d: %s: please implement this functionality\n", \
		       __FILE__, __LINE__, __FUNCTION__);		\
		exit(1);						\
	} while (0)

/* we will use this signal type for delivering "interrupts". */
#define SIG_TYPE SIGALRM // define SIGALRM as SIG_TYPE which means a process signalling itself 
/* the interrupt will be delivered every 50 usec */
#define SIG_INTERVAL 100 // The interval that the SIGALRM is delivered. 

// Register interrupt handler 
void register_interrupt_handler(int verbose);
// Turn interrupts On (Enable interrupts)  
int interrupts_on(void); // (a wrapper to interrupt_set)
// Disable interrupts 
int interrupts_off(void); // (a wrapper to interrupt_set) 

// Sets the signal state 
// enabled = 1 => allow timer signal 
// enabled = 0 => block timer signals 
// atomically returns previous signal state (O.S. handles this) 
// Use this function for disabling signals to run critical sections 
int interrupts_set(int enabled);
// 
int interrupts_enabled(); // returns the signal stat of current line of code 
// 
void interrupts_quiet(); // turn off printing signal handler functions 

void spin(int msecs);

// Turn of interrupts when printing cause printing is a non-reentrant function 
int unintr_printf(const char *fmt, ...);
#endif
