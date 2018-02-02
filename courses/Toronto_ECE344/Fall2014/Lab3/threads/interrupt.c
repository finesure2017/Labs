#include <assert.h>
#include <errno.h>
#include <signal.h>
#include <stdlib.h>
#include <ucontext.h>
#include <unistd.h>
#include <sys/time.h>
#include <stdarg.h>
#include "thread.h"
#include "interrupt.h"

static void interrupt_handler(int sig, siginfo_t * sip, void *contextVP);
static void set_interrupt();
static void set_signal(sigset_t * setp);

static int loud = 0; // To indicate if interrupt is quiet or not 

/* Should be called when you initialize threads package. Many of the calls won't
 * make sense at first -- study the man pages! */
 // If verbose = 1 => Print out each time interrupt handler gets called
 // If verbose = 0 => Don't print out when interrupt handler gets called 
void register_interrupt_handler(int verbose)
{
	struct sigaction action; // to store the action of the signal of interest (stores the function that is called) 
	int error;
	static int init = 0;

	assert(!init);	/* should only register once */
	init = 1;
	loud = verbose;
	action.sa_handler = NULL; // use sa_sigaction instead of sa_handler 
	action.sa_sigaction = interrupt_handler;	 //
	/* SIG_TYPE will be blocked while interrupt_handler() is running. */
	error = sigemptyset(&action.sa_mask); // initialize the set of signals to be empty 
	assert(!error);

	/* use sa_sigaction as handler instead of sa_handler */
	action.sa_flags = SA_SIGINFO;	// use sa_sigaction instead of sa_handler 
	// Set up the sigaction for SIG_TYPE signal to be action 
	if (sigaction(SIG_TYPE, &action, NULL)) {
		perror("Setting up signal handler");
		assert(0);
	}
	// Allow interrupts 
	set_interrupt();
}

/* enables interrupts. */
int interrupts_on()
{
	return interrupts_set(1);
}

/* disables interrupts */
int interrupts_off()
{
	return interrupts_set(0);
}

/* enables or disables interrupts, and returns whether interrupts were enabled
 * or not previously. */
int interrupts_set(int enabled)
{
	int ret;
	sigset_t mask, omask;

	// Include SIG_TYPE into signal set mask 
	set_signal(&mask);
	if (enabled) {
		ret = sigprocmask(SIG_UNBLOCK, &mask, &omask); // if enabled, unblock all signals from mask	// does so atomically in hardware
	} else {
		ret = sigprocmask(SIG_BLOCK, &mask, &omask); // if NOT enabled, block all signals from mask 	// does so atomically in hardware
	}
	assert(!ret);
	return (sigismember(&omask, SIG_TYPE) ? 0 : 1); // return previous signal state 
													// check if the SIG_TYPE is in the signal mask 
													// if it is, return 0, if it wasn't return 1 
}

int interrupts_enabled()
{
	sigset_t mask;
	int ret;

	ret = sigprocmask(0, NULL, &mask); // get the previous value of signal mask  atomically 
	assert(!ret);
	return (sigismember(&mask, SIG_TYPE) ? 0 : 1);	// check if the SIG_TYPE is in the signal mask 
													// if it is, return 0, if it wasn't return 1 
}

// Set loud to 0 
void interrupts_quiet()
{
	loud = 0;
}

// Return after usec seconds 
void spin(int usecs)
{
	struct timeval start, end, diff;
	int ret;

	ret = gettimeofday(&start, NULL); // get initial time 
	assert(!ret);
	while (1) {
		ret = gettimeofday(&end, NULL);	// get current time 
		timersub(&end, &start, &diff); // get time elapsed
		// break when time elapse is long enough 
		if ((diff.tv_sec * 1000000 + diff.tv_usec) >= usecs) {
			break;
		}
	}
}

/* turn off interrupts while printing */
int unintr_printf(const char *fmt, ...)
{
	int ret, enabled;
	va_list args;

	enabled = interrupts_off(); // get previous condition of interrupt , and  turn interrupts off
	// print 
	va_start(args, fmt);
	ret = vprintf(fmt, args);
	va_end(args);
	// return interrupts to previous state 
	interrupts_set(enabled);
	return ret;
}

/* static functions */

// Add the SIG_TYPE signal into the setp
static void set_signal(sigset_t * setp)
{
	int ret;
	ret = sigemptyset(setp); // initialize the set to be empty
	assert(!ret);
	ret = sigaddset(setp, SIG_TYPE); // Add SIG_TYPE to the set
	assert(!ret);
	return;
}

static int first = 1;
static struct timeval start, end, diff = { 0, 0 };

/*
 * STUB: once register_interrupt_handler() is called, this routine
 * gets called each time SIG_TYPE is sent to this process
 */
static void interrupt_handler(int sig, siginfo_t * sip, void *contextVP)
{
	// Create a context 
	ucontext_t *context = (ucontext_t *) contextVP;

	/* check that SIG_TYPE is blocked on entry */
	assert(!interrupts_enabled()); // make sure interrupts is off 
	if (loud) 
	{
		int ret;
		ret = gettimeofday(&end, NULL);
		assert(!ret);
		if (first) 
		{
			first = 0; // If first time this function is called, do nothing 
		} 
		else 
		{
			timersub(&end, &start, &diff); // else, get the difference in time 
		}
		start = end;	// Get the new start time 
		printf("%s: context at %10p, time diff = %ld us\n",
		       __FUNCTION__, context,
		       diff.tv_sec * 1000000 + diff.tv_usec);
	}
	// Set interupt  to SIG_ALRM to interrupt yourself 
	set_interrupt();
	/* implement preemptive threading by calling thread_yield */
	thread_yield(THREAD_ANY);
}

/*
 * Use the setitimer() system call to set an alarm in the future. At that time,
 * this process will receive a SIGALRM signal.
 */
static void set_interrupt()
{
	int ret;
	struct itimerval val;

	val.it_interval.tv_sec = 0;
	val.it_interval.tv_usec = 0;

	val.it_value.tv_sec = 0;
	val.it_value.tv_usec = SIG_INTERVAL;

	ret = setitimer(ITIMER_REAL, &val, NULL); // set the timer, documented in real time, using the timer value , val , 
											  // this automatically sends a SIG_ALRM to itself when timer expires
	assert(!ret);
}
