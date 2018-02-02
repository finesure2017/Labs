#include <assert.h>
#include <stdlib.h>
#include "thread.h"
#include "interrupt.h"
#include "test_thread.h"

// This file shows how to use interrupt.h file 

// To test if enabled is enabled correctly for interrupts 
static void test(int enabled)
{
	int i;
	int is_enabled;

	// 16 just for fun, it prints a new line every 16, 
	// note: That 
	for (i = 0; i < 16; i++) 
	{
		spin(SIG_INTERVAL / 5);	/* spin for a short period */ // do nothing for some time
		unintr_printf(".");	// print a . 
		fflush(stdout);	// flush it out 
		/* check whether interrupts are enabled or not correctly */
		is_enabled = interrupts_enabled();
		assert(enabled == is_enabled);	// check tha enabled is is_enabled 
	}
	unintr_printf("\n");
	// Note: This show_handler will also print each time the handler is invoked by the interrupt
	// which means it will print it's own new message and new line 
}

int main(int argc, char **argv)
{
	int enabled;

	thread_init(); // initialize thread 
	/* show interrupt handler output */
	register_interrupt_handler(1);	// register interrupt handler 
	test(1);	// test that interrupts are enabled

	/*  test interrupts_on/off/set */
	enabled = interrupts_off();	// turn off interrupts
	assert(enabled);	// test that previous value was on 
	test(0);	 // test that it is now off
	enabled = interrupts_off();	// now turn it off again
	assert(!enabled);	// check that it was still off
	test(0);	// check that it it is now off
	enabled = interrupts_set(1);	// turn it back on
	assert(!enabled);	// check that it used to be off
	test(1);	// check that is it now on
	enabled = interrupts_set(1);	// turn it back on
	assert(enabled);	// check that it was on
	test(1);	// check it is now on
	exit(0);	 // thats it 
}
