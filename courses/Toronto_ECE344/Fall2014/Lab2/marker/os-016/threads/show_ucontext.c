#include <assert.h>
#include <stdlib.h>
#include <ucontext.h>
#include "interrupt.h"

#ifndef __x86_64__
#error "Do this project on a 64-bit x86-64 linux machine"
#endif /* __x86_64__ */

#if __WORDSIZE != 64
#error "word size should be 64 bits"
#endif

static void call_setcontext(ucontext_t * context);
static void show_interrupt(void);

/* zero out the context */
ucontext_t mycontext = { 0 }; // ucontext_t is defined in <ucontext.h> , it is initialize as memory pointer to variables. 

int main(int argc, char **argv)
{
	int setcontext_called = 0; // a boolean to know if setcontext is called 
	int err;

	/*
	 * DO NOT CHANGE/ADD ANY CODE UNTIL BELOW TBD(). SEE BELOW.
	 */

	/* Get context: make sure to read the man page of getcontext in detail,
	 * or else you will not be able to answer the questions below. */
	err = getcontext(&mycontext); // pass by pointer, to store the current context into mycontext
	assert(!err); // make sure it is successful. 

	/* QUESTION: which of the fields of mycontext changed due to the call
	 * above? Hint: It will help to run the program using gdb and put a
	 * breakpoint at entry to main and before and after the calls to
	 * getcontext().
	 * - Use "info registers" to see the values of the registers.
	 * - Use "next"/"step" to advance to the next line of code.
	 * - Use "print mycontext" to see the values stored in mycontext.
	 *   Compare them with the output of "info registers".
	 * - Use "ptype mycontext" so see the type/fields of mycontext */
	 
	 // Which of the fields of mycontext is changed due to call above? 
	 // only  mcontext_t       uc_mcontext gets changed, everything else stays as zero as shown in gdb 

	// Note: __FUNCTION__ prints the name of current function , in this case, it is main 
	printf("%s: setcontext_called = %d\n", __FUNCTION__, setcontext_called);
	if (setcontext_called == 1) {
		/* will be get here? why or why not? */
		// No we won't because setcontext_called was initialize to 0 and setcontext was never called
		// therefore, we won't get to here 
		show_interrupt();
		exit(0);
	}

	/*
	 * UNCOMMENT TBD AND ONLY CHANGE/ADD CODE BELOW.
	 */
//	TBD(); // This automatically exits the function 
    // Note: Only change all code with (-1) and nothing else. 

	/* show size of ucontext_t structure. Hint: use sizeof(). */
	printf("ucontext_t size = %ld bytes\n", (long int) sizeof(ucontext_t));

	/* now, look inside of the context you just saved. */

	/* first, think about code */
	/* the program counter is called rip in x86-64 */
	// rip = register instruction pointer = program counter  
	printf("memory address of main() = 0x%lx\n", (unsigned long) main);
	printf("memory address of the program counter (RIP) saved "
	       "in mycontext = 0x%lx\n", (unsigned long) mycontext.uc_mcontext.gregs[REG_RIP]);

	/* now, think about parameters */
	printf("argc = %d\n", (int) mycontext.uc_mcontext.gregs[REG_RSI]); // RSI =>  Argument 2
	printf("argv = %p\n", (void *) mycontext.uc_mcontext.gregs[REG_RDI]); // RDI => Argument 1
	
	/* QUESTIONS: how are these parameters passed into the main function? 
	 * are there any saved registers in mycontext that store the parameter
	 * values above. why or why not? Hint: Use gdb, and then run
	 * "disassemble main" in gdb, and then scroll up to see the beginning of
	 * the main function. */ 
	 // These parameters are passed into the main function by first: 
	 //  push   %rbp				// Push the frame pointer onto the stack
     //  mov    %rsp,%rbp			// Make the frame pointer save the value of the stack pointer
     //  sub    $0x20,%rsp			// Decrement the stack pointer downwards to make space for 0x20 memory
     //  mov    %edi,-0x14(%rbp)	// Save the first argument from the frame pointer onwards into the call stack
     //  mov    %rsi,-0x20(%rbp)    // Save the 2nd argument from the frame pointer onwards into the call stick 
	/* now, think about the stack */
	/* QUESTIONS: Is variable setcontext_called and variable err stored on the stack? does the
	 * stack grow up or down? What are the stack related data in
	 * mycontext.uc_mcontext.gregs[]? */ 

	 // The variables setcontext_called and err are stored on the stack (based on code below) 
	 
	 // The stack grows down since you decrement the stack pointer downwards instead of incrementing it 
	 // Stack related data in mycontext_uc_mcontext.gregs[] are:
	 //  RSP => Stack Pointer Register 
	 //  RBP => Stack Base Pointer Register
	 
	printf("memory address of the variable setcontext_called = %p\n", (void *) &setcontext_called);
	
	printf("memory address of the variable err = %p\n", (void *) &err);
	
	printf("number of bytes pushed to the stack between setcontext_called "
	       "and err = %ld\n",  (unsigned long) &err - (unsigned long) &setcontext_called);

	printf("stack pointer register (RSP) stored in mycontext = 0x%lx\n", (unsigned long) mycontext.uc_mcontext.gregs[REG_RSP]);

	printf("number of bytes between err and the saved stack in mycontext "
	       "= %ld\n", (unsigned long) &err - mycontext.uc_mcontext.gregs[REG_RSP]);

	/* QUESTIONS: what is the uc_stack field in mycontext? does it point
	 * to the current stack pointer, top of the stack, bottom of the stack,
	 * or none of the above? */
	 // None of the above, uc_stack is not used in current implementation and not touched. 
	 // Proof: http://comments.gmane.org/gmane.comp.standards.posix.austin.general/9469
	 
	printf("value of uc_stack.ss_sp = 0x%lx\n", (unsigned long) mycontext.uc_stack.ss_sp);

	/* Don't move on to the next part of the lab until you know how to
	 * change the stack in a context when you manipulate a context to create
	 * a new thread. */

	/* now we will try to understand how setcontext works */
	setcontext_called = 1;
	call_setcontext(&mycontext);
	/* QUESTION: why does the program not fail at the assert below? */
	// Cause on successful, setcontext() doesn't return 
	assert(0);
}

static void call_setcontext(ucontext_t * context)
{
	int err = setcontext(context);
	assert(!err);
}

static void show_interrupt(void)
{
	int err;
	/* QUESTION: how did we get here if there was an assert above? */
	/* now think about interrupts. you will need to understand how they
	 * work, and how they interact with get/setcontext for implementing
	 * preemptive threading. */
	 // Interrupts will change control and use get/setcontext of current thread before changing to next thread
	 

	/* QUESTION: what does interrupts_on() do? see interrupt.c */
	interrupts_on();
	// Interrupts_on() enables interrupts  

	/* getcontext stores the process's signal mask */
	err = getcontext(&mycontext); // returns 0 when successful
	assert(!err); // should assert !0 => 1 => assert is true and doesn't fail 

	/* QUESTION: Are interrupts masked (i.e., disabled) in mycontext?
	 * HINT: use sigismember below. */
	printf("interrupt is disabled = %d\n", (unsigned int) sigismember(&mycontext.uc_sigmask, SIG_TYPE));
	// sigismember(signalmask, SIGNALTYPE) test if a signal is part of the signal mask 
	
	interrupts_off(); // disable interrupts 

	err = getcontext(&mycontext);
	assert(!err);

	/* QUESTION: which fields of mycontext changed as a result of the
	 * getcontext call above? */
	printf("interrupt is disabled = %d\n", (unsigned int) sigismember(&mycontext.uc_sigmask, SIG_TYPE));
	// The field of mycontext that changes as a result of getcontext call above is:
	// 
	
}
