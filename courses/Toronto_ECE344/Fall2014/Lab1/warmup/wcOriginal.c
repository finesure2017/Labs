#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include "wc.h"

struct wc {
	/* you can define this struct to have whatever fields you want. */
	char wordName[80]; 
	int numWords; // currently have 0 words in wc, but highest wc points refers to number of words in linked list 
	struct wc* next; 

};

struct wc *
wc_init()
{
	struct wc *wc;

	wc = (struct wc *)malloc(sizeof(struct wc));
	assert(wc);
	wc->numWords = 0; 
	strcpy(wc->wordName,""); 

	TBD();

	return wc;
}

int
wc_insert_word(struct wc *wc, char *word)
{
	TBD();
	return 0;
}

void
wc_output(struct wc *wc)
{
	TBD();
}
