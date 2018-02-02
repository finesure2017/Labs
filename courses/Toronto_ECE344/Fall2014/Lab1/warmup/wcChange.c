#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include "common.h"
#include "wc.h"

struct wc {
	/* you can define this struct to have whatever fields you want. */
	char wordName[80] = ""; 
	int numWords = 0; // currently have 0 words in wc, but highest wc points refers to number of words in linked list 
	struct wc* next; 
};

struct wc* wc_init()
{
	struct wc* wc;

	wc = (struct wc *)malloc(sizeof(struct wc));
	assert(wc);
	wc->next = NULL; 
//	TBD();
	return wc;
}
int wc_insert_word(struct wc *wc, char *word)
{
//	TBD();
	struct wc* newWord;
	newWord = (struct wc *)malloc(sizeof(struct wc));
	assert(newWord); // return 0 if ou of memory and nothing if successful.
	
	if (wc->next == NULL)
	{
		newWord = wc->next; 
		newWord->numWords = 1; 
		strcpy(newWord->wordName, word); 
		newWord->next = NULL; 
		wc->numWords++; 
		return 1; 
	}
	
	
	struct wc* prev;
	prev = wc->next; 
	if (prev->next == NULL)
	{
		// if same word 
		if(strcmp(prev->wordName, word) == 0)
		{
			prev->numWords++;
			free(newWord); 
			return 1; 
		}
		else
		{
			newWord = prev->next; 
			newWord->numWords = 1; 
			strcpy(newWord->wordName, word); 
			newWord->next = NULL; 
			wc->numWords++; 

			return 1; 
		}
	}
	struct wc* last;
	last = prev->next; 
	while(last)
	{
		if(strcmp(prev->wordName, word) == 0)
		{
			prev->numWords++;
			free(newWord); 
			return 1; 
		}
		else
		{
			prev = prev->next; 
			last = last->next; 
		}	
	}
	// end of linked list
	if(strcmp(prev->wordName, word) == 0)
	{
		prev->numWords++;
		free(newWord); 
		return 1; 
	}
	// add a new wor dto end of linked list 
	else
	{
		newWord = prev->next; 
		newWord->numWords = 1; 
		strcpy(newWord->wordName, word); 
		newWord->next = NULL; 
		wc->numWords++; 
		return 1; 
	}
	return 1; // if successful 
}

void wc_output(struct wc *wc)
{
//	TBD();
	if (wc == NULL)
	{
		return; 
	}
	struct wc* curr;
	curr = wc->next; 
	while(curr)
	{
		printf("%s", curr->wordName); 
		printf(":%d\n", curr->numWords); 
		curr  = curr->next; 
	}
	return; 
}
