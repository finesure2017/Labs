#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include "common.h"
#include "wc.h"

struct wc {
	/* you can define this struct to have whatever fields you want. */
	char wordName[80]; 
	int numWords; // currently have 0 words in wc, but highest wc points refers to number of words in linked list 
	struct wc* next; 
};

struct wc* wc_init()
{
//printf("B1\n"); 
	struct wc* wc;
	wc = (struct wc *)malloc(sizeof(struct wc));
	assert(wc);
	wc->next = NULL;
	wc->numWords = 0; 
	strcpy(wc->wordName, "");  
//	TBD();
//printf("B2\n"); 
	return wc;
}
int wc_insert_word(struct wc *wc, char *word)
{
//printf("B3\n"); 
//	TBD();
	struct wc* newWord;
	newWord = (struct wc *)malloc(sizeof(struct wc));
	assert(newWord); // return 0 if ou of memory and nothing if successful.
//printf("B58\n"); 
	if (wc->next == NULL)
	{
//printf("B59\n"); 
		wc->next = newWord;
//printf("B63\n"); 
		newWord->numWords = 1; 
//printf("B60\n"); 
		strcpy(newWord->wordName, word); 
//printf("B61\n"); 
		newWord->next = NULL; 
		wc->numWords++; 
//printf("B4\n"); 
		return 1; 
	}
	
//printf("B50\n"); 
	struct wc* prev;
	prev = wc->next; 
	if (prev->next == NULL)
	{
//printf("B5\n"); 
		// if same word 
		if(strcmp(prev->wordName, word) == 0)
		{
//printf("B71\n"); 
			prev->numWords++;
//printf("B72\n"); 
			free(newWord); 
//printf("B6\n"); 
			return 1; 
		}
		else
		{
//printf("B73\n"); 
			prev->next = newWord; 
			newWord->numWords = 1; 
			strcpy(newWord->wordName, word); 
			newWord->next = NULL; 
			wc->numWords++; 
//printf("B7\n"); 
			return 1; 
		}
	}
//printf("B51\n"); 
	struct wc* last;
	last = prev->next; 
	while(last)
	{
		if(strcmp(prev->wordName, word) == 0)
		{
			prev->numWords++;
			free(newWord); 
//printf("B8\n"); 
			return 1; 
		}
		else
		{
			prev = prev->next; 
			last = last->next; 
		}	
	}
//printf("B52\n"); 
	// end of linked list
	if(strcmp(prev->wordName, word) == 0)
	{
//printf("B81\n"); 
		prev->numWords++;
		free(newWord);
//printf("B9\n");  
		return 1; 
	}
	// add a new wor dto end of linked list 
	else
	{
//printf("B82\n"); 
		prev->next = newWord; 
		newWord->numWords = 1; 
		strcpy(newWord->wordName, word); 
		newWord->next = NULL; 
		wc->numWords++; 
//printf("B10\n"); 
		return 1; 
	}
	return 1; // if successful 
}

void wc_output(struct wc *wc)
{
//printf("B11\n"); 
//	TBD();
	if (wc == NULL)
	{
//printf("B12\n"); 
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
//printf("B13\n"); 
	return; 
}
