#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h> 
#include "common.h"
#include "point.h"
#include "sorted_points.h"

struct sorted_points {
	/* you can define this struct to have whatever fields you want. */
	double x; 
	double y; 
	struct sorted_points* next; 
};


// returns -1 if p2>p1 and 1 if p1>= p2 
int sorted_point_compare(const struct sorted_points *p1, const struct sorted_points *p2)
{
//printf("A1\n"); 
	// returns -1 if p1 < p2, 0 if p1 == p2, 1 if p1 > p2
	double distancep1 = sqrt(pow(p1->x, 2.0) + pow(p1->y, 2.0)); 
	double distancep2 = sqrt(pow(p2->x, 2.0) + pow(p2->y, 2.0)); 
	if (distancep1 < distancep2)
	    return -1; 
	else if (distancep1 == distancep2)
	{
		if (p1->x < p2->x)
		{
//printf("A0\n"); 

			return -1; 
		}
		else if (p1->y <= p2->y)
		{
//printf("A3\n"); 

			return -1; 
		}
		else 
		{
//printf("A4\n"); 

			return 1; 
		}
	}
//printf("A5\n"); 

	return 1; // p1 > p2  

}

// returns -1 if p2>p1 , 0 if p1 == p2, and 1 if p1>= p2 
int sorted_point_compare_duplicate(const struct sorted_points *p1, const struct sorted_points *p2)
{
//printf("A6\n"); 

	// returns -1 if p1 < p2, 0 if p1 == p2, 1 if p1 > p2
	double distancep1 = sqrt(pow(p1->x, 2.0) + pow(p1->y, 2.0)); 
	double distancep2 = sqrt(pow(p2->x, 2.0) + pow(p2->y, 2.0)); 
	if (distancep1 < distancep2)
	{
//printf("A7\n"); 

	    return -1; 
	}
	else if (distancep1 == distancep2)
	{
		if (p1->x < p2->x)
		{
//printf("A8\n"); 

			return -1; 
		}
		else if (p1->y < p2->y)
		{
//printf("A9\n"); 

			return -1; 
		}
		else 
		{
//printf("A10\n"); 

			return 0;  // exactly the same
		}
	}
//printf("A11\n"); 

	return 1; // p1 > p2  
}


struct sorted_points* sp_init()
{
//printf("A12\n"); 

	struct sorted_points *sp;
	sp = (struct sorted_points *)malloc(sizeof(struct sorted_points));
	assert(sp);
	sp->x = 0; 
	sp->y = 0; 
	sp->next = NULL; 

//	TBD();
//printf("A13\n"); 

	return sp;
}

void sp_destroy(struct sorted_points *sp)
{
//printf("A14\n"); 

	if (sp == NULL)
	{
//printf("A15\n"); 

		return; 
	}
	struct sorted_points *last; 
	last = sp->next; 
	while (last)
	{

		free(sp); 
		sp = last; 
		last = last->next; 
	}

	free(sp);
//printf("A16\n"); 

	return; 
}

int sp_add_point(struct sorted_points *sp, double x, double y)
{
//printf("A17\n"); 

	if (sp == NULL)
	{
//printf("A18\n"); 

		return 0; 
	}
	// Case 1: sp points to NULL 
	// Case 2: sp points to object which points to other object
	// Case 3: final object points to NULL 
	struct sorted_points *newSp;
	newSp = (struct sorted_points *)malloc(sizeof(struct sorted_points));
	assert(newSp);
	newSp->x = x; 
	newSp->y = y; 
	newSp->next = NULL; 
	struct sorted_points *prevSP;
	struct sorted_points *nextSP; 
	prevSP = sp;
	nextSP = sp->next; 
	int compare = 0; 
	while(nextSP != NULL)
	{

		compare = sorted_point_compare(nextSP, newSp);
		if ( compare == -1) // nextSP < newSp 
		{
			prevSP = nextSP;
			nextSP = nextSP->next; 
			// continue iterating 
		}
		else if ( compare == 1) // nextSp >= newSp
		{
			// Found location, add object 
			prevSP->next = newSp; 
			newSp->next = nextSP; 
//	printf("A19\n"); 

			return 1; 
		}
		else 
		{
			printf("ERROR! Should never be here\n"); 
//printf("A20\n"); 

			return 0; 
		}
	}
	// Case 3: Final object points to NULL 
	// nextSP is NULL
	prevSP->next = newSp; 
	newSp->next = NULL; 
//printf("A21\n"); 

	return 1; 
}

int sp_remove_first(struct sorted_points *sp, struct point *ret)
{
//	TBD();
	if(sp == NULL)
	{
//printf("A22\n"); 

		return 0; 
	}
	if (sp->next == NULL)
	{
//printf("A23\n"); 

		return 0; // list is empty 
	}	
	struct sorted_points *delSP; 
	delSP = sp->next; 
	// Now remove first element
	sp->next = delSP->next; 
	ret->x = delSP->x; 
	ret->y = delSP->y; 
	free(delSP); 
//	printf("A24\n"); 

	return 1;
}

int sp_remove_last(struct sorted_points *sp, struct point *ret)
{
//	TBD();
	if(sp == NULL)
	{
//printf("A25\n"); 

		return 0; 
	}
	if (sp->next == NULL)
	{
//printf("A26\n"); 

		return 0; // list is empty 
	}	
	struct sorted_points *prevSP; 
	struct sorted_points *delSP; 
	prevSP = sp->next; 
	
	// Case 1: Only 1 element
	if (prevSP->next == NULL)
	{
		free(prevSP); 
		sp->next = NULL;
//printf("A27\n"); 

		return 1; 
	}
	delSP= prevSP->next; 
	// Loop for last element
	while(delSP->next != NULL)
	{ 
		prevSP = delSP; 
		delSP= delSP->next; 
	}
	// Found last 
	ret->x = delSP->x; 
	ret->y = delSP->y; 
	free(delSP); 
	prevSP->next = NULL; 
//printf("A28\n"); 

	return 1;
}

/* Remove the point that appears in position <index> on the sorted list, storing
 * its value in *ret. Returns 1 on success and 0 on failure (too short list).
 * The first item on the list is at index 0. */
int sp_remove_by_index(struct sorted_points *sp, int index, struct point *ret)
{
//printf("A29\n"); 

	if (sp == NULL)
	{
//printf("A39\n"); 

		return 0;		
	}
	if (ret == NULL)
	{
//printf("A31\n"); 

		return 0; 
	}
	if (index < 0 )
	{
//printf("A32\n"); 

		return 0;
	}
	int iterate = 0; 
	if (index == 0)
	{
//printf("A33\n"); 

		return sp_remove_first(sp, ret); 
	}
	// not first index 
	struct sorted_points *prevSP; 
	struct sorted_points *delSP; 	
	prevSP = sp->next; 
	
	if (prevSP == NULL)
	{
		return 0; // no element at all 
	}
	
	
	// Case 1: Only 1 element
	if (prevSP->next == NULL)
	{
//printf("A34\n"); 

		return 0; // don't have index 
	}
	delSP= prevSP->next; 
	
	if (index == 1)
	{
//	printf("A101\n"); 

		prevSP->next = delSP->next;  
		ret->x = delSP->x; 
		ret->y = delSP->y; 
		free(delSP); 
//printf("A35\n"); 

		return 1; 
	}	
	iterate = 1; // current index is 1 for delSP; 
	
	// Loop for element
	while(delSP != NULL)
	{ 
		// Found the right index for deleteSP
		if (iterate == index)
		{
			prevSP->next = delSP->next;  
			ret->x = delSP->x; 
			ret->y = delSP->y; 		
			free(delSP); 
//	printf("A36\n"); 

			return 1; 			
		}
		prevSP = delSP; 
		delSP = delSP->next; // incremenetd deleteSP 
		iterate++; 
	}
	// no such element 
	//printf("A37\n"); 

	return 0; 
}

// Need iterate through entire loop 
int sp_delete_duplicates(struct sorted_points *sp)
{
//	TBD();
	if( sp == NULL)
	{
//	printf("A38\n"); 

		return 0;  // deleted 0 records
	}
	int numDelete = 0; // number of records deleted 
	
	if( sp->next == NULL)
	{
//	printf("A39\n"); 

		return 0; // only 0 record => deleted 0 records 
	} 
	struct sorted_points *prevSP; 
	struct sorted_points *delSP; 
	
	prevSP = sp->next; 
	if (prevSP->next == NULL)
	{
//printf("A50\n"); 

		return 0; // only 1 record => deleted 0 records 
	}
	delSP = prevSP->next; 
	int comparison = 0; 
	while (delSP != NULL)
	{
		comparison = sorted_point_compare_duplicate(prevSP, delSP);
		if (comparison == 0)
		{
			prevSP->next = delSP->next; 
			free(delSP); 
			numDelete++; // increment number of deleted nodes
			delSP = prevSP->next; 
		}
		// iterate next 
		else 
		{
			prevSP = delSP; 
			delSP = delSP->next; 
		}
	}
//printf("A51\n"); 

	return numDelete; 

	// return -1;
}
