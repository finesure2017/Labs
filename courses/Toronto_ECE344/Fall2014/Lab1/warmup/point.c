#include <assert.h>
#include <math.h> 
#include "common.h"
#include "point.h"

// update *p by increasing p->x by x and p->y by y 
void point_translate(struct point *p, double x, double y)
{	
	//	TBD();
	p->x += x; 
	p->y += y; 
}

// return the distance from p1 to p2 
double point_distance(const struct point *p1, const struct point *p2)
{
	//	TBD();
	double dx = abs(p1->x - p2->x); 
	double dy = abs(p1->y - p2->y); 
	double distance = sqrt(pow(dx, 2.0) + pow(dy, 2.0)); 
	return distance; 
}

int point_compare(const struct point *p1, const struct point *p2)
{

	// returns -1 if p1 < p2, 0 if p1 == p2, 1 if p1 > p2
	double distancep1 = sqrt(pow(p1->x, 2.0) + pow(p1->y, 2.0)); 
	double distancep2 = sqrt(pow(p2->x, 2.0) + pow(p2->y, 2.0)); 
	if (distancep1 < distancep2)
	    return -1; 
	else if (distancep1 == distancep2)
	    return 0; 
	return 1; // p1 > p2  
	//	TBD();
}
