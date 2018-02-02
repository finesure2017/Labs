#include "common.h"
#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 

int factorial(int n)
{
    if (n == 1)
	return 1;
    return n*factorial(n-1);  
}

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
	printf("Huh?\n"); 
	return 0; 
     }
    double value = 0.0;
    double checkValue = 0.0;   
    double length = strlen(argv[1]); 
    int i = 0; 
    for (i =0 ; i < length; i++)
    {
	if (argv[1][i] > '9' || argv[1][i] < '0')
	{
	    printf("Huh?\n"); 
	    return 0; 
	} 
    }
    value = atoi(argv[1]); 
    checkValue = atof(argv[1]); 
    if (value != checkValue)
    { 
	printf("Huh?\n"); 
	return 0; 
    }
    if(value > 12)
    {
	printf("Overflow\n"); 
	return 0; 
    }   
    if (value < 1) 
    {
	printf("Huh?\n");
	return 0; 
    }
    int final = factorial(value); 
    printf("%d\n", final); 

//	TBD();
	return 0;
}
