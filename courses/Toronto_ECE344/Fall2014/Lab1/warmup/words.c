#include "common.h"
#include <stdlib.h> 
#include <stdio.h> 
#include <string.h> 

int main(int argc , char* argv[])
{
//	TBD();
	int i = 0;
	if (argc == 0)
	    return 0;  
	for(i = 1; i < argc ; i++)
	{
	    printf("%s\n", argv[i]); 
	}
	return 0;
}
