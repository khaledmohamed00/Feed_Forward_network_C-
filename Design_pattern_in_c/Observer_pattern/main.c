#include <stdio.h>
#include <stdlib.h>
#include "Observer.h"
#include "AverageScoreDisplay.h"
#include "CurrentScoreDisplay.h"
#include "CricketData.h"
#include"Subject.h"
int main()
{AverageScoreDisplay_ptr A= create_AverageScoreDisplay();
 CurrentScoreDisplay_ptr C=create_CurrentScoreDisplay();

 Observer_ptr o1=AverageScoreDisplay_ptr_to_Observer_ptr(A);
 Observer_ptr o2=CurrentScoreDisplay_ptr_to_Observer_ptr(C);

 CricketData_ptr D= create_CricketData(90,2,10.2);
 registerObserver(D,o1);
 registerObserver(D,o2);
 dataChanged(D);

//*/

    //printf("Hello world!\n");
    return 0;
}
