#include <stdio.h>
#include <stdlib.h>
#include "DigitalObserver.h"
#include "AnalogObserver.h"
#include "TimeSubject.h"
int main()
{ DigitalStopWatchPtr D=createDigitalWatch();
  AnalogStopWatchPtr A=createAnalogWatch();

  //attach(D);
  //attach(A);
  msTick();
   // printf("Hello world!\n");
    return 0;
}
