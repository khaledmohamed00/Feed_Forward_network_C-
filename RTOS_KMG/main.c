#include "TM4C123GH6PM.h"
#include "tm4c123gh6pm.h"
#include "OS_interface.h"
/**
 * main.c
 */


#define TOG_BIT(REG,BIT_NO) REG^=(1<<BIT_NO);

uint64_t Count0=0,Count1=0,Count2=0;
void Task0(void){
  Count0 = 0;
  while(1){
    Count0++;
    TOG_BIT(GPIOF->DATA,1); // toggle bit
    OS_Sleep(200);
  }
}

void Task1(void){
  Count1 = 0;
  while(1){
    Count1++;
    TOG_BIT(GPIOF->DATA,2); // toggle bit
  }
}

void Task2(void){
  Count2 = 0;
  while(1){
    Count2++;
    TOG_BIT(GPIOF->DATA,3); // toggle bit
  }
}

int main(void)
{
SYSCTL->RCGCGPIO |= (1<<5); // enable clock on PortF
GPIOF->DIR = (1<<1)|(1<<2)|(1<<3);  // make LED pins (PF1, PF2, and PF3) outputs
GPIOF->DEN = (1<<1)|(1<<2)|(1<<3); // enable digital function on LED pins
GPIOF->DATA &= ~((1<<1)|(1<<2)|(1<<3)); // turn off leds
create_OS_Thread(0,&Task0);
create_OS_Thread(1,&Task1);
create_OS_Thread(2,&Task2);
OS_launch(500000);



while(1){

}
	return 0;
}
