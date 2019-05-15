#include "Fighter.h"
#include "RYU.h"
#include <stdio.h>
#include <stdlib.h>
static void display(void){

printf("RYU\n");
}

RYU_ptr Create_RYU_Fighter(fun kick,fun jump){

RYU_ptr me= (RYU_ptr)malloc(sizeof(struct RYU));
me->super=Create_Fighter(kick,jump);
me->display=display;
return me;
}
