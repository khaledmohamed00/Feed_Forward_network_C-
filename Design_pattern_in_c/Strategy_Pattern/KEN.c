#include "Fighter.h"
#include "KEN.h"
#include <stdio.h>
#include <stdlib.h>
static void display(void){

printf("KEN\n");
}

KEN_ptr Create_KEN_Fighter(fun kick,fun jump){

KEN_ptr me= (KEN_ptr)malloc(sizeof(struct KEN));
me->super=Create_Fighter(kick,jump);
me->display=display;
return me;
}
