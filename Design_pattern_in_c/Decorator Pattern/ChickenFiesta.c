#include "Pizza.h"

static char* get_description(Pizza_ptr pizza){
return pizza->description;
}
static int get_cost(Pizza_ptr pizza){
return 200;
}

Pizza_ptr create_ChickenFiesta(void){

Pizza_ptr me=malloc(sizeof (struct Pizza));
me->description="ChickenFiesta";
me->get_description=get_description;
me->get_cost=get_cost;

return me;
}
