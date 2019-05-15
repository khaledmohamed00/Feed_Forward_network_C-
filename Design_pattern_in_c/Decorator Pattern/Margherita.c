#include "Pizza.h"

static char* get_description(Pizza_ptr pizza){
return pizza->description;
}
static int get_cost(Pizza_ptr pizza){
return 100;
}

Pizza_ptr create_Margherita(void){

Pizza_ptr me=malloc(sizeof (struct Pizza));
me->description="Margherita";
me->get_description=get_description;
me->get_cost=get_cost;

return me;
}
