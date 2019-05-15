#include "Pizza.h"
static Pizza_ptr plain_pizza;
static char* get_description(Pizza_ptr me){
printf("%s ",plain_pizza->get_description(plain_pizza));
return me->description;

}
static int get_cost(Pizza_ptr me){
return 40+plain_pizza->get_cost(plain_pizza);
}

Pizza_ptr create_FreshTomato(Pizza_ptr pizza){
plain_pizza=pizza;
Pizza_ptr me=malloc(sizeof (struct Pizza));
me->description="FreshTomato";
me->get_description=get_description;
me->get_cost=get_cost;

return me;
}
