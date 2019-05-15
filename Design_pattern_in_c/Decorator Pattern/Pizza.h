#ifndef PIZZA_H
#define PIZZA_H
typedef struct Pizza * Pizza_ptr;
typedef char * (*fun_char_ptr)(Pizza_ptr);
typedef int (*fun_int)(Pizza_ptr);


struct Pizza{
char *description;
fun_char_ptr get_description;
fun_int get_cost;

};

#endif
