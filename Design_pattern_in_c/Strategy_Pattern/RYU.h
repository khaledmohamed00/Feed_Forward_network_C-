#ifndef RYU_H

typedef struct RYU * RYU_ptr;
struct RYU {
struct Fighter *super;
fun display;
};
RYU_ptr Create_RYU_Fighter(fun kick,fun jump);

#endif
