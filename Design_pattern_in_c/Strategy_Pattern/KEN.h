#ifndef KEN_H

typedef struct KEN * KEN_ptr;
struct KEN {
struct Fighter *super;
fun display;
};
KEN_ptr Create_KEN_Fighter(fun kick,fun jump);

#endif
