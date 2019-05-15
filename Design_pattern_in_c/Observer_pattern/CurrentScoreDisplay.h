#ifndef CUR_H
typedef struct CurrentScoreDisplay * CurrentScoreDisplay_ptr;
Observer_ptr CurrentScoreDisplay_ptr_to_Observer_ptr(CurrentScoreDisplay_ptr me);

CurrentScoreDisplay_ptr create_CurrentScoreDisplay(void);
#endif
