#ifndef __LONG_SHORT_TERM_MEMORY_SPARSE_KERNEL_DEFINES__
#define __LONG_SHORT_TERM_MEMORY_SPARSE_KERNEL_DEFINES__

#define FLAG(x) (1 << (x))

#define LSNN_FORWARD_PASS                                   FLAG(0)
#define LSNN_BACKWARD_PASS                                  FLAG(1)
#define LSNN_ELIGIBILTY_GRADIENTS                           FLAG(2)
#define LSNN_FIXED_BROADCAST_LEARNSIGNAL                    FLAG(3)
#define LSNN_READOUT_FORWARD_GRAIENTS                       FLAG(4)
#define LSNN_INITIALIZE_FORWARD_PASS                        FLAG(5)
#define LSNN_INITIALIZE_BACKWARD_PASS                       FLAG(6)
#define LSNN_REGRESSION_ERROR                               FLAG(7)
#define LSNN_EXTERNAL_ERROR                                 FLAG(8)
#define LSNN_INPUT_ERRORS                                   FLAG(9)
#define LSNN_BACKPROPAGATED_GRADIENTS                       FLAG(10)
#define LSNN_INITIALIZE_GRADIENTS                           FLAG(11)
#define LSNN_FAST_FRINIG_RATE                               FLAG(12)
#define LSNN_DEBUG_DELTA_ERRORS                             FLAG(13)
#define LSNN_BACKPROPAGATED_ELIGIBILITY_GRADIENTS           FLAG(14)
#define LSNN_SYMETRIC_EPROP                                 FLAG(15)
#define LSNN_NO_GRADIENT_COLLECTION                         FLAG(16)
#define LSNN_CLASSIFICATION_ERROR                           FLAG(17)

#define CONTAINS(x, flag) ((x) & (flag))

#endif
