#ifndef CUDARESIZE_H
#define cudaresize_h

#ifdef __cplusplus 
#define EXTERNC extern "C"
#else 
#define EXTERNC 
#endif

EXTERNC void interpolate(
    const float *input_array, float *output_array, unsigned int width, unsigned int height, unsigned int depth,
    unsigned int new_width, unsigned int new_height, unsigned int new_depth
);

#endif
