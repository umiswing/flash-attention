#include "reduce_launch_template.h"

template<>
void run_reduce_<cutlass::bfloat16_t, 160>(Reduce_attn_scores_params &params, cudaStream_t stream) {
    run_reduce_hdim160<cutlass::bfloat16_t>(params, stream);
}
