// Vectorization containers
template <typename scalar_t, size_t vec_size>
struct __align__(vec_size * sizeof(scalar_t)) vec_n_t
{
    scalar_t val[vec_size];
};
