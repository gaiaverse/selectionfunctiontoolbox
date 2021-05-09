data {
    int<lower=0> P;                       // number of pixels
    int<lower=0> M;                       // number of bins in magnitude space
    int<lower=0> M_subspace;              // number of inducing points in magnitude space
    int<lower=0> C;                       // number of bins in colour space
    int<lower=0> C_subspace;              // number of inducing points in colour space
    int<lower=0> L;                       // 2 * max l of hamonics + 1
    int<lower=0> S;                       // number of harmonics
    int<lower=0> R;                       // number of HEALPix isolatitude rings
    matrix[R,S] lambda;                   // spherical harmonics decomposed
    matrix[L,P] azimuth;                  // spherical harmonics decomposed
    int pixel_to_ring[P];                 // map P->R
    int lower[L];                         // map S->L
    int upper[L];                         // map S->L
    vector[S] mu;                         // mean of each harmonic
    vector[S] sigma;                      // sigma of each harmonic
    int k[M,C,P];                         // number of heads
    int n[M,C,P];                         // number of flips
    int cholesky_n_m;                     // sparse cholesky in magnitude - number of nonzero elements
    row_vector[cholesky_n_m] cholesky_w_m;// sparse cholesky in magnitude - nonzero elements
    int cholesky_v_m[cholesky_n_m];       // sparse cholesky in magnitude - columns of nonzero elements
    int cholesky_u_m[M+1];                // sparse cholesky in magnitude - where in w each row starts
    int cholesky_n_c;                     // sparse cholesky in colour - number of nonzero elements
    vector[cholesky_n_c] cholesky_w_c;    // sparse cholesky in colour - nonzero elements
    int cholesky_v_c[cholesky_n_c];       // sparse cholesky in colour - columns of nonzero elements
    int cholesky_u_c[C+1];                // sparse cholesky in colour - where in w each row starts
}
parameters {
    matrix[M_subspace,C_subspace] z[S];
}
transformed parameters {

    vector[P] x[M,C]; // Probability in logit-space
        
    // Loop over magnitude and colour
    for (m in 1:M){
        for (c in 1:C){

            // Local variables
            vector[S] a;
            matrix[R,L] F;

            // Compute a 
            for (s in 1:S){
                a[s] = mu[s] + sigma[s] * (cholesky_w_m[cholesky_u_m[m]:cholesky_u_m[m+1]-1] * z[s,cholesky_v_m[cholesky_u_m[m]:cholesky_u_m[m+1]-1], cholesky_v_c[cholesky_u_c[c]:cholesky_u_c[c+1]-1]] * cholesky_w_c[cholesky_u_c[c]:cholesky_u_c[c+1]-1]);
            }

            // Compute F
            for (l in 1:L) {
                F[:,l] = lambda[:,lower[l]:upper[l]] * a[lower[l]:upper[l]];
            }

            // Compute x
            for (p in 1:P){
                x[m,c,p] = dot_product(F[pixel_to_ring[p]],azimuth[:,p]);
            }

        }  
    }

}
model {

    // Prior
    for (s in 1:S){
        to_vector(z[s]) ~ std_normal();
    }
    
    // Likelihood
    for (m in 1:M){
        for (c in 1:C){
            k[m,c] ~ binomial_logit(n[m,c], x[m,c]);
        }
    }
    
}
