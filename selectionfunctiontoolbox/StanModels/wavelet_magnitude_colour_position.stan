data {
    int<lower=0> P;                       // number of pixels
    int<lower=0> M;                       // number of bins in magnitude space
    int<lower=0> M_subspace;              // number of inducing points in magnitude space
    int<lower=0> C;                       // number of bins in colour space
    int<lower=0> C_subspace;              // number of inducing points in colour space
    int<lower=0> S;                       // number of wavelets
    int wavelet_n;                        // sparse wavelets - number of nonzero elements
    vector[wavelet_n] wavelet_w;          // sparse wavelets - nonzero elements
    int wavelet_v[wavelet_n];             // sparse wavelets - columns of nonzero elements
    int wavelet_u[P+1];                   // sparse wavelets - where in w each row starts
    vector[S] mu;                         // mean of each wavelet
    vector[S] sigma;                      // sigma of each wavelet
    int k[M,C,P];                         // number of heads
    int n[M,C,P];                         // number of flips
    row_vector[M_subspace] cholesky_m[M]; // Cholesky factor in magnitude space
    vector[C_subspace] cholesky_c[C];     // Cholesky factor in colour space
}
parameters {
    matrix[M_subspace,C_subspace] z[S];
}
transformed parameters {

    vector[P] x[M,C]; // Probability in logit-space
    
    // Loop over magnitude and colour
    for (m in 1:M){
        for (c in 1:C){
            
            // Local variable
            vector[S] b;
                
            // Compute b
            for (s in 1:S){
                b[s] = mu[s] + sigma[s] * cholesky_m[m] * z[s] * cholesky_c[c];
            }
                
            // Compute x
            x[m,c] = csr_matrix_times_vector(P, S, wavelet_w, wavelet_v, wavelet_u, b);

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
