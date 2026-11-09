Searches for primitive integer solutions of **a⁴ + b⁴ + c⁴ = d⁴** 
using factorization of (d - c)(d + c)(d² + c²).

It scans candidate pairs (c, d) with fast modular filters and residue
tables, factors the reduced values, and reconstructs a and b via
Gaussian-integer (sum-of-two-squares) representations. 64-bit
factorization uses an SPF table plus Miller–Rabin/Lucas and Pollard
rho (Brent). Parallelized with OpenMP.

## Build

    gcc -O3 -fopenmp -march=native -DNDEBUG euler4.c -o search

## Run

    ./search [end]            # search d from 1 to end (default 3000000)
    ./search <start> <end>    # search a d range

