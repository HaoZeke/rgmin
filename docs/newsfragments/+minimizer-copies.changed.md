An interior point skips the bounds clip, and a Euclidean session keeps the gradient it already owns.
Polak-Ribiere forms `g · (g - g_old)` in the dot product.
A backtracking line search keeps its lowest trial without copying that vector.
FIRE 2.0 does not build the post-step force, which only FIRE 1.0 mixes.
L-BFGS reuses the curvature dots stored with each pair.
The Morse-chain iterates are unchanged.
