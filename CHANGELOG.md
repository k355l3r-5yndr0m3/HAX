# Revision history for Yam

## 0.1.0.0 -- 2023-mm-dd

* First version. Released on an unsuspecting world.

## 0.2.0.0 -- 2023-08-31
* Implemented the jit function used to convert functions defined in haskell to runtime compiled executable for accelerated tensor computation
* Implemented the rgrad function for reverse mode differentiation, the function output from it is intended to be consumed by jit
* Implemented the auto function, which convert an expression of type ===Tensor s t=== to ===Tracer s t=== to feed into a jitable function and have that compiled with an embeded constant

## 0.2.0.1 -- 2023-09-05
* Implemented broadcasting and tensor product
* Implemented VMap for tracers
