# ViolaJonesCUDA - NVIDIA accelerated face detection

This project was adapted from an assignment, published by Technische
Universiteit Eindhoven (TU/e), as outlined here:
https://sites.google.com/site/5kk73gpu2012/assignment/viola-jones-face-detection

Our tests revealed that its performance was 14-15 times faster than the C++
implementation; however, that boilerplate lacked optimization even within C++.

I developed this CUDA implementation as part of a project for EE 5351 at the
University of Minnesota, with James Zhang, Ahmed Elshamekh, and Eric Kahle.

This could be optimized further by completing the CUDA scan and translate
kernels, as well as experimenting with different segmentations of the cascade
(currently, there are 6 cascade segment kernels comprising 25 stages).

### License
This code, as with the C++ code that informed its operation, is released under
the GNU GPL v3.
