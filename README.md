BitBrain is a learning algorithm based upon a novel synthesis of ideas from sparse coding, computational neuroscience and information theory that support single-pass learning, accurate and robust inference, and the potential for continuous adaptive learning. They are designed to be
implemented efficiently on current and future neuromorphic devices as well as on more
conventional CPU and memory architectures.

This work was presented during NICE conference 2022: https://flagship.kip.uni-heidelberg.de/jss/HBPm?mI=235&publicVideoID=8944.

How to run the code:
1. Install all required Python libraries: cv2, numpy, scipy
2. Compile the C library: e.g. cc -fPIC -shared -o MH_comp.so MH_lib.c
3. Run the Python script: python BitBrain_Python_MH.py
