# Simulation for "Decentralized Expectation Consistent Signal Recovery for Phase Retrieval"
(c) 2020 Chang-Jen Wang and Chao-Kai Wen e-mail: dkman0988@gmail.com and chaokai.wen@mail.nsysu.edu.tw

--------------------------------------------------------------------------------------------------------------------------
# Information:
- IDE: Iterative discrete estimation
- IDE2: Low-complexity version of Iterative discrete estimation

IDE and IDE2 are efficient algorithms for a downlink massive MU-MIMO system with finite-alphabet precodings. For details, please refer to 

C. J. Wang, C. K. Wen, S. Jin, and S. H. Tsai, Finite-Alphabet Precoding for Massive MU-MIMO with Low-resolution DACs, IEEE Trans. Wireless Commun., 2018, to appear.

We provide the codes in a way that you can perfrom based on the simulator for "Quantized Precoding for Massive MU-MIMO". Therefore, you can compare severeal different precoding algorithms under the same setting.


# How to start a simulation:

- Step 1. Download the simulator for "Quantized Precoding for Massive MU-MIMO":

  https://github.com/quantizedmassivemimo/1bit_precoding


- Step 2. Download our proposed precoders (IDE.m & IDE2.m), which can be found

  https://github.com/Wangchangjen/Matlab_IDE


- Step 3. In precoder_sim.m, find the line 

  par.precoder = … 

  Replace the line by
  
  par.precoder = {'IDE','SQUID','IDE2','SDR1','SDRr'}; % select precoding scheme(s) to be evaluated
  
  
- Step 4. In precoder_sim.m, find the line

  switch (par.precoder{pp}) 

  Include the cases

    case 'IDE'

    [x, beta] = IDE(par.s,Hhat,N0);

    case 'IDE2'

    [x, beta] = IDE2(par.s,Hhat,N0);


- Step 5. Now, you are ready to run the precodes:

  precoder_sim

--------------------------------------------------------------------------------------------------------------------------------------
The simulator returns a plot of the BER as a function of the SNR.
<div align=center><img width="600" height="600" src="https://github.com/Wangchangjen/Matlab_deGEC-SR/blob/master/EXAMPLE.fig"/></div>

