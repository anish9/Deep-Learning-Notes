Unbox Deep Learning - Systems deep:
1) When paging memory is useful for performance gain in model training? <br>
   Paging memory is a memory management technique used in operating systems to enable processes to access more memory than is physically available. It works by dividing both physical memory 
   (RAM) and secondary storage (hard disk) into fixed-size blocks called pages and frames, respectively. This allows the operating system to load only the pages of a process that 
   are currently needed into RAM, while the rest of the process can reside on the hard disk. 
   Example : DataLoader can make use of paging memory for performance gain.


2) Calculating Model Informations:
   Give a number of parameters of a neural net, first find what is the precision of each parameter(float16, float32 or float64).<br />
   Formula is simple now,             total_memory = (number of paramters * precision_expressed_in_bytes) <br />
                                      Example :  PARAMTETERS : 2300012, PRECISION :  float32 (32/8=4 BYTES) <br />
                                      total_memory = 23,00,012*4 <br />
                                      1 MB = 1,048,576 bytes <br />
                                      to_get_model_size_in_mb = total_memory/ 1,048,576 <br />

