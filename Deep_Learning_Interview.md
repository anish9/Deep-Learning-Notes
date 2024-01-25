1 - Why VGG is called VGG16 and how many parameters it has?
* The 16 in VGG16 refers to it has 16 layers that have weights. This network is a pretty large network and it has about 138 million (approx) parameters.

2 - Explain the arrangement of VGG network?
* It follows this arrangement of convolution and max pool layers consistently throughout the whole architecture. In the end it has 2 FC(fully connected layers) followed by a softmax for output. 

3 - Explain a scenario where you prefer to use seperable convolution2D instead of standard convolution2D?
* When number of network parameters has to be reduced we can opt seperable Conv2D.

4- Explain a scenario when it is more suitable to use 1x1 kernel ?
* Ideally when the number of computations has to come down we can use 1x1 kernel.

5- which is parallelizable Transformers or RNN? 
* RNN by their training nature are not parallelizable but Transformers are parallelizable.

6- When paging memory is useful for performance gain in model training ?
* Paging memory is a memory management technique used in operating systems to enable processes to access more memory than is physically 
    available. It works by dividing both physical memory 
    (RAM) and secondary storage (hard disk) into fixed-size blocks called pages and frames, respectively. This allows the operating system to 
    load only the pages of a process that are currently needed into RAM, while the rest of the process can reside on the hard disk. 
    Example : DataLoader can make use of paging memory for performance gain.


