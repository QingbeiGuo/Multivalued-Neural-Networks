Multivalued Weight Neural Networks: Quantization-aware Training for Deep Network Compression
======

Abstract：Deep neural networks have substantially achieved the state-of-the-art performance in various tasks, relying on deep network architectures and numerous parameters. However, the requirements for heavy computations and memories impede their promoted applications on the embedded and mobile devices with limited resources. We introduce a novel multivalued quantization approach, called as MVQ, explicitly exploiting existing resources to match an optimal quantitative level for the tradeoff between compression and performance. We first partitions the weight range of each convolutional layer into multiple intervals. Next, we iteratively exploit interval shift and interval contraction to update these intervals and clip all the weights within their updated intervals. %, to our best knowledge, which is the first to quantize deep network models.
Finally, we enforce weight sharing for each interval. Theoretically, the multivalued quantization can be converted into specific ternary quantization with multiple scaling factors, which means that our method can offer more expressive power than existing binary and ternary counterparts. % and has few multiplications simultaneously. %%Furthermore, we can further sparsify the quantized weights by truncating the least important scaling factors.
The comprehensive empirical results demonstrate that our method can provide customized compression services for the embedded and mobile devices constrained by the hardware resources with different sizes. Within the same quantization level as binary and ternary networks, MVQ is significantly comparable/superior to them. Furthermore, for higher quantization levels which is unfeasible to these binary and ternary approaches, our method achieves higher performance in recognition accuracy. Our code can be found at:
\url{https://github.com/QingbeiGuo/Multivalued-Neural-Networks.git}.

摘要：深度神经网络在各种识别任务中已经取得了最该的水准，这依赖于深的网络框架和大量的参数。然而，对计算和内存的广泛需求阻碍了它向有限资源的嵌入式和移动设备的扩展应用。我们介绍了一个新颖的多值量化方法（简称为MVQ），它明确应用设备的现有资源匹配一个最优的量化水平来平衡压缩和性能。我们的方法首先划分每层的权重为多个区间，然后，我们利用均值移动和区间收缩来迫使每个区间权重共享。据我们所知，这种方法是我们第一次提出量化深度网络模型。理论上，这个多值量化可被转化为拥有多个缩放因子的特殊三值量化，这意味着我们的方法比存在的二值和三值网络提供更大的表达能力。大量的实验结果展示出我们的方法能为拥有有限资源的嵌入式和移动设备提供了最佳的压缩率和性能的定制性压缩服务。在与二值和三值网络同样的量化水平，我们的MVQ方法比得上它们。对于更高的量化水平，之前的二值和三值网络是做不到的，而我们的方法不仅能够实现而且获得更高的性能。
