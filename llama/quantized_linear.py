import torch

from typing import Callable, Optional
from torch import nn
import torch.nn.init as init
from torch.nn.parameter import Parameter
import torch.nn.functional as F

import math
from fairscale.nn.model_parallel.initialize import get_model_parallel_rank, get_model_parallel_world_size
from fairscale.nn.model_parallel.utils import divide_and_check_no_remainder
from fairscale.nn.model_parallel.mappings import (
    copy_to_model_parallel_region,
    gather_from_model_parallel_region,
    reduce_from_model_parallel_region,
    scatter_to_model_parallel_region,
)


def _initialize_affine_weight(
    weight: torch.Tensor,
    out_features: int,
    in_features: int,
    per_partition_size: int,
    partition_dim: int,
    init_method: Callable[[torch.Tensor], torch.Tensor],
    stride: int = 1,
    return_master_weight: bool = False,
) -> Optional[torch.Tensor]:
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    # If we only use 1 process for model parallelism, bypass scatter.
    world_size = get_model_parallel_world_size()
    if world_size == 1:
        init_method(weight)
        if return_master_weight:
            return weight
        return None

    # Initialize master weight
    master_weight = torch.empty(out_features, in_features, dtype=weight.dtype, requires_grad=False)
    init_method(master_weight)

    # Split and copy
    per_partition_per_stride_size = divide_and_check_no_remainder(per_partition_size, stride)
    weight_list = torch.split(master_weight, per_partition_per_stride_size, dim=partition_dim)
    rank = get_model_parallel_rank()
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None

    
        





class RowParallelQuantizedLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        in_features: first dimension of matrix A.
        out_features: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = False,
        init_method: Callable[[torch.Tensor], torch.Tensor] = init.xavier_normal_,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
        bits: int = 4,
        group_size: int = -1
    ):
        super(RowParallelQuantizedLinear, self).__init__()

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        world_size = get_model_parallel_world_size()
        self.input_size_per_partition = divide_and_check_no_remainder(in_features, world_size)
        
        if group_size != -1 and group_size < 32 and group_size != int(math.pow(2,int(math.log2(group_size)))):
            raise NotImplementedError("groupsize supports powers of 2 greater than 32. (e.g. : 32,64,128,etc)")
        group_size = group_size if group_size != -1 else  self.input_size_per_partition
        self.group_size = group_size
        self.register_buffer(
            'qzeros', 
            torch.zeros(
                (
                    math.ceil(self.input_size_per_partition/group_size),
                    out_features // 32 * bits 
                ), 
                dtype=torch.int))
        self.register_buffer('scales', torch.zeros((math.ceil(self.input_size_per_partition/group_size),out_features), dtype=torch.float32))
        
        self.register_buffer(
            'qweight', torch.zeros((self.input_size_per_partition // 32 * bits, out_features), dtype=torch.int)
        )
        
        if bias:
            self.register_buffer('bias', torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
            
        
        self.bits = 4
        self.register_buffer(
            'wf1',
            torch.tensor([0, 4, 8, 12, 16, 20, 24, 28], dtype=torch.int32).unsqueeze(0).unsqueeze(2),
            persistent=False
        )
        self.register_buffer(
            'wf2',
            torch.tensor([0, 4, 8, 12, 16, 20, 24, 28], dtype=torch.int32).unsqueeze(0).unsqueeze(0),
            persistent=False
        )
        
    @classmethod
    def from_float(self,module):
        quant_module = RowParallelQuantizedLinear(
            module.in_features,
            module.out_features,
            module.bias is None,
            module.input_is_parallel,
        )
        return quant_module
        
    def get_master_weight(self) -> torch.Tensor:
        return gather_from_model_parallel_region(self.weight.data)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type:ignore
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_model_parallel_region(input_)
            
            
        
        # Unpack 4bit weights
        weight = torch.bitwise_right_shift(torch.unsqueeze(self.qweight, 1).expand(-1, 8, -1), self.wf1).to(torch.int8)
        torch.bitwise_and(weight, 0x0000000F, out=weight)
        weight = weight.reshape(-1, self.group_size, weight.shape[2])

        zeros = torch.bitwise_right_shift(torch.unsqueeze(self.qzeros, 2).expand(-1, -1, 8), self.wf2).to(torch.int8)
        torch.bitwise_and(zeros, 0x0000000F, out=zeros)
        zeros = zeros + 1
        zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])

        scales = self.scales
        scales = scales.reshape(-1, 1, scales.shape[-1])

        weights = (scales * (weight - zeros))
        weights = weights.reshape(weights.shape[0] * weight.shape[1], weights.shape[2])
        output_parallel = torch.matmul(input_parallel, weights.to(input_parallel.dtype))
        # print('input:', input_parallel[0].sum() )
        # print('output:', output_parallel[0].sum())
        # import pdb;pdb.set_trace()
        # All-reduce across all the partitions.
        output_ = reduce_from_model_parallel_region(output_parallel)
        if self.bias is not None:
            output = output_ + self.bias
        else:
            output = output_
        return output
    
    
class ColumnParallelQuantizedLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        in_features: first dimension of matrix A.
        out_features: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gether on output and make Y avaiable
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = True,
        init_method: Callable[[torch.Tensor], torch.Tensor] = init.xavier_normal_,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
        bits: int = 4,
        group_size: int = -1
    ) -> None:
        super(ColumnParallelQuantizedLinear, self).__init__()

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        world_size = get_model_parallel_world_size()
        self.output_size_per_partition = divide_and_check_no_remainder(out_features, world_size)

        if group_size != -1 and group_size < 32 and group_size != int(math.pow(2,int(math.log2(group_size)))):
            raise NotImplementedError("groupsize supports powers of 2 greater than 32. (e.g. : 32,64,128,etc)")
        group_size = group_size if group_size != -1 else in_features
        self.group_size = group_size
        self.register_buffer(
            'qzeros', 
            torch.zeros(
                (
                    math.ceil(in_features/group_size),
                    self.output_size_per_partition // 32 * bits 
                ), 
                dtype=torch.int))
        self.register_buffer('scales', torch.zeros((math.ceil(in_features/group_size),self.output_size_per_partition), dtype=torch.float32))
        
        self.register_buffer(
            'qweight', torch.zeros((in_features // 32 * bits, self.output_size_per_partition), dtype=torch.int)
        )
        
        if bias:
            self.register_buffer('bias', torch.zeros(self.output_size_per_partition))
        else:
            self.register_parameter("bias", None)
            
        
        self.bits = 4
        self.register_buffer(
            'wf1',
            torch.tensor([0, 4, 8, 12, 16, 20, 24, 28], dtype=torch.int32).unsqueeze(0).unsqueeze(2),
            persistent=False
        )
        self.register_buffer(
            'wf2',
            torch.tensor([0, 4, 8, 12, 16, 20, 24, 28], dtype=torch.int32).unsqueeze(0).unsqueeze(0),
            persistent=False
        )
        
    @classmethod
    def from_float(self,module):
        quant_module = ColumnParallelQuantizedLinear(
            module.in_features,
            module.out_features,
            module.bias is None,
            module.gather_output,
        )
        return quant_module

    def get_master_weight(self) -> torch.Tensor:
        return gather_from_model_parallel_region(self.weight.data.transpose(0, 1)).transpose_(0, 1)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Set up backprop all-reduce.
        input_parallel = copy_to_model_parallel_region(input_)
        
        
        # Unpack 4bit weights
        weight = torch.bitwise_right_shift(torch.unsqueeze(self.qweight, 1).expand(-1, 8, -1), self.wf1).to(torch.int8)
        torch.bitwise_and(weight, 0x0000000F, out=weight)
        weight = weight.reshape(-1, self.group_size, weight.shape[2])

        zeros = torch.bitwise_right_shift(torch.unsqueeze(self.qzeros, 2).expand(-1, -1, 8), self.wf2).to(torch.int8)
        torch.bitwise_and(zeros, 0x0000000F, out=zeros)
        zeros = zeros + 1
        zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])

        scales = self.scales
        scales = scales.reshape(-1, 1, scales.shape[-1])

        weights = (scales * (weight - zeros))
        weights = weights.reshape(weights.shape[0] * weight.shape[1], weights.shape[2])
        output_parallel = torch.matmul(input_parallel, weights.to(input_parallel.dtype))
        # print('input:', input_parallel[0].sum() )
        # print('output:', output_parallel[0].sum())
        # import pdb;pdb.set_trace()
        if self.bias is not None:
            output = output_parallel + self.bias
        else:
            output = output_parallel
            
        if self.gather_output:
            # All-gather across the partitions.
            print(output_parallel.sum())
            import pdb;pdb.set_trace()
            output = gather_from_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        return output



