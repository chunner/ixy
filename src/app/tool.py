import accel_ip
import torch
from torch.autograd import Function
import torch.nn as nn

def quantize_tensor(tensor, mode):
    """
    Quantize the input tensor based on the specified mode.
    Mode: 0-Int8, 1-Int4, 2-FP16, 3-FP32
    return quantized_tensor, scale
    """
    if mode == 3: # FP32
        return tensor.contiguous(), 1.0
    elif mode == 2: # FP16
        return tensor.half().contiguous(), 1.0
    elif mode == 0: # Int8
        abs_max = tensor.abs().max()
        scale = 127.0 / abs_max
        quantize_tensor = (tensor * scale).round().clamp(-128, 127).to(torch.int8)
        return quantize_tensor.contiguous(), scale
    elif mode == 1: # Int4
        abs_max = tensor.abs().max()
        scale = 7.0 / abs_max
        quantize_tensor = (tensor * scale).round().clamp(-8, 7).to(torch.int8)

def dequantize_tensor(tensor, scale, mode):
    """
    Dequantize the input tensor to FP32 based on the specified mode.
    """
    if mode in [0, 1]:
        return tensor.float() / scale
    return tensor.float()


class FPGAMatMulFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, instance_ptr, mode):
        """
        input : [Batch, Sequence, In_Feature] -> Matrix A [N, K]
        weight: [Out_Feature, In_Feature] -> Matrix B [M, K]
        """
        # 1. dimension handling
        orig_shape = input.shape
        N = input.numel() // input.shape[-1]
        K = input.shape[-1]
        M = weight.shape[0]
        input_flat = input.view(N, K)
        weight_T = weight.t().contiguous() # [K, N]
        q_weight, scale_wt = quantize_tensor(weight_T, mode)
        # 3. config output tensor
        final_dtype = torch.float32
        final_output = torch.zeros((N, M), dtype=final_dtype, device='cpu')
        hw_output_dtype = torch.int32 if mode in [0, 1] else final_dtype
        # 4. FPGA matmul
        MAX_HW_N = 64
        for start_row in range(0, N, MAX_HW_N):
            end_row = min(start_row + MAX_HW_N, N)
            current_chunk_rows = end_row - start_row

            input_chunk = input_flat[start_row:end_row, :] # [current_chunk_rows, K]
            q_input_chunk , scale_in = quantize_tensor(input_chunk, mode)
            output_chunk = torch.zeros((current_chunk_rows, M), dtype=hw_output_dtype, device='cpu').contiguous()

            accel_ip.xmmult_mixed_execute(
                instance_ptr,
                q_input_chunk.data_ptr(),
                q_weight.data_ptr(),
                output_chunk.data_ptr(),
                current_chunk_rows,
                K,
                M,
                mode,
                1, # updateA
                q_input_chunk.nbytes,
                q_weight.nbytes,
                output_chunk.nbytes
            )
            output_chunk = dequantize_tensor(output_chunk, scale_in * scale_wt, mode)
            final_output[start_row:end_row, :] = output_chunk
        if bias is not None:
            final_output += bias.unsqueeze(0)
        final_output = final_output.view(*orig_shape[:-1], M)
        
        return final_output 

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward pass is not implemented for FPGAMatMulFunction.")


class FPGALinear(nn.Module):
    def __init__(self, original_linear, mode, instance_ptr, mem_manager):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features

        # copy weights and bias
        self.weight = nn.Parameter(original_linear.weight.data.clone())
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone())
        else:
            self.register_parameter('bias', None)

        self.mode = mode
        self.instace_ptr = instance_ptr
        
    def forward(self, input):
        return FPGAMatMulFunction.apply(
            input,
            self.weight,
            self.bias,
            self.instace_ptr,
            self.mode
        )


def replace_bert_layers(model, config_map, instance_ptr, path=''):
    """
    递归遍历模型，替换 Linear 层
    config_map: 字典，key 是层名称(regex或全名), value 是 mode (int)
    """
    for name, module in model.named_children():
        curr_path = f"{path}.{name}" if path else name
        
        if isinstance(module, nn.Linear):
            # 确定当前层应该用什么精度
            # 默认 FP32 (Mode 3)，如果在 config_map 里则覆盖
            selected_mode = 3 
            for layer_key, mode in config_map.items():
                if layer_key in curr_path:
                    selected_mode = mode
                    break
            
            # 只有当非 FP32 或显式要求加速时才替换
            if selected_mode != 3: 
                print(f"Replacing {curr_path} with FPGA Mode {selected_mode}")
                fpga_layer = FPGALinear(module, selected_mode, instance_ptr)
                setattr(model, name, fpga_layer)
        
        else:
            # 递归处理子模块 (如 Attention 块内部)
            replace_bert_layers(module, config_map, instance_ptr, curr_path)