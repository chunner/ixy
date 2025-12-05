import accel_ip
import torch
import numpy as np
import time
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

def call_fpga(A_buf, B_buf, C_buf, accel_ptr, N, K, M, update_A):
    """
    Runs a 2D matrix multiplication on the FPGA accelerator:
      (N, K) x (K, M) => (N, M)

    update_A: 1 to load A into BRAM (new input), 0 to reuse persistent A.
    """
    accel_ip.xmmult_accel_execute(
        accel_ptr,
        A_buf,
        B_buf,
        C_buf,
        N,
        K,
        M,
        update_A
    ) 
   


# # Helper Functions
# - **requantize()**: Converts int32 arrays to int8 using a scaling factor and zero point.
# - **display_model_confidence()**: Converts logits to human-readable class confidence.
def requantize(int32_array, scale, zero_point=0):
    """
    Requantizes an int32 numpy array to int8 using the provided scale and zero_point.
    Operation: int8_val = clip(round(int32_val * scale + zero_point), -128, 127)
    """
    scaled = np.round(int32_array * scale + zero_point)
    int8_array = np.clip(scaled, -128, 127).astype(np.int8)
    return int8_array

def display_model_confidence(logits, device_name="Model"):
    """
    Converts logits to probabilities and prints a user-friendly confidence message.

    Parameters:
    logits (torch.Tensor): The raw model output (logits).
    device_name (str): Name of the device (e.g., "CPU", "FPGA") for comparison.
    """
    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=1)

    # Get predicted class and confidence
    predicted_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0, predicted_class].item() * 100

    # Print result
    print(f"{device_name}: The model is {confidence:.2f}% confident in predicting class {predicted_class}.")



# # Custom Module for FPGA Offload
# #### FPGA-Optimized Linear Layer for Q, K, V Projections
# This block defines **FPGAQuantizedLinear**, a custom PyTorch module that replaces  
# the standard linear layers with FPGA-accelerated equivalents. It:
# - **Quantizes activations** before computation.
# - **Uses PYNQ buffers** to store inputs and weights.
# - **Invokes the FPGA accelerator** for matrix multiplications.
# - **Dequantizes the result** back to floating point.
# This module is later integrated into DistilBERT for hardware acceleration.
class FPGAQuantizedLinear(torch.nn.Module):
    def __init__(self, quantized_linear, act_scale, accel_ptr, hidden_size=768, update_A=True):
        """
        Parameters:
          quantized_linear : an instance of DynamicQuantizedLinear from the quantized model.
          act_scale        : scaling factor for quantizing input activations.
          accel_ptr         : the FPGA accelerator IP handle.
          hidden_size      : hidden dimension size (typically 768).
          update_A         : flag indicating whether to update A in persistent BRAM (True for Q, False for K/V).
        """
        total_fpga_compute_time = 0.0
        call_count = 0
        
        super(FPGAQuantizedLinear, self).__init__()
        self.accel_ptr = accel_ptr
        self.hidden_size = hidden_size
        self.act_scale = act_scale
        self.update_A = update_A  # Store the update flag
        
        # Extract quantized weight and its parameters.
        self.weight_int8_tensor = quantized_linear.weight().int_repr()
        self.weight_scale = quantized_linear.weight().q_scale()
        self.weight_zero_point = quantized_linear.weight().q_zero_point()
        # Transpose so that the weight shape becomes (in_features, out_features)
        self.weight_int8 = self.weight_int8_tensor.cpu().numpy().T  # shape: (hidden_size, hidden_size)
        self.weight_int8 = np.ascontiguousarray(self.weight_int8)  # Ensure contiguous layout
        
        # Effective scale: multiplication of activation scale and weight scale.
        self.effective_scale = self.act_scale * self.weight_scale
        
        # Check for bias. Note that in DynamicQuantizedLinear, bias remains in FP32.
        bias_val = quantized_linear.bias()  # This calls the bound method.
        if bias_val is not None:
            # Save bias as a NumPy array (shape: (hidden_size,))
            self.bias = bias_val.detach().cpu().numpy().astype(np.float32)
        else:
            self.bias = None

    def forward(self, x):
        """
        Forward pass for FPGA offload.
        Accepts input x which may be 2D (N, D) or 3D (B, S, D). In case of 3D input,
        the tensor is reshaped to 2D for matrix multiplication and then reshaped back.
        The input is quantized to int8 using self.act_scale. After the FPGA multiplication,
        the int32 result is dequantized to FP32 and the bias is added (if available).
        """
        # Save the original shape.
        orig_shape = x.shape
        if x.dim() == 3:
            B, S, D = x.shape
            x_flat = x.reshape(B * S, D)
        else:
            x_flat = x

        # Determine the number of rows for the FPGA call.
        N = x_flat.shape[0]

        # Quantize the input if it is in float32.
        if x_flat.dtype == torch.float32:
            x_int8 = torch.clamp(torch.round(x_flat / self.act_scale), -128, 127).to(torch.int8)
        else:
            x_int8 = x_flat

        # Convert to a NumPy int8 array.
        x_np = x_int8.cpu().numpy().astype(np.int8)
        x_np = np.ascontiguousarray(x_np)  # Ensure contiguous layout
        C_np = np.empty((N, self.hidden_size), dtype=np.int32)
        C_np = np.ascontiguousarray(C_np)  # Ensure contiguous layout
        
        # Convert input activation and weight to PYNQ buffers.
        A_buf = x_np.ctypes.data
        W_buf = self.weight_int8.ctypes.data
        C_buf = C_np.ctypes.data
        
        # Call the FPGA accelerator:
        # Instead of hardcoding update_A=1, we now use self.update_A:
        # Time just the FPGA computation
        start_fpga = time.time()
        call_fpga(______, _______, ________, ___________, ___, _________, __________, ____________)
        fpga_duration = time.time() - start_fpga
        
        FPGAQuantizedLinear.total_fpga_compute_time += fpga_duration
        FPGAQuantizedLinear.call_count += 1
        
        # Retrieve the int32 result.
        C_int32 = np.array(C_np)
        # Dequantize: convert int32 accumulator to FP32 using the effective scale.
        out_fp32 = C_int32.astype(np.float32) * self.effective_scale
        
        # If a bias is present, add it (broadcast along axis 0).
        if self.bias is not None:
            # Ensure bias is added to each row.
            out_fp32 = out_fp32 + self.bias
        
        # Convert back to a torch tensor.
        out_tensor = torch.tensor(out_fp32, dtype=torch.float32)
        
        # If the original input was 3D, reshape back to (B, S, hidden_size).
        if x.dim() == 3:
            out_tensor = out_tensor.reshape(B, S, self.hidden_size)
        return out_tensor

# # Replacing Q, K, V Layers with FPGA Versions
# This function walks through all transformer layers in the quantized DistilBERT model  
# and replaces the **Q, K, and V projection layers** with the custom **FPGAQuantizedLinear** module.
# - Ensures **Q projection updates A in BRAM** (update_A=True).
# - **K and V projections reuse A** for efficiency.
# - Enables model acceleration while preserving transformer layer structure.

def integrate_fpga_offload(model_quant, act_scale, accel_ptr, hidden_size=768):
    """
    Replaces the Q, K, V projection layers in each transformer layer with the FPGA-accelerated custom module.
    
    Parameters:
      model_quant  : Quantized DistilBertForSequenceClassification model.
      act_scale    : Scaling factor for quantizing activations (assumed same for demo).
      accel_ip     : Configured FPGA accelerator IP handle.
      hidden_size  : Hidden dimension (typically 768).
    """
    for layer in model_quant.distilbert.transformer.layer:
        # For the Q projection, set update_A to True so that the persistent A is updated.
        layer.attention.q_lin = FPGAQuantizedLinear(layer.attention.q_lin, act_scale, accel_ptr, hidden_size, update_A=True)
        # For K and V projections, set update_A to False to reuse A from BRAM.
        layer.attention.k_lin = ________________________________________________________________ 
        layer.attention.v_lin = ________________________________________________________________ 


def compute_activation_scale(activation_list, percentile=99.9, use_demo=0):
    """
    Computes a global activation scale from a calibration set of activations.
    
    Parameters:
      activation_list: List of NumPy arrays representing activations 
                       (for example, from the embedding layer).
      percentile:      The percentile to use for robust scale computation (if use_demo=0).
      use_demo:        If set to 1, uses the demo method (scale = max_abs_value/127.0);
                       otherwise, uses the robust method (scale = percentile_value/127.0).
    
    Returns:
      A scaling factor computed as:
         - Demo method: scale = (max(|activations|)) / 127.0
         - Robust method: scale = (percentile(|activations|)) / 127.0
    """
    # Concatenate all activations from the calibration samples into one array.
    all_activations = np.concatenate([act.flatten() for act in activation_list])
    
    if use_demo:
        # Demo method: use the maximum absolute value.
        act_abs_max = np.max(np.abs(all_activations))
        scale = act_abs_max / 127.0 if act_abs_max != 0 else 1.0
    else:
        # Robust method: use the specified percentile.
        act_abs_percentile = np.percentile(np.abs(all_activations), percentile)
        scale = act_abs_percentile / 127.0 if act_abs_percentile != 0 else 1.0
    
    return scale

# ==================================================================================
#                               MAIN EXECUTION BLOCK
# ==================================================================================
# This block demonstrates how to:
# 1. **Load and quantize a DistilBERT model**.
# 2. **Extract activations** from the embedding layer.
# 3. **Integrate FPGA acceleration** into transformer layers.
# 4. **Run a forward pass** through the modified model.
# Only the **Q, K, and V projections** are offloaded to FPGA; the remaining layers run on CPU/GPU.

pci_addr = "0000:00:04.0"
accel_ptr = accel_ip.xmmult_accel_device_init(pci_addr)

# 1. Load and Quantize the Model
model_name = "distilbert-base-uncased-finetuned-sst-2-english" # distilBERT fine-tuned on SST-2 for sentiment classification (positive/negative)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)
model.eval()

# Apply dynamic quantization to convert Linear layers to int8.
model_int8 = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=_______)
# Set to eval mode
model_int8.eval()

# 2. Gather a Calibration Set of Activations to Compute a Global Activation Scale
calib_sentences = [
    "The moonlight shimmered over the ocean as waves gently kissed the sandy shore, while distant lanterns flickered in the cool evening breeze. A lone traveler wandered along the coastline, footsteps sinking softly into the damp sand, lost in thought. The rhythmic sound of the water mixed with the rustling palms, creating a nice"
]
calib_activations = []
for sentence in calib_sentences:
    inputs = tokenizer(sentence, return_tensors="pt")   # shape: (1, L)
    with torch.no_grad():
        # Get the embedding output; shape: (B, L, 768). Here B=1.
        emb = model.distilbert.embeddings(inputs.input_ids)  # shape: (1, L, 768)
        # Remove the batch dimension.
        emb = emb.squeeze(0)  # shape: (L, 768)
        calib_activations.append(emb.cpu().numpy())

# Compute the activation scale using the robust method (percentile-based):
# global_act_scale_robust = compute_activation_scale(calib_activations, percentile=99.9, use_demo=0)
# print("Global Activation Scale (Robust):", global_act_scale_robust)

# # Compute the activation scale using the demo method (max-based):
global_act_scale_demo = compute_activation_scale(calib_activations, use_demo=1)
print("Global Activation Scale (Demo):", global_act_scale_demo)

test_sentence = calib_sentences[0]
print(f"input = '{test_sentence}'")

# CPU-only Inference
inputs = tokenizer(test_sentence, return_tensors="pt")

start_time = time.time()
with torch.no_grad():
    outputs_cpu = model(inputs.input_ids)
    logits_cpu = outputs_cpu.logits
cpu_time = time.time() - start_time

print(f"CPU Inference Time: {cpu_time:.6f} seconds")
# print("CPU Logits:", logits_cpu)

# FPGA-Offloaded Inference
integrate_fpga_offload(model_int8, global_act_scale_demo, accel_ptr, hidden_size=768)

# Reset the timing counters before inference
FPGAQuantizedLinear.total_fpga_compute_time = 0.0
FPGAQuantizedLinear.call_count = 0

# Run inference normally with your existing code
start_time = time.time()
with torch.no_grad():
    outputs_fpga = model_int8(inputs.input_ids)
    logits_fpga = outputs_fpga.logits
fpga_time = time.time() - start_time

# After inference, report the detailed timing
print(f"FPGA calls: {FPGAQuantizedLinear.call_count}")
print(f"Total time in FPGA compute: {FPGAQuantizedLinear.total_fpga_compute_time:.6f} seconds")
print(f"Average time per FPGA call: {FPGAQuantizedLinear.total_fpga_compute_time/FPGAQuantizedLinear.call_count:.6f} seconds")
# print(f"Adjusted speedup (FPGA compute only): {cpu_time / FPGAQuantizedLinear.total_fpga_compute_time:.2f}x")

# Compute differences
diff = logits_cpu - logits_fpga
max_diff = diff.abs().max().item()
mean_diff = diff.abs().mean().item()

# Compute softmax probabilities
probs_cpu = torch.softmax(logits_cpu, dim=1)
probs_fpga = torch.softmax(logits_fpga, dim=1)

# Get predicted class and confidence
predicted_class_cpu = torch.argmax(probs_cpu, dim=1).item()
confidence_cpu = probs_cpu[0, predicted_class_cpu].item() * 100

predicted_class_fpga = torch.argmax(probs_fpga, dim=1).item()
confidence_fpga = probs_fpga[0, predicted_class_fpga].item() * 100

# Print results
# print(f"Max Logits Difference: {max_diff:.6f}")
# print(f"Mean Logits Difference: {mean_diff:.6f}")

display_model_confidence(logits_cpu, device_name="CPU")
display_model_confidence(logits_fpga, device_name="FPGA")

# Calculate and print speedup
print(f"Total FPGA-Offloaded Inference Time: {fpga_time:.6f} seconds")
print(f"Overall Speedup (CPU time / FPGA time): {cpu_time / fpga_time:.2f}x")
