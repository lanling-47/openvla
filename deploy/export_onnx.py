"""
Export OpenVLA LLM backbone to ONNX for TensorRT deployment.

Exports the Llama-2 7B LLM backbone (the action generation bottleneck)
to ONNX format for subsequent TensorRT INT8 compilation on Jetson Orin.

The vision encoder is exported separately (smaller, less critical).
"""

import torch
import argparse
import os


def export_llm_onnx(model_path: str, output_path: str, opset: int = 17):
    """
    Export OpenVLA's LLM backbone to ONNX.

    The LLM backbone (Llama-2 7B) is the inference bottleneck.
    Exporting to ONNX enables TensorRT optimization with:
      - INT8 quantization (2x memory reduction)
      - Layer fusion (reduce kernel launch overhead)
      - Optimized GEMM selection for Jetson Orin's GPU (sm_87)
    """
    from transformers import AutoModelForVision2Seq
    
    print("Loading OpenVLA from %s..." % model_path)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path, torch_dtype=torch.float16,
        low_cpu_mem_usage=True, trust_remote_code=True,
    ).eval()
    
    llm = model.language_model
    hidden_size = llm.config.hidden_size  # 4096 for Llama-2 7B
    
    # Create dummy input (single decode step)
    # In VLA: input_embeds = [visual_tokens(256) + text_tokens(~19)] = 275 tokens
    batch_size = 1
    seq_len = 275  # Typical VLA prefill length
    dummy_input = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16)
    
    print("Exporting LLM backbone to ONNX...")
    print("  Input shape: [%d, %d, %d]" % (batch_size, seq_len, hidden_size))
    print("  Output: %s" % output_path)
    
    # Export with dynamic axes for variable sequence length
    torch.onnx.export(
        llm,
        (dummy_input,),  # inputs_embeds
        output_path,
        input_names=["input_embeds"],
        output_names=["logits"],
        dynamic_axes={
            "input_embeds": {0: "batch", 1: "seq_len"},
            "logits": {0: "batch", 1: "seq_len"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    
    file_size = os.path.getsize(output_path) / 1e9
    print("Exported: %.2f GB" % file_size)
    print("\nNext step: transfer to Jetson and run TensorRT conversion:")
    print("  $ python deploy/tensorrt_convert.py")


def export_vision_onnx(model_path: str, output_path: str, opset: int = 17):
    """Export vision encoder (DINOv2 + SigLIP + projector) to ONNX."""
    from transformers import AutoModelForVision2Seq
    
    print("Loading vision encoder...")
    model = AutoModelForVision2Seq.from_pretrained(
        model_path, torch_dtype=torch.float16,
        low_cpu_mem_usage=True, trust_remote_code=True,
    ).eval()
    
    # Vision backbone expects [batch, channels=6, H=224, W=224]
    # (6 channels = DINOv2(3) + SigLIP(3) concatenated)
    dummy_pixels = torch.randn(1, 6, 224, 224, dtype=torch.float16)
    
    # Export vision backbone + projector as one module
    class VisionModule(torch.nn.Module):
        def __init__(self, backbone, projector):
            super().__init__()
            self.backbone = backbone
            self.projector = projector
        
        def forward(self, pixel_values):
            features = self.backbone(pixel_values)
            return self.projector(features)
    
    vision_module = VisionModule(model.vision_backbone, model.projector).eval()
    
    print("Exporting vision encoder to ONNX...")
    torch.onnx.export(
        vision_module,
        (dummy_pixels,),
        output_path,
        input_names=["pixel_values"],
        output_names=["visual_embeddings"],
        opset_version=opset,
        do_constant_folding=True,
    )
    
    file_size = os.path.getsize(output_path) / 1e9
    print("Exported: %.2f GB" % file_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export OpenVLA to ONNX")
    parser.add_argument("--model", default="/root/models/openvla-7b")
    parser.add_argument("--output-llm", default="openvla_llm.onnx")
    parser.add_argument("--output-vision", default="openvla_vision.onnx")
    parser.add_argument("--llm-only", action="store_true")
    args = parser.parse_args()

    export_llm_onnx(args.model, args.output_llm)
    if not args.llm_only:
        export_vision_onnx(args.model, args.output_vision)
