# Dog Breed Classifier using Hydra, TorchScript, and Gradio

## Overview
This repository contains a PyTorch-based deep learning model for classifying dog breeds. The project demonstrates an end-to-end ML workflow, from model training to deployment, using Github Actions, Hydra, TorchScript, and Gradio.

## 🚀 Key Features
- Deep learning model built with PyTorch for dog breed classification
- Configuration management using Hydra
- Model optimization using TorchScript's trace method
- Automated workflow with GitHub Actions
- Interactive web interface using Gradio
- Deployed on Hugging Face Spaces

## 🛠️ Technical Stack
- PyTorch
- Hydra
- TorchScript
- GitHub Actions
- Gradio
- Hugging Face Spaces

## 🔄 Workflow
1. **Model Training**: Train the dog breed classifier using PyTorch
2. **Model Optimization**: Convert the trained model to TorchScript using trace method
3. **Automated Pipeline**: GitHub Actions workflow handles:
   - Model training
   - Model optimization
   - Deployment to Hugging Face
4. **Deployment**: Serve the model through a Gradio interface on Hugging Face Spaces

## 🔍 Tracing vs. Scripting
This project uses TorchScript's trace method for model optimization. Here's why:

### Tracing Benefits
- Maintains better code quality and doesn't limit Python syntax features
- Generates simpler and potentially faster computation graphs
- Has clear limitations with known solutions
- Impact is limited to only the outer-most module's input/output format

### Implementation Details
1. **Model Tracing**: The model is traced using `torch.jit.trace()` with sample inputs
2. **Generalization**: 
   - Careful attention to `TracerWarning` messages
   - Unit tests to verify traced model outputs match original model
   - Use of symbolic shapes for dynamic tensor operations
3. **Mixed Approach**: Where necessary, we use `@torch.jit.script_if_tracing` for sections requiring dynamic control flow

### Why Tracing Over Scripting?
Tracing was chosen because it:
- Preserves code quality and maintainability
- Has minimal impact on the codebase
- Provides better performance through simpler computation graphs
- Allows easier debugging and troubleshooting

## Code Details

**Saving model using tracing method:**

```python
# Create example input
example_input = torch.randn(1, 3, 160, 160)  

# Trace the model
log.info("Tracing model...")
traced_model = model.to_torchscript(method="trace", example_inputs=example_input)

# Create output directory if it doesn't exist
output_dir = Path("traced_models")
output_dir.mkdir(exist_ok=True)

# Save the traced model
output_path = output_dir / "model_tracing.pt"
torch.jit.save(traced_model, output_path)
log.info(f"Traced model saved to: {output_path}")
```

**Loading model using tracing method:**

```python
self.model = torch.jit.load(model_path)
self.model = self.model.to(self.device)
```

## References
- [Tracing vs. Scripting](https://ppwwyyxx.com/blog/2022/TorchScript-Tracing-vs-Scripting/)
- [Gradio](https://www.gradio.app/)
- [Hugging Face Spaces](https://huggingface.co/spaces)