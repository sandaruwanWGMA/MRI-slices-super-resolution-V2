import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from models.volumetric_resnet.modified_3d_resnet_generator import Modified3DResNet


def save_model_layers_to_csv(model, input_tensor, file_path):
    """
    Save the model's layer details (type, parameters, input shape, output shape) to a CSV file.

    Parameters:
    model (torch.nn.Module): The model whose layers are to be saved.
    input_tensor (torch.Tensor): The input tensor to pass through the model for shape extraction.
    file_path (str): The file path to save the CSV file.
    """
    # Create a list to hold the table data
    table_data = []

    # Forward hook to capture layer input and output shapes
    def hook_fn(module, input, output):
        layer_name = module.__class__.__name__
        input_shape = list(input[0].shape)  # Input shape
        output_shape = list(output.shape)  # Output shape
        num_params = sum(p.numel() for p in module.parameters())  # Number of parameters
        table_data.append([layer_name, num_params, input_shape, output_shape])

    # Register forward hooks to capture input and output shapes of each layer, excluding the main model
    hooks = []
    for name, layer in model.named_modules():
        # Avoid registering a hook on the parent model (i.e., `VideoResNet`)
        if layer != model:
            hook = layer.register_forward_hook(hook_fn)
            hooks.append(hook)

    # Feed the dummy input to the model to trigger the hooks and collect the shapes
    with torch.no_grad():
        model(input_tensor)

    # Remove hooks after forward pass
    for hook in hooks:
        hook.remove()

    # Convert the list to a pandas DataFrame for a clean table-like structure
    df = pd.DataFrame(
        table_data,
        columns=["Layer Type", "Number of Parameters", "Input Shape", "Output Shape"],
    )

    # Save the DataFrame to a CSV file
    df.to_csv(file_path, index=False)
    print(f"Model layers saved to {file_path} successfully.")


# Example usage
if __name__ == "__main__":
    from torchvision.models.video import r3d_18, R3D_18_Weights

    # Load a pre-trained 3D ResNet model
    # model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    model = Modified3DResNet()

    # Create a dummy input tensor for testing [N, C, D, H, W]
    dummy_input = torch.randn(1, 1, 30, 256, 256)

    # Call the function to save the model layers info to a CSV file
    save_model_layers_to_csv(
        model, dummy_input, "models/modeified_3d_resnet_layers.csv"
    )
