import argparse
import os
import torch
from PIL import Image as PILImage
import torchvision.transforms as transforms
import networks
from utils.transforms import BGR2RGB_transform, transform_parsing



def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Self Correction for Human Parsing")

    # Network Structure
    parser.add_argument("--arch", type=str, default='resnet101')
    # Data Preference
    parser.add_argument("--data-dir", type=str, default='./data/LIP')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-size", type=str, default='473,473')
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--random-mirror", action="store_true")
    parser.add_argument("--random-scale", action="store_true")
    # Evaluation Preference
    parser.add_argument("--log-dir", type=str, default='./log')
    parser.add_argument("--model-restore", type=str, default='./log/checkpoint.pth.tar')
    parser.add_argument("--gpu", type=str, default='0', help="choose gpu device.")
    parser.add_argument("--save-results", action="store_true", help="whether to save the results.")
    parser.add_argument("--flip", action="store_true", help="random flip during the test.")
    parser.add_argument("--multi-scales", type=str, default='1', help="multiple scales during the test")
    return parser.parse_args()
def load_model(model_path):
    args = get_arguments()
    # Create an instance of the model
    model = networks.init_model(args.arch, num_classes=args.num_classes, pretrained=None)
    # Load the pre-trained weights
    state_dict = torch.load(model_path)['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    # Move the model to GPU
    model.cuda()
    model.eval()
    return model

def inference(input_image_path, model, output_dir):
    # Load and preprocess the input image
    input_image = PILImage.open(input_image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
        BGR2RGB_transform(),
        transforms.Normalize(mean=model.mean, std=model.std),
    ])
    input_tensor = transform(input_image).unsqueeze(0).cuda()

    # Perform inference
    parsing, _ = multi_scale_testing(model, input_tensor, flip=False, multi_scales=[1])

    # Save the predicted mask
    parsing_result = transform_parsing(parsing, [0, 0], 1.0, input_image.width, input_image.height, [473, 473])
    output_image_path = os.path.join(output_dir, "predicted_mask.png")
    output_im = PILImage.fromarray(parsing_result.astype('uint8'))
    output_im.save(output_image_path)

    print(f"Predicted mask saved at: {output_image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for semantic segmentation")
    parser.add_argument("--image-path", type=str, help="Path to the input image", required=True)
    parser.add_argument("--model-path", type=str, help="Path to the trained model", required=True)
    parser.add_argument("--output-dir", type=str, help="Directory to save the predicted mask", default="./output")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model = load_model(args.model_path)
    inference(args.image_path, model, args.output_dir)
