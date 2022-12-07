import gradio
import numpy as np # this should come first to mitigate mlk-service bug
from src.models.utils import get_image_arr, load_model
from src.data import TAIMGANTokenizer
from torchvision import transforms
from src.config import config_dict
from pathlib import Path
from PIL import Image
import gradio as gr
import logging
import torch
from src.models.modules import (
    VGGEncoder,
    InceptionEncoder,
    TextEncoder,
    Generator
)

##########
# PARAMS #
##########

IMG_CHANS = 3  # RGB channels for image
IMG_HW = 256  # height and width of images
HIDDEN_DIM = 128  # hidden dimensions of lstm cell in one direction
C = 2 * HIDDEN_DIM  # length of embeddings

Ng = config_dict["Ng"]
cond_dim = config_dict["condition_dim"]
z_dim = config_dict["noise_dim"]


###############
# LOAD MODELS #
###############

models = {
    "COCO": {
        "dir": "weights/coco"
    },
    "Bird": {
        "dir": "weights/bird"
    },
    "UTKFace": {
        "dir": "weights/utkface"
    }
}

for model_name in models:
    # create tokenizer
    models[model_name]["tokenizer"] = TAIMGANTokenizer(captions_path=f"{models[model_name]['dir']}/captions.pickle")
    vocab_size = len(models[model_name]["tokenizer"].word_to_ix)
    # instantiate models
    models[model_name]["generator"] = Generator(Ng=Ng, D=C, conditioning_dim=cond_dim, noise_dim=z_dim).eval()
    models[model_name]["lstm"] = TextEncoder(vocab_size=vocab_size, emb_dim=C, hidden_dim=HIDDEN_DIM).eval()
    models[model_name]["vgg"] = VGGEncoder().eval()
    models[model_name]["inception"] = InceptionEncoder(D=C).eval()
    # load models
    load_model(
        generator=models[model_name]["generator"],
        discriminator=None,
        image_encoder=models[model_name]["inception"],
        text_encoder=models[model_name]["lstm"],
        output_dir=Path(models[model_name]["dir"]),
        device=torch.device("cpu")
    )


def change_image_with_text(image: Image, text: str, model_name: str) -> Image:
    """
    Create an image modified by text from the original image
    and save it with _modified postfix

    :param gr.Image image: Path to the image
    :param str text: Desired caption
    """
    global models
    tokenizer = models[model_name]["tokenizer"]
    G = models[model_name]["generator"]
    lstm = models[model_name]["lstm"]
    inception = models[model_name]["inception"]
    vgg = models[model_name]["vgg"]
    # generate some noise
    noise = torch.rand(z_dim).unsqueeze(0)
    # transform input text and get masks with embeddings
    tokens = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    mask = (tokens == tokenizer.pad_token_id)
    word_embs, sent_embs = lstm(tokens)
    # open the image and transform it to the tensor
    image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMG_HW, IMG_HW)),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ])(image).unsqueeze(0)
    # obtain visual features of the image
    vgg_features = vgg(image)
    local_features, global_features = inception(image)
    # generate new image from the old one
    fake_image, _, _ = G(noise, sent_embs, word_embs, global_features,
                         local_features, vgg_features, mask)
    # denormalize the image
    fake_image = Image.fromarray(get_image_arr(fake_image)[0])
    # return image in gradio format
    return fake_image


##########
# GRADIO #
##########
gradio.close_all()
demo = gr.Interface(
    fn=change_image_with_text,
    inputs=[gr.Image(type="pil"), "text", gr.inputs.Dropdown(list(models.keys()))],
    outputs=gr.Image(type="pil"),
    examples=[
        ["src/data/stubs/bird.jpg", "black bird with blue wings", "Bird"],
        ["src/data/stubs/lady.jpg", "lady with blue eyes", "UTKFace"],
        ["src/data/stubs/bird.jpg", "white bird with black wings", "Bird"]
    ]
)
print("Please visit http://0.0.0.0:7861")
demo.launch(
    server_name="0.0.0.0",
    server_port=7861,
    show_error=True,
    debug=True
)
