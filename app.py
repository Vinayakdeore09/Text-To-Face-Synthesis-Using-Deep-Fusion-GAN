# app.py
import streamlit as st
import torch
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


# Streamlit UI
st.set_page_config(page_title="Text-to-Face Generator", layout="wide")

# Import your model classes and text encoder
from models import NetG, NetD  # From your original code
from FGTD.scripts.text_encoder.sentence_encoder import SentenceEncoder  # From your original code

# Configuration
class Config:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    noise_size = 100
    image_size = 128
    ndf = 64

cfg = Config()

@st.cache_resource
def load_models():
    # Initialize models
    generator = NetG(64, 100).to(cfg.device)
    discriminator = NetD(cfg.ndf).to(cfg.device)
    
    # Load pretrained weights
    generator.load_state_dict(torch.load('generatorVNPS.pth', map_location=cfg.device))
    discriminator.load_state_dict(torch.load('discriminatorVNPS.pth', map_location=cfg.device))
    
    generator.eval()
    discriminator.eval()
    return generator, discriminator

# Load models and encoder
generator, discriminator = load_models()
sentence_encoder = SentenceEncoder(cfg.device)



# Main interface
st.title("Criminal Portrait Synthesis from Text-to-Image using Deep Learning")
st.markdown("Generate realistic faces from text descriptions using GAN")

# Input section
col1, col2 = st.columns([3, 1])
with col1:
    text_input = st.text_area(
        "Describe the face you want to generate:",
        "A young woman with blonde hair and blue eyes",
        height=100
    )

with col2:
    seed = st.number_input("Random seed", value=42, min_value=0)
    temp = st.slider("Creativeness", 0.1, 1.0, 0.7)
    show_confidence = st.checkbox("Show confidence score")

# Generation button
if st.button("Generate Face"):
    try:
        with st.spinner("Generating face..."):
            # Process text
            text_emb = sentence_encoder.convert_text_to_embeddings([text_input])
            
            # Generate noise
            torch.manual_seed(seed)
            noise = torch.randn(1, cfg.noise_size).to(cfg.device) * temp
            
            # Generate image
            with torch.no_grad():
                generated = generator(noise, text_emb)
                
                # Get discriminator confidence
                if show_confidence:
                    features = discriminator(generated)
                    confidence = discriminator.COND_DNET(features, text_emb)
                    confidence = torch.sigmoid(confidence).item()

            # Convert tensor to PIL Image
            img = generated.cpu().squeeze().permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min())  # Normalize to 0-1
            img = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img).resize((256, 256))
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.image(pil_img, caption="Generated Face", use_container_width=True)
            
            if show_confidence:
                with col2:
                    st.metric("Realness Confidence", f"{confidence:.2%}")
                    st.progress(confidence)
                    
    except Exception as e:
        st.error(f"Generation failed: {str(e)}")

# Sidebar information
st.sidebar.header("About")
st.sidebar.markdown("""
This AI-powered app uses a Deep Convolutional GAN to generate human faces from text descriptions. 
The model was trained on the CelebA dataset with 200K+ celebrity images.
""")

st.sidebar.markdown("### Example Prompts")
st.sidebar.write("- Young Asian male with black hair and glasses")
st.sidebar.write("- Old Caucasian woman with gray curly hair")
st.sidebar.write("- Middle-aged man with beard and mustache")

st.sidebar.markdown("### Model Info")
st.sidebar.code(f"Device: {cfg.device}")
st.sidebar.code(f"Generator Parameters: {sum(p.numel() for p in generator.parameters()):,}")
st.sidebar.code(f"Discriminator Parameters: {sum(p.numel() for p in discriminator.parameters()):,}")