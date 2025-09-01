"""
Streamlit Image Enhancer
File: streamlit_image_enhancer.py

Features:
- Upload single or multiple images
- Denoise (OpenCV fastNlMeansDenoisingColored)
- Sharpen / Unsharp Mask
- Contrast, Brightness, Color adjustments (PIL.ImageEnhance)
- CLAHE (histogram equalization for better local contrast)
- Upscale (bicubic / lanczos interpolation)
- One-click Auto-Enhance (combines a few improvements)
- Side-by-side preview and download enhanced image

Dependencies:
pip install streamlit pillow opencv-python-headless numpy

Run:
streamlit run streamlit_image_enhancer.py

"""

import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2
import io

st.set_page_config(page_title="🖼️ Image Enhancer", page_icon="🛠️", layout="centered")
st.title("🖼️ Image Enhancer — Clean, Sharpen & Upscale")
st.write("Upload images and apply denoising, contrast, sharpening, CLAHE, and upscaling. Download results when you're happy.")

# Sidebar controls
st.sidebar.header("Enhancement Controls")
use_auto = st.sidebar.checkbox("One-click Auto-Enhance", value=False)

# Individual control groups (only used when not using auto)
denoise_strength = st.sidebar.slider("Denoise strength (0 = off)", 0.0, 1.0, 0.2, step=0.05)
sharpen_amount = st.sidebar.slider("Sharpen amount (0 = off)", 0.0, 3.0, 1.0, step=0.1)
contrast = st.sidebar.slider("Contrast", 0.5, 2.0, 1.1, step=0.05)
brightness = st.sidebar.slider("Brightness", 0.5, 2.0, 1.0, step=0.05)
color = st.sidebar.slider("Color / Saturation", 0.0, 2.0, 1.0, step=0.05)
use_clahe = st.sidebar.checkbox("Apply CLAHE (local contrast)", value=False)
upscale_factor = st.sidebar.selectbox("Upscale (nearest quality) ", [1, 2, 3, 4], index=0)
upscale_method = st.sidebar.selectbox("Upscale method", ["bicubic", "lanczos"], index=0)

# Advanced options
st.sidebar.markdown("---")
apply_unsharp_mask = st.sidebar.checkbox("Use Unsharp Mask (alternative sharpening)", value=True)

# File uploader
uploaded_files = st.file_uploader("Upload image(s)", type=["png", "jpg", "jpeg", "webp", "tiff"], accept_multiple_files=True)

# Utility functions

def pil_to_cv2(img_pil: Image.Image) -> np.ndarray:
    arr = np.array(img_pil.convert('RGB'))
    # Convert RGB to BGR
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv2_to_pil(img_cv2: np.ndarray) -> Image.Image:
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def denoise_image_cv2(img_cv2: np.ndarray, strength: float) -> np.ndarray:
    """Denoise using OpenCV fastNlMeansDenoisingColored.
    strength: 0.0 (off) to 1.0 (strong)
    We'll map strength to h parameters.
    """
    if strength <= 0:
        return img_cv2
    h = max(3, int(10 * strength))
    hColor = h
    templateWindowSize = 7
    searchWindowSize = 21
    denoised = cv2.fastNlMeansDenoisingColored(img_cv2, None, h, hColor, templateWindowSize, searchWindowSize)
    return denoised


def apply_clahe_cv2(img_cv2: np.ndarray) -> np.ndarray:
    """Apply CLAHE on the L channel in LAB color space for local contrast enhancement."""
    lab = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def unsharp_mask_pil(img: Image.Image, amount: float = 1.0, radius: int = 2) -> Image.Image:
    """Unsharp mask implemented with PIL filters. amount 0=>no effect."""
    if amount <= 0:
        return img
    blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
    # PIL.ImageChops is another option but we can blend
    return Image.blend(img, blurred, alpha=-0.5 * amount + 0.5) if False else Image.composite(img, blurred, Image.new('L', img.size, int(255 * 0)))


def sharpen_pil(img: Image.Image, amount: float = 1.0) -> Image.Image:
    if amount <= 0:
        return img
    enhancer = ImageEnhance.Sharpness(img)
    return enhancer.enhance(1.0 + amount)


def enhance_pil(img: Image.Image, contrast_v: float, brightness_v: float, color_v: float) -> Image.Image:
    img_out = img
    if contrast_v != 1.0:
        img_out = ImageEnhance.Contrast(img_out).enhance(contrast_v)
    if brightness_v != 1.0:
        img_out = ImageEnhance.Brightness(img_out).enhance(brightness_v)
    if color_v != 1.0:
        img_out = ImageEnhance.Color(img_out).enhance(color_v)
    return img_out


def upscale_cv2(img_cv2: np.ndarray, factor: int, method: str = 'bicubic') -> np.ndarray:
    if factor <= 1:
        return img_cv2
    h, w = img_cv2.shape[:2]
    new_w, new_h = w * factor, h * factor
    interp = cv2.INTER_CUBIC if method == 'bicubic' else cv2.INTER_LANCZOS4
    return cv2.resize(img_cv2, (new_w, new_h), interpolation=interp)


def auto_enhance_pipeline(img_pil: Image.Image) -> Image.Image:
    """A simple automatic pipeline: mild denoise -> CLAHE -> contrast/color boost -> sharpen -> upscale (if set)"""
    img_cv2 = pil_to_cv2(img_pil)
    img_cv2 = denoise_image_cv2(img_cv2, strength=0.25)
    img_cv2 = apply_clahe_cv2(img_cv2)
    pil = cv2_to_pil(img_cv2)
    pil = enhance_pil(pil, contrast_v=1.15, brightness_v=1.02, color_v=1.1)
    pil = sharpen_pil(pil, amount=1.0)
    return pil

# Main processing

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.markdown(f"### {uploaded_file.name}")
        file_bytes = uploaded_file.read()
        img = Image.open(io.BytesIO(file_bytes)).convert('RGB')

        # Preview
        st.image(img, caption="Original", use_container_width=True)

        if use_auto:
            with st.spinner("Applying one-click auto-enhance..."):
                enhanced = auto_enhance_pipeline(img)
        else:
            # Start from original and apply selected transforms
            img_proc = img.copy()

            # Denoise (cv2)
            if denoise_strength > 0:
                img_cv = pil_to_cv2(img_proc)
                img_cv = denoise_image_cv2(img_cv, denoise_strength)
                img_proc = cv2_to_pil(img_cv)

            # CLAHE
            if use_clahe:
                img_cv = pil_to_cv2(img_proc)
                img_cv = apply_clahe_cv2(img_cv)
                img_proc = cv2_to_pil(img_cv)

            # Color / Contrast / Brightness
            if contrast != 1.0 or brightness != 1.0 or color != 1.0:
                img_proc = enhance_pil(img_proc, contrast, brightness, color)

            # Sharpening
            if apply_unsharp_mask:
                # Use PIL's sharpness enhancer
                img_proc = sharpen_pil(img_proc, amount=sharpen_amount)
            else:
                if sharpen_amount > 0:
                    img_proc = sharpen_pil(img_proc, amount=sharpen_amount)

            # Upscale
            if upscale_factor > 1:
                img_cv = pil_to_cv2(img_proc)
                img_cv = upscale_cv2(img_cv, upscale_factor, method=upscale_method)
                img_proc = cv2_to_pil(img_cv)

            enhanced = img_proc

        # Display comparison side-by-side
        st.subheader("Preview")
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original", use_container_width=True)
        with col2:
            st.image(enhanced, caption="Enhanced", use_container_width=True)

        # Download enhanced image
        buf = io.BytesIO()
        enhanced.save(buf, format='PNG')
        byte_im = buf.getvalue()

        st.download_button(label="📥 Download Enhanced PNG", data=byte_im, file_name=f"enhanced_{uploaded_file.name}.png", mime="image/png")

else:
    st.info("👆 Upload one or more images to enhance them")

# Footer
st.markdown("---")
st.caption("Tip: Try the One-click Auto-Enhance first, then tweak sliders for fine control.")
st.caption("Developed with ❤️ using Streamlit, Pillow, and OpenCV")
st.caption("Version 1.0")
    
