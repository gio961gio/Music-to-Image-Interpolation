#  Music to Image Interpolation
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gio961gio/Music-to-Image-Interpolation/blob/main/Music_to_Image_Interpolation_.ipynb) <--- Link to project in Google Colab



---

Generative AI pipeline that produces image interpolations from an audio track, leveraging Stable Diffusion technology.

---
## Examples
https://github.com/gio961gio/Music-to-Image-Interpolation/assets/163283326/b822079f-1638-4ac5-8d64-ad25272b6e1b

Steve Reich -  Music for Pieces of Wood  (30 second extract) (fps=7, num_inference_steps=20)

---
https://github.com/gio961gio/Music-to-Image-Interpolation/assets/163283326/b248b6f3-53c8-44ab-a7a3-1b40d463d89b

Karlheinz Stockhausen -  Helicopter String Quartet (25 seconds) (fps=5, num_inference_steps=30)


---
https://github.com/gio961gio/Music-to-Image-Interpolation/assets/163283326/1ba2023e-22c9-479e-aef7-53f4b94a4335

Jean-Claude Risset - SUD (30 second extract) (fps=7, num_inference_steps=20)

---



https://github.com/gio961gio/Music-to-Image-Interpolation/assets/163283326/7db11dd8-a384-42f7-bc1c-6b6acff63c99

Antonio Vivaldi - Winter (15 seconds extract) (fps=7, num_inference_steps=20)


---
## Pipeline
![Pipeline](https://github.com/gio961gio/Music-to-Image-Interpolation/assets/163283326/a2b7fc86-e986-4c0e-bc19-12801649902a)

---
# Informations
The core of the system is the Stable Diffusion 'img2img' by Hugging Face. Image embeddings are created using the Image Bind model by Meta, which employs multimodality and transforms audio data into image embeddings. 

The interpolation part is adapted from the publicly available code by nateraw (https://github.com/nateraw/stable-diffusion-videos.git), and the detextifier is also adapted from the publicly available code by iuliaturc (https://github.com/iuliaturc/detextify.git).
The Stable Diffusion and ImageBind models are incorporated into the public code provided by Zeqiang-Lai (https://github.com/Zeqiang-Lai/Anything2Image.git).

