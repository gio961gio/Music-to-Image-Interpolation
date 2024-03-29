import torch
import os
import numpy as np
import math
from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict
import librosa
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from torchvision.io import write_video
import json
import time
from torch import Tensor
import glob
from tqdm import tqdm 
from detext import Detextifier




#### GENERATE INPUTS #########à
def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """Helper function to spherically interpolate two arrays v1 v2"""

    inputs_are_torch = isinstance(v0, torch.Tensor)
    if inputs_are_torch:
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2


def init_noise(seed, noise_shape, dtype):
    """Helper to initialize noise"""
    # randn does not exist on mps, so we create noise on CPU here and move it to the device after initialization
    noise = torch.randn(
        noise_shape,
        device='cuda',
        generator=torch.Generator(device='cuda').manual_seed(seed),
        dtype=dtype,
    )

    return noise


def generate_inputs(prompt_a, prompt_b, seed_a, seed_b, noise_shape, T, batch_size):
    embeds_a = prompt_a
    embeds_b = prompt_b
    latents_dtype = embeds_a.dtype
    latents_a = init_noise(seed_a, noise_shape, latents_dtype)
    latents_b = init_noise(seed_b, noise_shape, latents_dtype)


    batch_idx = 0
    embeds_batch, noise_batch = None, None

    for i, t in enumerate(T):
        embeds = torch.lerp(embeds_a, embeds_b, t)
        noise = slerp(float(t), latents_a, latents_b)

        embeds_batch = embeds if embeds_batch is None else torch.cat([embeds_batch, embeds])
        noise_batch = noise if noise_batch is None else torch.cat([noise_batch, noise])
        embeds_batch = embeds_batch.to(torch.float16)
        noise_batch = noise_batch.to(torch.float16)


        batch_is_ready = embeds_batch.shape[0] == batch_size or i + 1 == T.shape[0]
        if not batch_is_ready:
            continue
        yield batch_idx, embeds_batch, noise_batch

        batch_idx += 1
        del embeds_batch, noise_batch
        torch.cuda.empty_cache()
        embeds_batch, noise_batch = None, None




##### MAKE CLIP FRAMES #######
def make_clip_frames(
    prompt_a: str,
    prompt_b: str,
    seed_a: int,
    seed_b: int,
    num_interpolation_steps: int = 5,
    save_path: Union[str, Path] = "outputs/",
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    eta: float = 0.0,
    height: Optional[int] = 512,
    width: Optional[int] = 512,
    batch_size: int = 1,
    image_file_ext: str = ".png",
    T: np.ndarray = None,
    skip: int = 0,
    negative_prompt: str = None,
    step: Optional[Tuple[int, int]] = None,
    pipe=None
):


    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    T = T if T is not None else np.linspace(0.0, 1.0, num_interpolation_steps)
    if T.shape[0] != num_interpolation_steps:
        raise ValueError(f"Unexpected T shape, got {T.shape}, expected dim 0 to be {num_interpolation_steps}")

    batch_generator = generate_inputs(
        prompt_a=prompt_a,
        prompt_b=prompt_b,
        seed_a=seed_a,
        seed_b=seed_b,
        noise_shape = (1, 4, height // 8, width // 8),
        T = T[skip:],
        batch_size=batch_size,
    )
    num_batches = math.ceil(num_interpolation_steps / batch_size)

    log_prefix = '' if step is None else f'[{step[0]}/{step[1]}] '

    frame_index = skip
    for batch_idx, embeds_batch, noise_batch in batch_generator:
        if batch_size == 1:
            msg = f"Generating frame {frame_index}"
        else:
            msg = f"Generating frames {frame_index}-{frame_index + embeds_batch.shape[0] - 1}"
        # logger.info(f'{log_prefix}[{batch_idx}/{num_batches}] {msg}')



        outputs = pipe(prompt=['']*batch_size,
                       image_embeds=embeds_batch,
                       negative_prompt= ['']*batch_size,
                       latents=noise_batch,
                       height=height,
                       width=width,
                       guidance_scale=guidance_scale,
                       eta=eta,
                       num_inference_steps=num_inference_steps,
                       output_type="pil" ,).images


        for image in outputs:
            frame_filepath = save_path / (f"frame%06d{image_file_ext}" % frame_index)
            image.save(frame_filepath)
            frame_index += 1





#### MAKE VIDEO ###
def get_timesteps_arr(audio_filepath, offset, duration, fps=30, margin=1.0, smooth=0.0):
    y, sr = librosa.load(audio_filepath, offset=offset, duration=duration)

    """
    Calcola e restituisce un array di passaggi temporali normalizzati
    basati sull'analisi dello spettrogramma audio percussivo.

    Argomenti:
    audio_filepath (str): Percorso del file audio.
    offset (float): Offset temporale in secondi dall'inizio del file audio.
    duration (float): Durata in secondi dell'intervallo audio da considerare.
    fps (int, optional): Frame rate desiderato per l'array temporale. Default è 30.
    margin (float, optional): Margine per la decomposizione HPSS. Default è 1.0.
    smooth (float, optional): Fattore di smoothing per l'array temporale. Default è 0.0.

    Restituisce:
    np.ndarray: Array di passaggi temporali normalizzati.
    """

    # librosa.stft hardcoded defaults...
    # n_fft defaults to 2048
    # hop length is win_length // 4
    # win_length defaults to n_fft
    D = librosa.stft(y, n_fft=2048, hop_length=2048 // 4, win_length=2048)

    # Extract percussive elements
    D_harmonic, D_percussive = librosa.decompose.hpss(D, margin=margin)
    y_percussive = librosa.istft(D_percussive, length=len(y))

    # Get normalized melspectrogram
    spec_raw = librosa.feature.melspectrogram(y=y_percussive, sr=sr)
    spec_max = np.amax(spec_raw, axis=0)
    spec_norm = (spec_max - np.min(spec_max)) / np.ptp(spec_max)

    # Resize cumsum of spec norm to our desired number of interpolation frames
    x_norm = np.linspace(0, spec_norm.shape[-1], spec_norm.shape[-1])
    y_norm = np.cumsum(spec_norm)
    y_norm /= y_norm[-1]
    x_resize = np.linspace(0, y_norm.shape[-1], int(duration * fps))

    T = np.interp(x_resize, x_norm, y_norm)

    # Apply smoothing
    return T * (1 - smooth) + np.linspace(0.0, 1.0, T.shape[0]) * smooth


def make_video_pyav(
    frames_or_frame_dir: Union[str, Path, torch.Tensor],
    audio_filepath: Union[str, Path] = None,
    fps: int = 30,
    audio_offset: int = 0,
    audio_duration: int = 2,
    sr: int = 22050,
    output_filepath: Union[str, Path] = "output.mp4",
    glob_pattern: str = "*.png",
) -> str:
    """
    Create a video from frames or a directory of images, with optional audio.

    Args:
        frames_or_frame_dir (Union[str, Path, torch.Tensor]): Either a directory of images,
            or a tensor of shape (T, C, H, W) in range [0, 255].
        audio_filepath (Union[str, Path], optional): Path to audio file (e.g., .wav) to be added to the video.
        fps (int, optional): Frames per second for the output video. Default is 30.
        audio_offset (int, optional): Offset in seconds for the audio file. Default is 0.
        audio_duration (int, optional): Duration in seconds for the audio file. Default is 2.
        sr (int, optional): Sample rate for the audio file. Default is 22050.
        output_filepath (Union[str, Path], optional): Path to save the output video file. Default is "output.mp4".
        glob_pattern (str, optional): Glob pattern for image files when `frames_or_frame_dir` is a directory.
            Default is "*.png".

    Returns:
        str: Path to the output video file.
    """
    # Convert output_filepath to string as torchvision write_video doesn't support pathlib paths
    output_filepath = str(output_filepath)

    if isinstance(frames_or_frame_dir, (str, Path)):
        frames = None
        for img in sorted(Path(frames_or_frame_dir).glob(glob_pattern)):
            frame = pil_to_tensor(Image.open(img)).unsqueeze(0)
            frames = frame if frames is None else torch.cat([frames, frame])
    else:
        frames = frames_or_frame_dir

    # TCHW -> THWC
    frames = frames.permute(0, 2, 3, 1)

    if audio_filepath:
        # Read audio, convert to tensor
        audio, sr = librosa.load(
            audio_filepath,
            sr=sr,
            mono=True,
            offset=audio_offset,
            duration=audio_duration,
        )
        audio_tensor = torch.tensor(audio).unsqueeze(0)

        write_video(
            output_filepath,
            frames,
            fps=fps,
            audio_array=audio_tensor,
            audio_fps=sr,
            audio_codec="aac",
            options={"crf": "10", "pix_fmt": "yuv420p"},
        )
    else:
      write_video(
            output_filepath,
            frames,
            fps=fps,
            options={"crf": "10", "pix_fmt": "yuv420p"},
        )

    return output_filepath





### WALK ###
def walk (prompts: Optional[List[Tensor]] = None,
        seeds: Optional[List[int]] = None,
        num_interpolation_steps: Optional[Union[int, List[int]]] = 5,  # int or list of int
        output_dir: Optional[str] = "./dreams",
        name: Optional[str] = None,
        image_file_ext: Optional[str] = ".png",
        fps: Optional[int] = 30,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        batch_size: Optional[int] = 1,
        audio_filepath: str = None,
        audio_start_sec: Optional[Union[int, float]] = None,
        margin: Optional[float] = 1.0,
        smooth: Optional[float] = 0.0,
        negative_prompt: Optional[str] = None,
        make_video: Optional[bool] = True,
        detext: Optional[bool] = False,
        text_detector=None,
        inpainter=None,
        pipe = None
        ):

  output_path = Path(output_dir)
  name = name or time.strftime("%Y%m%d-%H%M%S")
  save_path_root = output_path / name
  save_path_root.mkdir(parents=True, exist_ok=True)
  # Where the final video of all the clips combined will be saved
  output_filepath = save_path_root / f"{name}.mp4"
  if isinstance(num_interpolation_steps, int):
            num_interpolation_steps = [num_interpolation_steps] * (len(prompts) - 1)

  # Use tqdm to create a progress bar
  progress_bar = tqdm(enumerate(zip(prompts, prompts[1:], seeds, seeds[1:], num_interpolation_steps)),
                      total=len(prompts) - 1, desc="Generating clips")

  for i, (prompt_a, prompt_b, seed_a, seed_b, num_step) in progress_bar:
    # {name}_000000 / {name}_000001 / ...
    save_path = save_path_root / f"{name}_{i:06d}"

    # Where the individual clips will be saved
    step_output_filepath = save_path / f"{name}_{i:06d}.mp4"


    skip = 0

    #audio settings
    audio_offset = audio_start_sec + sum(num_interpolation_steps[:i]) / fps
    audio_duration = num_step / fps

    make_clip_frames(
        prompt_a=prompt_a,
        prompt_b=prompt_b,
        seed_a=seed_a,
        seed_b=seed_b,
        num_interpolation_steps=num_step,
        save_path=save_path,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        eta=eta,
        height=height,
        width=width,
        batch_size=batch_size,
        T=get_timesteps_arr(
            audio_filepath,
            offset=audio_offset,
            duration=audio_duration,
            fps=fps,
            margin=margin,
            smooth=smooth,)
        if audio_filepath else None,
        skip=skip,
        negative_prompt=negative_prompt,
        step=(i, len(prompts) - 1),
        pipe=pipe)


    if make_video:
      make_video_pyav(
          save_path,
          audio_filepath=audio_filepath,
          fps=fps,
          output_filepath=step_output_filepath,
          glob_pattern=f"*{image_file_ext}",
          audio_offset=audio_offset,
          audio_duration=audio_duration,
          sr=44100
      )



    percorso_immagini = os.path.join(save_path, "*.png")

    if detext:
      detextifier = Detextifier(text_detector, inpainter)
      for img_file in glob.glob(percorso_immagini):
        detextifier.detextify(img_file, img_file.replace(".png", ".png"))

  if make_video:
    return make_video_pyav(
        save_path_root,
        audio_filepath=audio_filepath,
        fps=fps,
        audio_offset=audio_start_sec,
        audio_duration=sum(num_interpolation_steps) / fps,
        output_filepath=output_filepath,
        glob_pattern=f"**/*{image_file_ext}",
        sr=44100,
    )
