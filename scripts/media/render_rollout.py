"""Render recorded rollouts (scripts/media/record_rollout.py) with MuJoCo offscreen: solid robot = policy in
simulation, translucent blue ghost = reference motion. Writes one mp4 per clip and optionally a concatenated
highlight; GIF conversion is left to ffmpeg (see README).

  MUJOCO_GL=glfw uv run python scripts/media/render_rollout.py media/rollouts.npz media/out --fps 25 --seconds 6
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Literal

os.environ.setdefault("MUJOCO_GL", "glfw")
import mujoco
import numpy as np
import tyro
from PIL import Image, ImageDraw, ImageFont

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

GHOST_RGBA = "0.18 0.52 0.92 0.82"


def two_robot_xml(mjcf: Path, root_body: str, ghost_rgba: str = GHOST_RGBA) -> str:
    xml = mjcf.read_text()
    xml = re.sub(r'meshdir="([^"]*)"', lambda m: f'meshdir="{(mjcf.parent / m.group(1)).resolve()}"', xml, count=1)
    # strip anything that is not needed for drawing (actuators / sensors / keyframes keep working but are unused)
    body = xml[xml.index(f'<body name="{root_body}"') : xml.rindex("</worldbody>")]
    ghost = re.sub(r' name="([^"]+)"', r' name="ref_\1"', body)
    # keep only the visual mesh geoms of the ghost, visual-only and translucent
    ghost = re.sub(r"<geom\b[^>]*/>", lambda m: "" if "mesh=" not in m.group(0) else m.group(0), ghost)
    ghost = re.sub(r' (contype|conaffinity|group|material|rgba|class)="[^"]*"', "", ghost)
    ghost = ghost.replace("<geom ", f'<geom type="mesh" contype="0" conaffinity="0" group="1" rgba="{ghost_rgba}" ')
    ghost = re.sub(r"<site[^>]*/>", "", ghost)
    ghost = re.sub(r"<freejoint[^>]*/>", '<freejoint name="ref_root"/>', ghost, count=1)
    scene = """
    <light name="key" pos="-2 -3 5" dir="0.3 0.4 -1" directional="true" diffuse="0.48 0.48 0.48" specular="0.10 0.10 0.10" castshadow="true"/>
    <light name="fill" pos="3 1 4" dir="-0.4 -0.2 -1" directional="true" diffuse="0.18 0.20 0.24" specular="0.02 0.02 0.02" castshadow="false"/>
    
    <geom name="floor" type="plane" size="0 0 0.05" material="grid" contype="0" conaffinity="0"/>
    """
    xml = xml.replace("</worldbody>", ghost + scene + "</worldbody>")
    assets = """
    <material name="grid" rgba="0.26 0.29 0.33 1" reflectance="0.0" specular="0.0" shininess="0.0"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.28 0.34 0.41" rgb2="0.16 0.20 0.26" width="256" height="256"/>
    """
    xml = xml.replace("<asset>", "<asset>" + assets, 1)
    xml = xml.replace("<mujoco", '<mujoco', 1)
    visual = '<visual><global offwidth="1920" offheight="1080"/><quality shadowsize="4096" offsamples="8"/><map znear="0.01"/><headlight ambient="0.34 0.34 0.36" diffuse="0.18 0.18 0.20" specular="0.05 0.05 0.05"/></visual>\n'
    xml = re.sub(r"<visual>.*?</visual>", "", xml, flags=re.S)
    xml = xml.replace("<asset>", visual + "  <asset>", 1)
    # actuators/sensors/keyframes reference names that still exist; keep the file valid by removing them anyway
    xml = re.sub(r"<actuator>.*?</actuator>", "", xml, flags=re.S)
    xml = re.sub(r"<sensor>.*?</sensor>", "", xml, flags=re.S)
    xml = re.sub(r"<keyframe>.*?</keyframe>", "", xml, flags=re.S)
    return xml


def draw_label(img: np.ndarray, text: str, sub: str, font, font_small, layout: str = "overlay") -> np.ndarray:
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im, "RGBA")
    width, height = im.size
    pad = 12
    w = min(width - 32, max(dr.textlength(text, font=font), dr.textlength(sub, font=font_small)) + 2 * pad)
    dr.rounded_rectangle([16, 16, 16 + w, 72], radius=10, fill=(22, 31, 43, 220))
    dr.text((16 + pad, 23), text, font=font, fill=(255, 255, 255, 255))
    dr.text((16 + pad, 50), sub, font=font_small, fill=(205, 220, 235, 255))
    if layout == "side-by-side":
        for x, caption, color in ((width * .34, "POLICY", (235, 240, 247, 255)),
                                  (width * .66, "REFERENCE", (103, 189, 255, 255))):
            tw = dr.textlength(caption, font=font_small)
            dr.rounded_rectangle((x - tw / 2 - 12, height - 43, x + tw / 2 + 12, height - 13),
                                 radius=8, fill=(22, 31, 43, 210))
            dr.text((x - tw / 2, height - 37), caption, font=font_small, fill=color)
    else:
        dr.rounded_rectangle((16, height - 43, 207, height - 13), radius=8, fill=(22, 31, 43, 210))
        dr.ellipse((28, height - 33, 36, height - 25), fill=(230, 235, 241, 255))
        dr.text((43, height - 37), "POLICY", font=font_small, fill=(235, 240, 247, 255))
        dr.ellipse((106, height - 33, 114, height - 25), fill=(79, 177, 255, 255))
        dr.text((121, height - 37), "REFERENCE", font=font_small, fill=(143, 205, 255, 255))
    return np.asarray(im)


def pretty(name: str) -> str:
    base = name.split("__")[0]
    base = re.sub(r"_(R|L|M)$", "", base)
    base = re.sub(r"_\d{3}$", "", base)
    return base.replace("_", " ")


def main(traj: Path, out_dir: Path, fps: int = 25, seconds: float | None = None, width: int = 1280, height: int = 720,
         azimuth: float = 140.0, elevation: float = -25.0, distance: float = 2.8, ghost: bool = True, label: bool = True, clips: list[str] | None = None,
         layout: Literal["side-by-side", "overlay"] = "overlay",
         concat: Path | None = None, segments: list[str] | None = None, titles: list[str] | None = None) -> None:
    """``segments``: per-clip time windows as ``name:start:end`` (seconds); ``titles``: ``name=Display title`` overrides."""

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError as exc:
            raise RuntimeError("Install ffmpeg or imageio-ffmpeg to render videos") from exc
    ghost_rgba = GHOST_RGBA if layout == "side-by-side" else "0.18 0.58 1.0 0.36"
    model = mujoco.MjModel.from_xml_string(two_robot_xml(Path(G1_XML), "pelvis", ghost_rgba))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height, width)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[1] = 1  # ghost
    opt.geomgroup[2] = 1  # visual meshes
    opt.geomgroup[0] = 1  # floor
    opt.flags[mujoco.mjtVisFlag.mjVIS_LIGHT] = 1
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    try:
        font_dir = Path("C:/Windows/Fonts") if os.name == "nt" else Path("/usr/share/fonts/truetype/dejavu")
        font = ImageFont.truetype(str(font_dir / ("segoeuib.ttf" if os.name == "nt" else "DejaVuSans-Bold.ttf")), 20)
        font_small = ImageFont.truetype(str(font_dir / ("segoeui.ttf" if os.name == "nt" else "DejaVuSans.ttf")), 13)
    except OSError:
        font = font_small = ImageFont.load_default()

    d = dict(np.load(traj))
    names = [str(n) for n in d["names"]]
    nq_robot = d["robot_qpos"].shape[-1]
    assert model.nq == 2 * nq_robot, (model.nq, nq_robot)
    out_dir.mkdir(parents=True, exist_ok=True)
    stride = max(1, int(round(50 / fps)))
    outputs = []
    seg = {x.split(":")[0]: (float(x.split(":")[1]), float(x.split(":")[2])) for x in (segments or [])}
    ttl = {x.split("=", 1)[0]: x.split("=", 1)[1] for x in (titles or [])}
    order = [n for n in (clips or names) if n in names]
    for name in order:
        i = names.index(name)
        T = int(d["lengths"][i])
        t0, t1 = seg.get(name, (0.0, seconds if seconds is not None else T / 50))
        a, b = int(t0 * 50), min(T, int(t1 * 50) + 1)
        qr, qf = d["robot_qpos"][i, a:b], d["ref_qpos"][i, a:b]
        T = b - a
        out = out_dir / f"{name}.mp4"
        ff = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
                               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
        # smooth camera: follow a low-passed root position
        target = (qr[0, :3] + qf[0, :3]) / 2
        angle = np.deg2rad(azimuth)
        side = np.array([-np.sin(angle), np.cos(angle)]) * 0.70
        for t in range(0, T, stride):
            data.qpos[:nq_robot] = qr[t]
            data.qpos[nq_robot:] = qf[t]
            if layout == "side-by-side" and ghost:
                data.qpos[:2] += side
                data.qpos[nq_robot:nq_robot + 2] -= side
            if not ghost:
                data.qpos[nq_robot:nq_robot + 3] = [0, 0, -5]
            mujoco.mj_forward(model, data)
            target = 0.92 * target + 0.08 * ((qr[t, :3] + qf[t, :3]) / 2)
            cam.lookat[:] = [target[0], target[1], 0.65]
            cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
            renderer.update_scene(data, cam, opt)
            img = renderer.render()
            if label:
                img = draw_label(img, ttl.get(name, pretty(name)), f"mean return {float(d['norm_return'][i]):.2f}   |   {(a + t)/50:4.1f} s", font, font_small, layout)
            ff.stdin.write(np.ascontiguousarray(img).tobytes())
        ff.stdin.close()
        if ff.wait() != 0:
            raise RuntimeError(f"ffmpeg failed while writing {out}")
        outputs.append(out)
        print("wrote", out, f"{T/50:.1f} s")
    if concat and outputs:
        lst = out_dir / "concat.txt"
        lst.write_text("".join(f"file '{p.resolve()}'\n" for p in outputs))
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(concat)], check=True)
        print("wrote", concat)


if __name__ == "__main__":
    tyro.cli(main)
