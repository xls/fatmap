#!/usr/bin/env python3
"""
Convert the sandbox assets in data/ (JPG) into uncompressed 32-bit TGA, a
format fatmap reads without any decoder library (fm_surface_load_tga).

  troll_diffuse.tga  colors.jpg + emissive.jpg (additive glow), alpha from alpha.jpg
  pedestal.tga       pedestal.jpg
  grass.tga          R.jpg (ground texture)
  <Model>_<i>.tga    images embedded in every data/*.glb (glTF binary)

Needs ffmpeg on PATH (development tool only; nothing links against it).
Usage: python tools/convert_assets.py [data_dir]
"""
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data")

JOBS = [
    ("troll_diffuse.tga", ["colors.jpg", "emissive.jpg", "alpha.jpg"],
     "[0]format=gbrp[c];[1]format=gbrp[e];[c][e]blend=all_mode=addition,format=rgb24[ce];"
     "[2]format=gray[a];[ce][a]alphamerge,format=bgra"),
    ("pedestal.tga", ["pedestal.jpg"], "format=bgra"),
    ("grass.tga", ["R.jpg"], "format=bgra"),
]


def glb_images(path):
    """yield (index, mime, bytes) for images stored in the GLB binary chunk"""
    d = open(path, "rb").read()
    if d[:4] != b"glTF":
        return
    jlen = struct.unpack("<I", d[12:16])[0]
    j = json.loads(d[20:20 + jlen])
    boff = 20 + jlen
    blen = struct.unpack("<I", d[boff:boff + 4])[0]
    bin_ = d[boff + 8:boff + 8 + blen]
    for i, img in enumerate(j.get("images", [])):
        if "bufferView" not in img:
            continue
        bv = j["bufferViews"][img["bufferView"]]
        o = bv.get("byteOffset", 0)
        yield i, img.get("mimeType", "image/png"), bin_[o:o + bv["byteLength"]]


def convert_glb(path):
    name = os.path.splitext(os.path.basename(path))[0]
    for i, mime, data in glb_images(path):
        ext = ".jpg" if "jpeg" in mime else ".png"
        fd, tmp = tempfile.mkstemp(suffix=ext)
        os.write(fd, data)
        os.close(fd)
        out = os.path.join(DATA, "%s_%d.tga" % (name, i))
        print("->", os.path.basename(out))
        subprocess.check_call(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", tmp,
                               "-vf", "format=bgra", "-rle", "0", "-frames:v", "1", out])
        os.remove(tmp)


def main():
    if not shutil.which("ffmpeg"):
        sys.exit("error: ffmpeg not found on PATH")
    for out, inputs, filt in JOBS:
        srcs = [os.path.join(DATA, i) for i in inputs]
        missing = [s for s in srcs if not os.path.exists(s)]
        if missing:
            print("skip %s (missing %s)" % (out, ", ".join(missing)))
            continue
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
        for s in srcs:
            cmd += ["-i", s]
        if len(srcs) > 1:
            cmd += ["-filter_complex", filt]
        else:
            cmd += ["-vf", filt]
        cmd += ["-rle", "0", "-frames:v", "1", os.path.join(DATA, out)]
        print("->", out)
        subprocess.check_call(cmd)
    for f in sorted(os.listdir(DATA)):
        if f.lower().endswith(".glb"):
            convert_glb(os.path.join(DATA, f))


if __name__ == "__main__":
    main()
