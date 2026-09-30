#!/usr/bin/env python3
"""
Convert the sandbox assets in data/ (JPG) into uncompressed 32-bit TGA, a
format fatmap reads without any decoder library (fm_surface_load_tga).

  troll_diffuse.tga  colors.jpg + emissive.jpg (additive glow), alpha from alpha.jpg
  pedestal.tga       pedestal.jpg
  grass.tga          R.jpg (ground texture)

Needs ffmpeg on PATH (development tool only; nothing links against it).
Usage: python tools/convert_assets.py [data_dir]
"""
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data")

JOBS = [
    ("troll_diffuse.tga", ["colors.jpg", "emissive.jpg", "alpha.jpg"],
     "[0]format=gbrp[c];[1]format=gbrp[e];[c][e]blend=all_mode=addition,format=rgb24[ce];"
     "[2]format=gray[a];[ce][a]alphamerge,format=bgra"),
    ("pedestal.tga", ["pedestal.jpg"], "format=bgra"),
    ("grass.tga", ["R.jpg"], "format=bgra"),
]


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


if __name__ == "__main__":
    main()
