# Sandbox sample data

Assets used by `fatmap_sandbox`. The library itself never reads these.
Textures are decoded at load time (JPEG / PNG, including images embedded in
`.glb` files) with `sandbox/stb/stb_image.h`, so no conversion step is needed.

| file | scene | source | license |
|---|---|---|---|
| `Fox.glb` | 9 characters | [Khronos glTF-Sample-Assets: Fox](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/Fox) | CC0 1.0 (model) + CC BY 4.0 (rig, animation, glTF conversion) |
| `CesiumMan.glb` | 9 characters | [Khronos glTF-Sample-Assets: CesiumMan](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/CesiumMan) | CC BY 4.0; the texture shows the Cesium logo (trademark) |
| `DamagedHelmet.glb` | 0 helmet | [Khronos glTF-Sample-Assets: DamagedHelmet](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/DamagedHelmet) | CC BY 4.0 (rebuild) + **CC BY-NC 4.0** (original model): non-commercial use only |

Attribution:

- Fox: model by PixelMannen (CC0 1.0); rigging and animation by tomkranis
  (CC BY 4.0); conversion to glTF by @AsoboStudio and @scurest (CC BY 4.0).
- CesiumMan: (c) 2017 Cesium, CC BY 4.0. The Cesium logo is a trademark of
  Cesium.
- DamagedHelmet: (c) 2018 ctxwing, rebuild and conversion to glTF, CC BY 4.0;
  (c) 2016 theblueturtle_, original "Battle Damaged Sci-fi Helmet",
  CC BY-NC 4.0.

License texts: [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/legalcode),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode),
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode).
These licenses cover the asset files only, not fatmap's source code.

## Not included: the troll scene (8)

Scene 8 renders a troll model whose origin and license are unknown, so it is
not in the repository. If you have the files, put them here and the scene
picks them up; otherwise it shows an empty background, and scene 9 uses a
generated ground texture instead of `R.jpg`:

```
troll2.obj  troll.mtl  troll2.mtl
colors.jpg  emissive.jpg  alpha.jpg  pedestal.jpg  R.jpg
```
