"""Convert a single .obj or .stl mesh into MuJoCo MJCF assets.

This is a generic mesh-to-MJCF converter, derived from
github.com/discoverse-dev/DISCOVERSE/scripts/mesh2mjcf.py but stripped of any
DISCOVERSE-specific imports or scene wiring. It is designed to consume the
per-object meshes that this pipeline emits under `<scene>/3d_assets/<obj>.obj`,
but works on any standalone mesh file.

Output layout (under --output-dir, which defaults to the input file's parent —
typically `scene_dir/3d_assets/` when consuming the v2 pipeline outputs):

    <output>/
        <asset_name>/                       (per-asset folder, named after the obj stem)
            <asset_name>.obj                (copy of the input mesh)
            <asset_name>.mtl                (if multi-material)
            <texture files...>              (referenced by the MTL)
            part_0.obj part_1.obj ...       (if --convex_decomposition)
            mjcf/
                <asset_name>.xml
                <asset_name>_dependencies.xml

Mesh paths inside the emitted XML are written as `<asset_name>/<file>`, so the
consuming MuJoCo scene should set `meshdir` (and `texturedir`) to <output>.

Examples:

    # Basic conversion (default RGBA, mass, inertia; no free joint; no decomp).
    python pipeline/mesh2mjcf.py path/to/cup.obj

    # Specify RGBA, mass, inertia.
    python pipeline/mesh2mjcf.py path/to/cup.obj \\
        --rgba 0.8 0.2 0.2 1.0 --mass 0.5 --diaginertia 0.01 0.01 0.005

    # Free-floating object.
    python pipeline/mesh2mjcf.py path/to/cup.obj --free_joint

    # Convex decomposition for accurate collisions.
    python pipeline/mesh2mjcf.py path/to/cup.obj -cd

    # Preview in MuJoCo viewer after conversion.
    python pipeline/mesh2mjcf.py path/to/cup.obj --verbose

Notes:
    - Multi-material OBJ files are auto-detected (via the MTL file) and split
      into one sub-mesh per material; each material yields a MuJoCo
      `<material>`, with textures (`map_Kd`) copied alongside.
    - Convex decomposition requires `pip install coacd trimesh`.
    - Material splitting requires `pip install trimesh`.
"""

import argparse
import logging
import os
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ===== MTL handling =====

# MTL fields relevant to MuJoCo.
MTL_FIELDS = (
    "Ka",      # Ambient color
    "Kd",      # Diffuse color
    "Ks",      # Specular color
    "d",       # Transparency (alpha)
    "Tr",      # 1 - transparency
    "Ns",      # Shininess
    "map_Kd",  # Diffuse texture map
)


@dataclass
class Material:
    """Convenience container for MTL → MuJoCo material conversion."""

    name: str
    Ka: Optional[str] = None
    Kd: Optional[str] = None
    Ks: Optional[str] = None
    d: Optional[str] = None
    Tr: Optional[str] = None
    Ns: Optional[str] = None
    map_Kd: Optional[str] = None

    @staticmethod
    def from_string(lines: Sequence[str]) -> "Material":
        attrs = {"name": lines[0].split(" ")[1].strip()}
        for line in lines[1:]:
            for attr in MTL_FIELDS:
                if line.startswith(attr):
                    elems = line.split(" ")[1:]
                    elems = [elem for elem in elems if elem != ""]
                    attrs[attr] = " ".join(elems)
                    break
        return Material(**attrs)

    def mjcf_rgba(self) -> str:
        Kd = self.Kd or "1.0 1.0 1.0"
        if self.d is not None:
            alpha = self.d
        elif self.Tr is not None:
            alpha = str(1.0 - float(self.Tr))
        else:
            alpha = "1.0"
        return f"{Kd} {alpha}"

    def mjcf_shininess(self) -> str:
        if self.Ns is not None:
            # Ns values are typically 0-1000; normalize to [0, 1].
            ns_val = float(self.Ns) / 1_000
        else:
            ns_val = 0.5
        return f"{ns_val}"

    def mjcf_specular(self) -> str:
        if self.Ks is not None:
            # Average the specular RGB to a scalar.
            ks_val = sum(map(float, self.Ks.split(" "))) / 3
        else:
            ks_val = 0.5
        return f"{ks_val}"


def parse_mtl_name(lines: Sequence[str]) -> Optional[str]:
    """Return the .mtl filename referenced by an OBJ file's `mtllib` directive."""
    mtl_regex = re.compile(r"^mtllib\s+(.+?\.mtl)(?:\s*#.*)?\s*\n?$")
    for line in lines:
        match = mtl_regex.match(line)
        if match is not None:
            return match.group(1)
    return None


def copy_obj_with_mtl(obj_source: Path, obj_target: Path) -> None:
    """Copy an OBJ file, plus the MTL file it references (if any)."""
    obj_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(obj_source, obj_target)

    try:
        with open(obj_source, "r") as f:
            lines = f.readlines()
        for line in lines:
            if line.strip().startswith("mtllib "):
                mtl_filename = line.strip().split()[1]
                mtl_source = obj_source.parent / mtl_filename
                mtl_target = obj_target.parent / mtl_filename
                if mtl_source.exists():
                    shutil.copy2(mtl_source, mtl_target)
                    print(f"Copied MTL file: {mtl_source} -> {mtl_target}")
                break
    except Exception as e:
        print(f"Warning: failed to check/copy MTL file for {obj_source}: {e}")


def parse_mtl_file(mtl_path: Path) -> Dict[str, Material]:
    """Parse an MTL file into a name → Material dict."""
    materials: Dict[str, Material] = {}
    if not mtl_path.exists():
        return materials

    with open(mtl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    lines = [line for line in lines if not line.startswith("#")]
    lines = [line for line in lines if line.strip()]
    lines = [line.strip() for line in lines]

    sub_mtls: List[List[str]] = []
    for line in lines:
        if line.startswith("newmtl"):
            sub_mtls.append([])
        if sub_mtls:
            sub_mtls[-1].append(line)

    for sub_mtl in sub_mtls:
        if sub_mtl:
            material = Material.from_string(sub_mtl)
            materials[material.name] = material

    return materials


def split_obj_by_materials(
    obj_path: Path, output_dir: Path
) -> Tuple[Dict[str, Material], List[str]]:
    """Split a multi-material OBJ into one sub-mesh per material.

    Returns (materials, submesh_files). If the OBJ has zero or one materials,
    submesh_files is empty and the OBJ is left as a single file.
    """
    materials: Dict[str, Material] = {}
    submesh_files: List[str] = []

    with open(obj_path, "r", encoding="utf-8") as f:
        obj_lines = f.readlines()

    mtl_name = parse_mtl_name(obj_lines)
    if mtl_name:
        mtl_path = obj_path.parent / mtl_name
        materials = parse_mtl_file(mtl_path)

    if len(materials) <= 1:
        return materials, []

    try:
        import trimesh
    except ImportError:
        print("Warning: trimesh not installed; cannot split multi-material OBJ.")
        return materials, []

    try:
        mesh = trimesh.load(
            obj_path,
            split_object=True,
            group_material=True,
            process=False,
            maintain_order=False,
        )

        if isinstance(mesh, trimesh.base.Trimesh):
            # Single mesh after grouping; nothing to split.
            target_file = output_dir / f"{obj_path.stem}.obj"
            shutil.copy(obj_path, target_file)
            return materials, []

        obj_stem = obj_path.stem
        print(f"Splitting OBJ by material: {len(mesh.geometry)} sub-meshes")
        for i, (material_name, geom) in enumerate(mesh.geometry.items()):
            submesh_file = f"{obj_stem}_{i}.obj"
            submesh_path = output_dir / submesh_file

            geom.visual.material.name = material_name
            geom.export(str(submesh_path), include_texture=True, header=None)
            submesh_files.append(submesh_file)
            print(f"  saved sub-mesh: {submesh_file} (material: {material_name})")

        # trimesh sometimes emits a stray `material.mtl` next to the export.
        temp_mtl = output_dir / "material.mtl"
        if temp_mtl.exists():
            temp_mtl.unlink()

        return materials, submesh_files
    except Exception as e:
        print(f"Warning: failed to split OBJ by material: {e}")
        return materials, []


# ===== XML builders =====

def create_asset_xml(asset_name, convex_parts=None, materials=None, submesh_files=None):
    """Build the `<mujocoinclude>` element listing meshes/materials/textures."""
    root = ET.Element("mujocoinclude")
    asset = ET.SubElement(root, "asset")

    if materials:
        for material_name, material in materials.items():
            material_elem = ET.SubElement(asset, "material")
            material_elem.set("name", f"{asset_name}_{material_name}")
            material_elem.set("rgba", material.mjcf_rgba())
            material_elem.set("specular", material.mjcf_specular())
            material_elem.set("shininess", material.mjcf_shininess())

            if material.map_Kd:
                texture_elem = ET.SubElement(asset, "texture")
                texture_elem.set("type", "2d")
                texture_elem.set("name", f"{asset_name}_{material_name}_texture")
                texture_elem.set("file", f"{asset_name}/{material.map_Kd}")

                material_elem.set("texture", f"{asset_name}_{material_name}_texture")
                material_elem.attrib.pop("rgba", None)

    # Main mesh (only when not split by material).
    if not submesh_files:
        mesh_elem = ET.SubElement(asset, "mesh")
        mesh_elem.set("name", asset_name)
        mesh_elem.set("file", f"{asset_name}/{asset_name}.obj")

    # Per-material sub-meshes.
    if submesh_files:
        for submesh_file in submesh_files:
            submesh_name = submesh_file.replace(".obj", "")
            part_mesh = ET.SubElement(asset, "mesh")
            part_mesh.set("name", submesh_name)
            part_mesh.set("file", f"{asset_name}/{submesh_file}")

    # Convex-decomposition parts.
    if convex_parts:
        for i in range(convex_parts):
            part_mesh = ET.SubElement(asset, "mesh")
            part_mesh.set("name", f"{asset_name}_part_{i}")
            part_mesh.set("file", f"{asset_name}/part_{i}.obj")

    return root


def create_geom_xml(
    asset_name,
    mass,
    diaginertia,
    rgba,
    free_joint=False,
    convex_parts=None,
    materials=None,
    submesh_files=None,
    output_dir=None,
):
    """Build the `<mujocoinclude>` element with the body's geoms + inertial."""
    root = ET.Element("mujocoinclude")

    if free_joint:
        joint_elem = ET.SubElement(root, "joint")
        joint_elem.set("type", "free")

    inertial_elem = ET.SubElement(root, "inertial")
    inertial_elem.set("pos", "0 0 0")
    inertial_elem.set("mass", str(mass))
    inertial_elem.set(
        "diaginertia", f"{diaginertia[0]} {diaginertia[1]} {diaginertia[2]}"
    )

    if submesh_files and materials:
        # Multi-material: one geom per sub-mesh.
        for submesh_file in submesh_files:
            submesh_name = submesh_file.replace(".obj", "")
            geom_elem = ET.SubElement(root, "geom")
            geom_elem.set("type", "mesh")
            geom_elem.set("mesh", submesh_name)
            geom_elem.set("class", "obj_visual")

            material_assigned = False
            submesh_path = Path(output_dir) / asset_name / submesh_file
            if submesh_path.exists():
                try:
                    with open(submesh_path, "r", encoding="utf-8") as f:
                        submesh_lines = f.readlines()
                    for line in submesh_lines:
                        line = line.strip()
                        if line.startswith("usemtl "):
                            mtl_name = line.split()[1]
                            geom_elem.set("material", f"{asset_name}_{mtl_name}")
                            material_assigned = True
                            break
                except Exception as e:
                    print(f"Warning: could not read sub-mesh {submesh_path}: {e}")

            if not material_assigned:
                geom_elem.set(
                    "rgba", f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"
                )

    elif materials and len(materials) == 1:
        # Single material with possible texture.
        geom_elem = ET.SubElement(root, "geom")
        geom_elem.set("type", "mesh")
        geom_elem.set("mesh", asset_name)
        geom_elem.set("class", "obj_visual")
        material_name = next(iter(materials))
        geom_elem.set("material", f"{asset_name}_{material_name}")

    elif convex_parts:
        # Visual geom (full mesh) + collision geoms (convex parts).
        visual_geom = ET.SubElement(root, "geom")
        visual_geom.set("rgba", f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}")
        visual_geom.set("mesh", asset_name)
        visual_geom.set("class", "obj_visual")

        for i in range(convex_parts):
            collision_geom = ET.SubElement(root, "geom")
            collision_geom.set("type", "mesh")
            collision_geom.set("rgba", f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}")
            collision_geom.set("mesh", f"{asset_name}_part_{i}")

    else:
        # Simple solid-colour mesh geom.
        geom_elem = ET.SubElement(root, "geom")
        geom_elem.set("type", "mesh")
        geom_elem.set("rgba", f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}")
        geom_elem.set("mesh", asset_name)

    # When a material/sub-mesh path was taken AND convex decomposition is on,
    # still emit invisible collision geoms.
    if convex_parts and (submesh_files or (materials and len(materials) == 1)):
        for i in range(convex_parts):
            collision_geom = ET.SubElement(root, "geom")
            collision_geom.set("type", "mesh")
            collision_geom.set("rgba", f"{rgba[0]} {rgba[1]} {rgba[2]} 0")
            collision_geom.set("mesh", f"{asset_name}_part_{i}")

    return root


def save_xml_with_formatting(root, filepath):
    """Indent and write an ElementTree XML file (Python 3.9+)."""
    ET.indent(root, space="  ", level=0)
    tree = ET.ElementTree(root)
    tree.write(filepath, encoding="utf-8", xml_declaration=False)


def create_preview_xml(asset_name):
    """Build a minimal preview scene for `mujoco.viewer`."""
    root = ET.Element("mujoco")
    root.set("model", "temp_preview_env")

    option = ET.SubElement(root, "option")
    option.set("gravity", "0 0 -9.81")

    compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", ".")
    compiler.set("texturedir", ".")

    include = ET.SubElement(root, "include")
    include.set("file", f"{asset_name}/mjcf/{asset_name}_dependencies.xml")

    default = ET.SubElement(root, "default")
    obj_default = ET.SubElement(default, "default")
    obj_default.set("class", "obj_visual")
    geom_default = ET.SubElement(obj_default, "geom")
    geom_default.set("group", "2")
    geom_default.set("type", "mesh")
    geom_default.set("contype", "0")
    geom_default.set("conaffinity", "0")

    worldbody = ET.SubElement(root, "worldbody")

    floor_geom = ET.SubElement(worldbody, "geom")
    floor_geom.set("name", "floor")
    floor_geom.set("type", "plane")
    floor_geom.set("size", "2 2 0.1")
    floor_geom.set("rgba", ".8 .8 .8 1")

    light = ET.SubElement(worldbody, "light")
    light.set("pos", "0 0 3")
    light.set("dir", "0 0 -1")

    body = ET.SubElement(worldbody, "body")
    body.set("name", asset_name)
    body.set("pos", "0 0 0.5")

    body_include = ET.SubElement(body, "include")
    body_include.set("file", f"{asset_name}/mjcf/{asset_name}.xml")

    return root


# ===== Main =====

def _build_argparser():
    parser = argparse.ArgumentParser(
        description="Convert a .obj or .stl mesh into MuJoCo MJCF assets."
    )
    parser.add_argument(
        "input_file", type=str, help="Path to the input mesh (.obj or .stl)."
    )
    parser.add_argument(
        "--rgba",
        nargs=4,
        type=float,
        default=[0.5, 0.5, 0.5, 1],
        help="Mesh RGBA colour. Default: [0.5, 0.5, 0.5, 1].",
    )
    parser.add_argument(
        "--mass",
        type=float,
        default=0.001,
        help="Mesh mass (kg). Default: 0.001.",
    )
    parser.add_argument(
        "--diaginertia",
        nargs=3,
        type=float,
        default=[0.00002, 0.00002, 0.00002],
        help="Diagonal inertia tensor. Default: [2e-5, 2e-5, 2e-5].",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help=(
            "Output assets root. Default: the input file's parent directory, "
            "so that `scene_dir/3d_assets/foo.obj` writes to `scene_dir/`."
        ),
    )
    parser.add_argument(
        "--free_joint",
        action="store_true",
        help="Add a free joint so the body can move.",
    )
    parser.add_argument(
        "-cd",
        "--convex_decomposition",
        action="store_true",
        help=(
            "Decompose the mesh into convex parts for accurate collision. "
            "Requires `coacd` and `trimesh`."
        ),
    )
    parser.add_argument(
        "--scene",
        action="store_true",
        help="Use high-precision CoACD config (smaller threshold).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Open MuJoCo viewer with a preview scene after conversion.",
    )
    return parser


def main():
    args = _build_argparser().parse_args()

    input_file = args.input_file
    rgba = args.rgba
    mass = args.mass
    diaginertia = args.diaginertia
    free_joint = args.free_joint
    convex_de = args.convex_decomposition
    verbose = args.verbose

    if args.output is None:
        output_assets_dir = str(Path(input_file).resolve().parent)
    else:
        output_assets_dir = args.output

    if convex_de:
        try:
            import coacd  # noqa: F401
            import trimesh  # noqa: F401
        except ImportError:
            print(
                "Error: `coacd` and `trimesh` are required for "
                "--convex_decomposition. Install with `pip install coacd trimesh`."
            )
            raise SystemExit(1)

    if input_file.endswith(".obj"):
        asset_name = os.path.basename(input_file)[: -len(".obj")]
    elif input_file.endswith(".stl"):
        asset_name = os.path.basename(input_file)[: -len(".stl")]
    else:
        raise SystemExit(
            f"Error: {input_file} is not a supported mesh type. Use .obj or .stl."
        )

    # Per-asset folder lives directly under <output>, with an `mjcf/` subfolder
    # for the generated XML files. This way the whole asset (meshes + MTL +
    # textures + convex parts + MJCF) is self-contained in one directory.
    output_dir = os.path.join(output_assets_dir, asset_name)
    mjcf_obj_dir = os.path.join(output_dir, "mjcf")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    os.makedirs(mjcf_obj_dir, exist_ok=True)

    # Copy the mesh (and MTL if relevant) into the per-asset folder.
    if os.path.dirname(input_file) != output_dir:
        if input_file.endswith(".obj"):
            copy_obj_with_mtl(
                Path(input_file), Path(output_dir) / Path(input_file).name
            )
        else:
            shutil.copy(input_file, output_dir)

    # Material splitting (OBJ only).
    materials: Dict[str, Material] = {}
    submesh_files: List[str] = []
    if input_file.endswith(".obj"):
        print("Checking OBJ for multiple materials...")
        obj_path = Path(output_dir) / f"{asset_name}.obj"
        materials, submesh_files = split_obj_by_materials(obj_path, Path(output_dir))

        # Copy referenced texture files (single or multi-material case).
        if materials:
            input_parent = Path(input_file).parent
            for _name, material in materials.items():
                if material.map_Kd:
                    texture_src = input_parent / material.map_Kd
                    if texture_src.exists():
                        texture_dst = Path(output_dir) / material.map_Kd
                        shutil.copy(texture_src, texture_dst)
                        print(f"Copied texture: {material.map_Kd}")

        if submesh_files:
            print(f"Split into {len(submesh_files)} sub-meshes.")
        elif len(materials) == 1:
            print("Single material; no split needed.")
        else:
            print("No materials detected.")

    convex_parts_count = 0
    if convex_de:
        import coacd
        import trimesh

        print(f"Running convex decomposition on {asset_name}...")
        mesh = trimesh.load(input_file, force="mesh")
        mesh_coacd = coacd.Mesh(mesh.vertices, mesh.faces)
        coacd_config_scene = {
            "threshold": 0.01,
            "preprocess_resolution": 100,
        }
        coacd_config = coacd_config_scene if args.scene else {}
        parts = coacd.run_coacd(mesh_coacd, **coacd_config)

        for i, part in enumerate(parts):
            part_filename = f"part_{i}.obj"
            output_part_file = os.path.join(output_dir, part_filename)
            part_mesh = trimesh.Trimesh(vertices=part[0], faces=part[1])
            part_mesh.export(output_part_file)

        convex_parts_count = len(parts)
        print(f"{asset_name} decomposed into {convex_parts_count} convex parts.")

    # Emit the asset dependency XML.
    asset_xml = create_asset_xml(
        asset_name,
        convex_parts_count if convex_de else None,
        materials if (submesh_files or len(materials) == 1) else None,
        submesh_files if submesh_files else None,
    )
    asset_file_path = os.path.join(mjcf_obj_dir, f"{asset_name}_dependencies.xml")
    save_xml_with_formatting(asset_xml, asset_file_path)

    # Emit the body geom XML.
    geom_xml = create_geom_xml(
        asset_name,
        mass,
        diaginertia,
        rgba,
        free_joint,
        convex_parts_count if convex_de else None,
        materials if (submesh_files or len(materials) == 1) else None,
        submesh_files if submesh_files else None,
        output_assets_dir,
    )
    geom_file_path = os.path.join(mjcf_obj_dir, f"{asset_name}.xml")
    save_xml_with_formatting(geom_xml, geom_file_path)

    print(f"Converted {asset_name} to MJCF.")
    print(f"  meshes: {output_dir}")
    print(f"  dependencies: {asset_file_path}")
    print(f"  body geom: {geom_file_path}")
    if submesh_files:
        print(
            f"  material split: {len(submesh_files)} sub-meshes, "
            f"{len(materials)} materials"
        )

    if verbose:
        print("\nLaunching MuJoCo viewer...")
        py_dir = shutil.which("python") or shutil.which("python3")
        if not py_dir:
            print("Error: no `python`/`python3` on PATH; cannot launch viewer.")
            raise SystemExit(1)

        tmp_world_mjcf = os.path.join(output_assets_dir, "_tmp_preview.xml")
        preview_xml = create_preview_xml(asset_name)
        save_xml_with_formatting(preview_xml, tmp_world_mjcf)

        cmd_line = f"{py_dir} -m mujoco.viewer --mjcf {tmp_world_mjcf}"
        print(f"Running: {cmd_line}")
        os.system(cmd_line)


if __name__ == "__main__":
    main()
