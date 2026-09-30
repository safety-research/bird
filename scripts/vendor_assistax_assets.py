#!/usr/bin/env python3
"""Vendor the Assistax scenes into `bird/envs/assets/assistax/`.

    uv run --no-sync --with pyfqmr python3 scripts/vendor_assistax_assets.py [--src DIR]
    uv run --no-sync --with pyfqmr python3 scripts/vendor_assistax_assets.py --check

Upstream is `assistive-autonomy/assistax` (Apache-2.0), whose five robot-human scenes are
MuJoCo MJCF files with OBJ meshes. `bird/envs/assistax.py` drives them through the plain
`mujoco` package, so the scenes have to be IN this repo rather than in `refs/`, which is
not distributed: an adapter whose asset resolves on one machine and not on another dies at
env construction with no run output.

WHY THIS IS A SCRIPT AND NOT A `cp`. The five scenes reference 58 OBJ meshes totalling
33.3 MB -- the Panda's shells, the bed, the wheelchair -- with six files between 3.0 and
3.9 MB. That is far over what a 448x320 rollout frame can show. The saving (33.3 MB -> 6.9 MB) is almost entirely OBJ text -> binary STL: the
OBJ files carry no texture coordinates the scenes use (every mesh geom takes its colour
from a `material`/`rgba`), so the conversion loses nothing a frame shows. Decimation is a
second, smaller step, and it has a RULE:

  * No mesh geom collides in any of the five scenes (every mesh-type geom has `contype`
    and `conaffinity` 0; collision is capsules, boxes and spheres throughout). But SIX
    colliding primitives are FITTED from meshes -- `<geom type="box" mesh="link6_16">`
    and the like on the Panda's hand and fingers -- and MuJoCo fits the primitive to the
    mesh's vertices, so a decimated `link6_16` moves the hand's collision box.
    "Decimation changes only the picture" holds for the current pass only by accident:
    pyfqmr, asked for max(1500, 10%) triangles, reduces only the bed and the wheelchair
    and leaves every Panda mesh within 8 triangles of upstream's, so the fitted boxes come
    out identical (measured against upstream's OBJs: 0.0 on every colliding geom's size
    and pose). A re-vendor with a simplifier that did reduce them would change the
    robot's collision geometry under a manifest that said nothing had.
  * So `build` PROTECTS a mesh from decimation when any geom that references it -- as a
    mesh geom or as the template of a fitted primitive, in any scene -- collides, or sits
    on a body that can move and takes its inertia from its geoms (a decimated visual
    shell on such a link would move its inertia; the Panda's links carry explicit
    <inertial> elements, so theirs do not). Which meshes those are is read off the
    compiled models (`geom_dataid` is kept on fitted primitives too), never hand-listed
    (`protected_meshes`). Everything else is decimated to max(1500, 10%) triangles
    (pyfqmr) and, protected or not, written as binary STL.
  * `build` then compiles upstream's scenes (OBJ) and the vendored ones (STL) side by side
    and asserts, for every colliding geom, identical `geom_size`/`geom_pos`/`geom_quat`,
    and for every body that can move, identical mass, inertia and centre of mass. The
    measurement is written to `PROVENANCE.json:fidelity` and `--check` re-derives it.
  * The XML files are copied with ONE mechanical edit: each `<mesh ... file="X.obj"/>`
    is repointed at `X.stl`. Nothing else in them changes -- not a joint, gain, size
    or camera -- and `--check` proves it by re-deriving the copies from a fresh source.

Every number needed to check the transformation (upstream commit, per-file sha256 in and
out, triangle counts, which meshes were protected and why, the fidelity comparison) is
written to `PROVENANCE.json` beside the assets, and `--check` re-derives the output from
the source, diffs the digests AND hashes the committed files against them (both halves: a
fresh build that matches a stale manifest says nothing about the bytes in the tree, and an
unrecorded file beside them is drift too).

The source tree is the upstream clone at the pinned commit -- fetched into a temporary
directory by default, or taken from `--src DIR` (a checkout whose HEAD is that commit; a
directory with no `.git` is accepted with a warning and recorded as unverified).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DST = REPO / "bird" / "envs" / "assets" / "assistax"

UPSTREAM_REPO = "assistive-autonomy/assistax"
#: The commit the vendored files were derived from. Move it deliberately, then re-run.
UPSTREAM_COMMIT = "a7d94f4e20636b9b0c07370344b8a2b521db58b2"
UPSTREAM_ASSETS = "assistax/envs/assets"

#: The five robot-human scenes (`pushcoop` and `handover` are robot-robot and not ported)
#: and the include directories they pull in.
SCENES = ("wheelchair_scene.xml", "bed_scene.xml", "bed_scene_armmanip.xml",
          "feeding_scene.xml", "teethbrushing_scene.xml")
ASSET_DIRS = ("wheelchair_assets", "bed_assets", "feeding_assets", "teethbrushing_assets")
MIN_TRIANGLES = 1500
KEEP_FRACTION = 0.10
AGGRESSIVENESS = 7

_MESH_RE = re.compile(r'(<mesh\b[^>]*\bfile=")([^"]+)\.obj(")')


# --------------------------------------------------------------------------
# OBJ in, binary STL out, pure numpy
# --------------------------------------------------------------------------

_TRI_DTYPE = np.dtype([("normal", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])


def read_obj(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Vertices (n, 3) float64 and triangles (m, 3) int64 from an OBJ file.

    Positions and faces only: normals, texture coordinates, groups and materials are
    dropped (see the module docstring for why that loses nothing here). Polygons are
    fan-triangulated; negative indices are relative, as the format allows.
    """
    verts: List[List[float]] = []
    tris: List[List[int]] = []
    with path.open("r", errors="replace") as fh:
        for line in fh:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                idx = []
                for tok in line.split()[1:]:
                    i = int(tok.split("/")[0])
                    idx.append(i - 1 if i > 0 else len(verts) + i)
                for k in range(1, len(idx) - 1):
                    tris.append([idx[0], idx[k], idx[k + 1]])
    v = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    t = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    if v.shape[0] == 0 or t.shape[0] == 0:
        raise ValueError(f"{path}: no vertices or no faces")
    if t.max() >= v.shape[0] or t.min() < 0:
        raise ValueError(f"{path}: face index out of range")
    return v, t


def write_stl(path: Path, v: np.ndarray, t: np.ndarray) -> None:
    tri = v[t]                                           # (m, 3, 3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.where(norm > 0, n / np.where(norm > 0, norm, 1.0), 0.0)
    rec = np.zeros(tri.shape[0], dtype=_TRI_DTYPE)
    rec["normal"] = n.astype(np.float32)
    rec["v"] = tri.astype(np.float32)
    # fixed text ("decimated" is inaccurate for a protected mesh) so the binaries' bytes,
    # pinned by sha256, do not depend on the protection rule; PROVENANCE.json is the
    # record of what happened to each mesh
    header = b"bird vendored assistax mesh (decimated, see PROVENANCE.json)".ljust(80, b" ")[:80]
    with path.open("wb") as fh:
        fh.write(header)
        fh.write(np.asarray([tri.shape[0]], dtype="<u4").tobytes())
        fh.write(rec.tobytes())


def decimate(v: np.ndarray, t: np.ndarray, target: int) -> Tuple[np.ndarray, np.ndarray]:
    if t.shape[0] <= target:
        return v, t
    import pyfqmr

    simp = pyfqmr.Simplify()
    simp.setMesh(np.ascontiguousarray(v, dtype=np.float64),
                 np.ascontiguousarray(t, dtype=np.int32))
    simp.simplify_mesh(target_count=int(target), aggressiveness=AGGRESSIVENESS,
                       preserve_border=True, verbose=0)
    v2, t2, _n = simp.getMesh()
    return np.asarray(v2, dtype=np.float64), np.asarray(t2, dtype=np.int64)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# the source tree
# --------------------------------------------------------------------------

def _head(dir_: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(dir_), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def fetch_source(tmp: Path) -> Path:
    """A shallow checkout of exactly `UPSTREAM_COMMIT` (the commit, not a branch)."""
    dst = tmp / "assistax"
    dst.mkdir()
    run = lambda *a: subprocess.run(["git", "-C", str(dst), *a], check=True,  # noqa: E731
                                    capture_output=True, text=True)
    run("init", "-q")
    run("remote", "add", "origin", f"https://github.com/{UPSTREAM_REPO}.git")
    run("fetch", "-q", "--depth", "1", "origin", UPSTREAM_COMMIT)
    run("checkout", "-q", "FETCH_HEAD")
    return dst


def referenced_meshes(xml_paths: List[Path]) -> List[str]:
    names = set()
    for p in xml_paths:
        for m in _MESH_RE.finditer(p.read_text()):
            names.add(m.group(2))
    return sorted(names)


# --------------------------------------------------------------------------
# what the physics reads off a mesh
# --------------------------------------------------------------------------

def _moving_bodies(m) -> List[bool]:
    """A body moves if it or any ancestor carries a degree of freedom."""
    moving = [False] * m.nbody
    for b in range(1, m.nbody):
        moving[b] = bool(m.body_dofnum[b] > 0) or moving[int(m.body_parentid[b])]
    return moving


def _explicit_inertials(all_xmls: List[Path]) -> Dict[str, bool]:
    """body name -> whether the XML gives it an <inertial> of its own. The compiled model
    does not record where a body's mass came from, and under the compiler's default
    `inertiafromgeom="auto"` a body WITH an <inertial> ignores its geoms' mass while one
    without derives mass and inertia from them -- so a visual mesh on the latter is
    physics. `inertiafromgeom="true"` anywhere makes every body the latter."""
    import xml.etree.ElementTree as ET
    out: Dict[str, bool] = {}
    force_geoms = False
    for p in all_xmls:
        text = p.read_text()
        if 'inertiafromgeom="true"' in text:
            force_geoms = True
        # an include file may hold several top-level elements: parse under a synthetic root
        body = re.sub(r"^\s*<\?xml[^>]*\?>", "", text, count=1)
        root = ET.fromstring("<_bird_root>" + body + "</_bird_root>")
        for b in root.iter("body"):
            name = b.get("name")
            if name:
                out[name] = b.find("inertial") is not None
    return {} if force_geoms else out


def protected_meshes(scene_xmls: List[Path], all_xmls: List[Path]) -> Dict[str, List[str]]:
    """mesh name -> the reasons it must not be decimated, over every scene.

    A mesh matters to the physics when a geom that references it (a) collides, or (b) sits
    on a body that can move and takes its inertia from its geoms (no <inertial> of its
    own -- `_explicit_inertials`). Every geom that references a mesh -- as a mesh geom or
    as a primitive FITTED from one (`type="box" mesh="X"`) -- keeps the mesh id in the
    compiled model's `geom_dataid` (measured on mujoco 3.3.0: the eleven fitted
    Panda primitives all report it), so this is read off the compiled scenes and nothing
    is hand-listed or parsed out of the XML.
    """
    import mujoco
    out: Dict[str, List[str]] = {}
    explicit = _explicit_inertials(all_xmls)
    for scene in scene_xmls:
        m = mujoco.MjModel.from_xml_path(str(scene))
        moving = _moving_bodies(m)
        for g in range(m.ngeom):
            if m.geom_dataid[g] < 0:
                continue
            mesh = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, int(m.geom_dataid[g]))
            why = ""
            if m.geom_contype[g] or m.geom_conaffinity[g]:
                why = "collides"
            else:
                b = int(m.geom_bodyid[g])
                if moving[b] and m.body_mass[b] > 0:
                    bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
                    if not bname or not explicit.get(bname, False):
                        why = f"is on the moving body {bname or b}, whose inertia comes from its geoms"
            if not why:
                continue
            kind = "mesh geom" if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH else "fitted primitive"
            gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
            note = f"{scene.name}: {kind} {gname} {why}"
            out.setdefault(mesh, [])
            if note not in out[mesh]:
                out[mesh].append(note)
    return dict(sorted(out.items()))


def compare_compiled(upstream_scene: Path, vendored_scene: Path) -> Dict:
    """Upstream's OBJ-backed model against the vendored STL-backed one, on what the
    physics reads: colliding geoms' size/pose, moving bodies' mass/inertia/CoM."""
    import mujoco
    a = mujoco.MjModel.from_xml_path(str(upstream_scene))
    b = mujoco.MjModel.from_xml_path(str(vendored_scene))
    if a.ngeom != b.ngeom or a.nbody != b.nbody:
        raise SystemExit(f"{vendored_scene.name}: compiled model shape differs from upstream's")
    coll = [g for g in range(a.ngeom) if a.geom_contype[g] or a.geom_conaffinity[g]]
    moving = [bd for bd, mv in enumerate(_moving_bodies(a)) if mv]
    diff = lambda x, y, idx: float(np.max(np.abs(np.asarray(x)[idx] - np.asarray(y)[idx]))) if idx else 0.0  # noqa: E731
    fixed = [bd for bd in range(1, a.nbody) if bd not in moving]
    return {
        "colliding_geoms": len(coll),
        "colliding_geom_size_max_abs_diff": diff(a.geom_size, b.geom_size, coll),
        "colliding_geom_pos_max_abs_diff": diff(a.geom_pos, b.geom_pos, coll),
        "colliding_geom_quat_max_abs_diff": diff(a.geom_quat, b.geom_quat, coll),
        "moving_bodies": len(moving),
        "moving_body_mass_max_abs_diff": diff(a.body_mass, b.body_mass, moving),
        "moving_body_inertia_max_abs_diff": diff(a.body_inertia, b.body_inertia, moving),
        "moving_body_ipos_max_abs_diff": diff(a.body_ipos, b.body_ipos, moving),
        "fixed_body_mass_max_abs_diff": diff(a.body_mass, b.body_mass, fixed),
    }


_MUST_BE_ZERO = ("colliding_geom_size_max_abs_diff", "colliding_geom_pos_max_abs_diff",
                 "colliding_geom_quat_max_abs_diff", "moving_body_mass_max_abs_diff",
                 "moving_body_inertia_max_abs_diff", "moving_body_ipos_max_abs_diff")


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def build(src: Path, out: Path, source_note: str) -> Dict:
    assets = src / UPSTREAM_ASSETS
    if not assets.is_dir():
        raise SystemExit(f"{assets} is not a directory; is --src an assistax checkout?")
    if out.exists():
        shutil.rmtree(out)
    (out / "meshes").mkdir(parents=True)
    prov: Dict = {
        "upstream": {"repo": UPSTREAM_REPO, "commit": UPSTREAM_COMMIT, "source": source_note,
                     "assets_dir": UPSTREAM_ASSETS},
        "rule": "every mesh OBJ -> binary STL; a mesh is decimated to max(min_triangles, "
                "keep_fraction) UNLESS a geom that references it (mesh geom or fitted "
                "primitive, any scene) collides or carries mass on a moving body (`protected`, "
                "read off the compiled models); XML copied verbatim except "
                "<mesh file=\"X.obj\"> -> X.stl; `fidelity` compares the compiled models",
        "decimation": {"tool": "pyfqmr", "min_triangles": MIN_TRIANGLES,
                       "keep_fraction": KEEP_FRACTION, "aggressiveness": AGGRESSIVENESS},
        "protected": {}, "fidelity": {}, "xml": {}, "meshes": {},
    }
    xml_out: List[Path] = []
    xml_in: List[Path] = []
    for rel in list(SCENES) + [f"{d}/{p.name}" for d in ASSET_DIRS
                               for p in sorted((assets / d).glob("*.xml"))]:
        s = assets / rel
        if not s.is_file():
            raise SystemExit(f"upstream has no {rel}")
        text = s.read_text()
        edited, n = _MESH_RE.subn(r"\1\2.stl\3", text)
        d = out / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        d.write_text(edited)
        prov["xml"][rel] = {"sha256_upstream": sha256(s), "sha256": sha256(d),
                            "mesh_refs_repointed": n}
        xml_in.append(s)
        xml_out.append(d)
    names = referenced_meshes(xml_in)
    if not names:
        raise SystemExit("no <mesh file=...obj> references found; the regex or the layout moved")
    # what the physics reads, off UPSTREAM's compiled scenes (the vendored ones do not
    # exist yet); mesh names are the file stems -- no <mesh> here carries a name attribute
    upstream_scenes = [assets / sc for sc in SCENES]
    protected = protected_meshes(upstream_scenes, xml_in)
    prov["protected"] = protected
    for name in names:
        s = assets / "meshes" / f"{name}.obj"
        if not s.is_file():
            raise SystemExit(f"upstream scene references a mesh that is not there: {s}")
        v, t = read_obj(s)
        if name in protected:
            v2, t2 = v, t
        else:
            target = max(MIN_TRIANGLES, int(KEEP_FRACTION * t.shape[0]))
            v2, t2 = decimate(v, t, target)
        d = out / "meshes" / f"{name}.stl"
        write_stl(d, v2, t2)
        prov["meshes"][name] = {
            "sha256_upstream": sha256(s), "sha256": sha256(d),
            "triangles_upstream": int(t.shape[0]), "triangles": int(t2.shape[0]),
            "bytes_upstream": s.stat().st_size, "bytes": d.stat().st_size,
            "protected": name in protected,
        }
    for sc in SCENES:
        rec = compare_compiled(assets / sc, out / sc)
        bad = [k for k in _MUST_BE_ZERO if rec[k] != 0.0]
        if bad:
            raise SystemExit(f"{sc}: the vendored model differs from upstream's on what the "
                             f"physics reads: {', '.join(f'{k}={rec[k]:g}' for k in bad)}")
        prov["fidelity"][sc] = rec
    (out / "PROVENANCE.json").write_text(json.dumps(prov, indent=1, sort_keys=True) + "\n")
    return prov


def _tree_digests(root: Path) -> Dict[str, str]:
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != "PROVENANCE.json"}


def check(src: Path, source_note: str) -> int:
    committed = json.loads((DST / "PROVENANCE.json").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        fresh = build(src, Path(tmp) / "out", source_note)
        problems = []
        for group in ("xml", "meshes"):
            for name, rec in fresh[group].items():
                have = committed.get(group, {}).get(name)
                if have is None:
                    problems.append(f"{group}/{name}: not in the committed PROVENANCE.json")
                elif have["sha256"] != rec["sha256"] or have["sha256_upstream"] != rec["sha256_upstream"]:
                    problems.append(f"{group}/{name}: digests differ from a fresh build")
            for name in committed.get(group, {}):
                if name not in fresh[group]:
                    problems.append(f"{group}/{name}: in PROVENANCE.json but a fresh build does not produce it")
        # what the physics reads: the committed manifest must carry the fresh measurement
        if committed.get("protected") != fresh["protected"]:
            problems.append("protected: the committed list differs from what the compiled models say")
        if committed.get("fidelity") != fresh["fidelity"]:
            problems.append("fidelity: the committed comparison differs from a fresh one")
        for sc, rec in committed.get("fidelity", {}).items():
            for k in _MUST_BE_ZERO:
                if rec.get(k) != 0.0:
                    problems.append(f"fidelity/{sc}: {k} = {rec.get(k)} (must be 0)")
        for name, rec in committed.get("meshes", {}).items():
            if rec.get("protected") and rec["triangles"] != rec["triangles_upstream"]:
                problems.append(f"meshes/{name}: protected yet decimated")
        # the committed BYTES against the committed manifest
        on_disk = _tree_digests(DST)
        expected = {}
        for rel, rec in committed["xml"].items():
            expected[rel] = rec["sha256"]
        for name, rec in committed["meshes"].items():
            expected[f"meshes/{name}.stl"] = rec["sha256"]
        for rel, digest in expected.items():
            if on_disk.get(rel) != digest:
                problems.append(f"{rel}: committed bytes do not match PROVENANCE.json")
        for rel in on_disk:
            if rel not in expected:
                problems.append(f"{rel}: on disk but not in PROVENANCE.json (unrecorded drift)")
    if problems:
        print("\n".join(problems))
        print(f"{len(problems)} problem(s)")
        return 1
    print(f"ok: {len(committed['xml'])} xml files and {len(committed['meshes'])} meshes match")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", type=Path, default=None,
                    help="an existing upstream checkout at UPSTREAM_COMMIT (default: fetch one)")
    ap.add_argument("--check", action="store_true", help="re-derive and compare, write nothing")
    ap.add_argument("--allow-unpinned-src", action="store_true",
                    help="accept a --src whose HEAD is not UPSTREAM_COMMIT (recorded as such)")
    args = ap.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        if args.src is not None:
            src = args.src.resolve()
            head = _head(src)
            if not head:
                note = f"--src {src} (no git metadata; commit UNVERIFIED)"
                print("warning:", note, file=sys.stderr)
            elif head != UPSTREAM_COMMIT and not args.allow_unpinned_src:
                raise SystemExit(f"--src HEAD is {head}, not the pinned {UPSTREAM_COMMIT}; "
                                 "pass --allow-unpinned-src to record it anyway")
            else:
                note = f"--src checkout at {head}"
        else:
            src = fetch_source(Path(tmp))
            note = f"fetched {UPSTREAM_COMMIT} from github.com/{UPSTREAM_REPO}"
        if args.check:
            return check(src, note)
        prov = build(src, DST, note)
    total_in = sum(r["bytes_upstream"] for r in prov["meshes"].values())
    total_out = sum(r["bytes"] for r in prov["meshes"].values())
    n_dec = sum(1 for r in prov["meshes"].values() if r["triangles"] < r["triangles_upstream"])
    print(f"wrote {len(prov['xml'])} xml files and {len(prov['meshes'])} meshes to {DST}: "
          f"{total_in / 1e6:.1f} MB of OBJ -> {total_out / 1e6:.1f} MB of STL; "
          f"{len(prov['protected'])} meshes protected, {n_dec} actually decimated")
    for name, why in prov["protected"].items():
        print(f"  protected {name}: {why[0]}" + (f" (+{len(why) - 1})" if len(why) > 1 else ""))
    for sc, rec in prov["fidelity"].items():
        print(f"  fidelity {sc}: {rec['colliding_geoms']} colliding geoms, {rec['moving_bodies']} "
              f"moving bodies, all must-be-zero diffs 0; fixed-body mass max diff "
              f"{rec['fixed_body_mass_max_abs_diff']:.3g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
