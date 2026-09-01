#!/usr/bin/env python
# Release-readiness validator for the KINEWORLD / KineOne-WM filing package.
# Runs on CPU, no GPU / credentials / network. Verifies that every artifact the
# reviewer will open is present, internally cross-referenced, self-contained, and
# that the shippable Python scripts at least compile.
#
# Usage:  python validate_package.py
# Exit 0 = package ready to send;  exit 1 = blocking issue found.
import os, re, sys, py_compile

ROOT = os.path.dirname(os.path.abspath(__file__))

HTML = [
    "kineworld_capability_deck.html",
    "counterfactual_interactive.html",
    "kineworld_index.html",
    "bench_report.html",
]
MD = ["PITCH_CN.md", "MILESTONES_CN.md", "README.md", "BENCH_GPU.md"]
PY = [
    "build_deck.py", "counterfactual_interactive.py", "bench_gpu_launcher.py",
    "prep_bench_data.py", "posttrain.py", "real_feature_posttrain.py",
    "real_feature_smoke.py",
]

problems = []


def ok(msg):
    print(f"  ✓ {msg}")


def bad(msg):
    print(f"  ✗ {msg}")
    problems.append(msg)


# ----------------------------------------------------------------- 1) presence
print("[1/4] 产物存在性")
all_files = HTML + MD + PY
for f in all_files:
    p = os.path.join(ROOT, f)
    if not os.path.isfile(p):
        bad(f"缺失: {f}")
    else:
        sz = os.path.getsize(p)
        if sz < 64:
            bad(f"空文件/过小: {f} ({sz}B)")
        else:
            ok(f"{f}  ({sz:,}B)")


# --------------------------------------------------- 2) cross-reference integrity
print("[2/4] 交叉引用完整性（href / src 指向本地文件）")
ref_re = re.compile(r'(?:href|src)\s*=\s*"([^"]+)"')
for f in HTML:
    p = os.path.join(ROOT, f)
    if not os.path.isfile(p):
        continue
    txt = open(p, encoding="utf-8").read()
    for ref in ref_re.findall(txt):
        if ref.startswith(("http://", "https://", "//", "mailto:")):
            continue  # external link, not our concern
        if ref.startswith("#"):
            continue  # in-page anchor
        target = os.path.normpath(os.path.join(ROOT, ref))
        if not os.path.isfile(target):
            bad(f"{f} 引用了不存在的本地文件: {ref}")
        else:
            ok(f"{f} → {ref}")


# ----------------------------------------------------- 3) HTML self-containment
print("[3/4] HTML 自包含性（无外部 http(s) 资源依赖）")
ext_res = re.compile(r'(?:src|href)\s*=\s*"((?:https?:)?//[^"]+)"')
for f in HTML:
    p = os.path.join(ROOT, f)
    if not os.path.isfile(p):
        continue
    txt = open(p, encoding="utf-8").read()
    external = [m for m in ext_res.findall(txt)]
    # allow only harmless anchors / protocol-free; flag real remote assets
    remote = [e for e in external if e.startswith("http")]
    if remote:
        bad(f"{f} 含外部远程资源（离线打开会碎）: {remote}")
    else:
        ok(f"{f}  无外部远程资源依赖")


# --------------------------------------------------------- 4) python compiles
print("[4/4] Python 脚本可编译（可发布性）")
for f in PY:
    p = os.path.join(ROOT, f)
    if not os.path.isfile(p):
        continue
    try:
        py_compile.compile(p, doraise=True)
        ok(f"{f}  编译通过")
    except py_compile.PyCompileError as e:
        bad(f"{f}  编译失败: {e}")


# --------------------------------------------------------------------- summary
print("\n" + "=" * 56)
if problems:
    print(f"申报包未就绪：发现 {len(problems)} 处问题")
    for p in problems:
        print(f"  - {p}")
    sys.exit(1)
else:
    print("申报包就绪 ✓  全部产物存在、引用不断链、HTML 自包含、脚本可编译")
    print("可整体发送评审；唯余 GPU 真实评测数字（用户设备侧）与站点 DNS 上线。")
    sys.exit(0)
