"""Export YOLO .pt and ZipDepth .onnx to TensorRT engines in weights/.

Default (--stage all) runs each export in its own process so the YOLO
builder exits and frees Jetson memory before ZipDepth starts.

Usage:
    python ./utils/export_weights_trt.py
    python ./utils/export_weights_trt.py --stage yolo
    python ./utils/export_weights_trt.py --stage zipdepth
"""

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "weights")
YOLO_PT = os.path.join(WEIGHTS, "best_yolo26_drone_bird_aircraft_junho_2026.pt")
YOLO_ENGINE = os.path.join(WEIGHTS, "best_yolo26_drone_bird_aircraft_junho_2026.engine")
ZIP_ONNX = os.path.join(WEIGHTS, "zipdepth_base_384x384.onnx")
ZIP_TRT = os.path.join(WEIGHTS, "zipdepth_base_384x384_fp16.trt")
ZIP_INPUT_SHAPE = (1, 3, 384, 384)
WORKSPACE_BYTES = 1 << 30


def export_yolo():
    from ultralytics import YOLO

    model = YOLO(YOLO_PT)
    model.export(
        format="engine",
        device=0,
        half=True,
        batch=4,
        imgsz=640,
        dynamic=True,
    )
    print("Exportação YOLO para TensorRT (.engine) concluída!")
    print(f"Engine em {WEIGHTS}")


def _set_static_profile(builder, network, config):
    """Pin dynamic ONNX inputs to the ZipDepth shape (1, 3, 384, 384)."""
    profile = builder.create_optimization_profile()
    needs_profile = False
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        shape = tuple(tensor.shape)
        if any(dim < 0 for dim in shape):
            needs_profile = True
            profile.set_shape(tensor.name, ZIP_INPUT_SHAPE, ZIP_INPUT_SHAPE, ZIP_INPUT_SHAPE)
    if needs_profile:
        config.add_optimization_profile(profile)


def export_zipdepth():
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(ZIP_ONNX, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError(f"Falha ao parsear ONNX: {ZIP_ONNX}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_BYTES)
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        print("FP16 rápido indisponível; engine será FP32")

    _set_static_profile(builder, network, config)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT não gerou o engine do ZipDepth")

    with open(ZIP_TRT, "wb") as f:
        f.write(serialized)
    print(f"Exportação ZipDepth para TensorRT concluída: {ZIP_TRT}")


def _finish_stage():
    """Skip interpreter shutdown. TensorRT/PyTorch free() aborts (SIGABRT) on exit."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def _run_stage(stage, artifact):
    script = os.path.abspath(__file__)
    result = subprocess.run([sys.executable, script, "--stage", stage])
    if result.returncode == 0:
        return
    if result.returncode < 0 and os.path.isfile(artifact) and os.path.getsize(artifact) > 0:
        print(
            f"Etapa {stage} encerrou com sinal {-result.returncode} "
            f"depois de gravar {artifact}. Seguindo."
        )
        return
    raise subprocess.CalledProcessError(result.returncode, result.args)


def main():
    parser = argparse.ArgumentParser(description="Export YOLO and ZipDepth weights to TensorRT")
    parser.add_argument(
        "--stage",
        choices=("all", "yolo", "zipdepth"),
        default="all",
        help="all runs YOLO then ZipDepth in separate processes",
    )
    args = parser.parse_args()

    if args.stage == "all":
        _run_stage("yolo", YOLO_ENGINE)
        _run_stage("zipdepth", ZIP_TRT)
        return

    if args.stage == "yolo":
        export_yolo()
        _finish_stage()

    export_zipdepth()
    _finish_stage()


if __name__ == "__main__":
    main()
