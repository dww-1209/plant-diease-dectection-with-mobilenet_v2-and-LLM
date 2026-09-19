"""MobileNetV2 推理模型封装。

把"加载权重 → 图片预处理 → 前向 → 类别映射"这一整条链路封装在
``InferenceModel`` 里，构造一次、复用多次。Web 服务启动时由 ``create_app``
持有一个实例存在 ``app.config["INFERENCE_MODEL"]`` 中，所有请求共享。
"""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torchvision import models, transforms

from plant_disease.data.class_map import ClassInfo, load_class_map, lookup_class
from plant_disease.errors import InferenceError

logger = logging.getLogger(__name__)

NUM_CLASSES = 61
IMG_SIZE = 224


def _select_device() -> torch.device:
    """按 ``cuda → mps → cpu`` 的优先级挑可用设备。

    Mac M 系列没 CUDA 但有 MPS（Metal），不挑出来会回落到 CPU 慢一个数量级。
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class InferenceModel:
    """加载 MobileNetV2 权重并提供 ``predict()`` 接口。

    构造时会一次性完成：选设备 → 建网络结构 → 加载权重 → 加载类别映射 →
    切到 eval 模式。

    Raises:
        InferenceError: 权重文件不存在 / 解码失败时抛出。Web 层会捕获并返回
            503（启动期）或 500（请求期）。
    """

    def __init__(
        self,
        weights_path: Path,
        classes_txt: Path,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        self.device = _select_device()
        logger.info("inference device: %s", self.device)

        # ImageNet 上预训练的均值/方差；训练时用了同一组，推理保持一致才正确
        self.transform = transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.CenterCrop(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.class_info: list[ClassInfo] = load_class_map(classes_txt)
        self.num_classes = num_classes

        self.model = self._build_model(num_classes)
        self._load_weights(weights_path)
        self.model.to(self.device)
        self.model.eval()

    def _build_model(self, num_classes: int) -> nn.Module:
        """构造未训练的 MobileNetV2 骨架，仅替换分类头到 ``num_classes`` 个类别。"""
        model = models.mobilenet_v2()
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.last_channel, num_classes),
        )
        return model

    def _load_weights(self, weights_path: Path) -> None:
        """从本地 .pth 文件加载权重，缺/多键位仅记录日志、不抛异常。

        ``weights_only=True`` 是 PyTorch 2.4+ 的安全加载选项；老版本不支持
        会抛 ``TypeError``，这里 fallback 到不带该参数的旧调用以兼容。
        """
        if not weights_path.exists():
            raise InferenceError(f"权重文件未找到：{weights_path}")
        try:
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        except TypeError:
            state_dict = torch.load(weights_path, map_location=self.device)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            logger.info(
                "weights partial match: missing=%d, unexpected=%d", len(missing), len(unexpected)
            )

    def _prepare_image(self, file_bytes: bytes) -> torch.Tensor:
        """字节流 → PIL → Tensor，并加上 batch 维度。解码失败抛 InferenceError。"""
        try:
            img = Image.open(BytesIO(file_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise InferenceError(f"无法解码图片：{exc}") from exc
        return torch.unsqueeze(self.transform(img), dim=0)

    def predict(self, file_bytes: bytes) -> dict:
        """对一张图片做一次推理。

        Args:
            file_bytes: 原始图片字节，比如 Flask 端 ``request.files["image"].read()``

        Returns:
            形如::

                {
                    "class_id": 5,
                    "probability": 0.87,
                    "plant_class": "玉米",
                    "health_status": "患病",
                    "disease_name": "玉米灰斑病",
                    "disease_degree": "一般",
                }

        Raises:
            InferenceError: 图片解码失败、前向推理异常等。
        """
        tensor = self._prepare_image(file_bytes).to(self.device)
        try:
            with torch.no_grad():
                logits = torch.squeeze(self.model(tensor)).cpu()
                probs = torch.softmax(logits, dim=0)
                cls_idx = int(torch.argmax(probs).item())
                prob = float(probs[cls_idx].item())
        except Exception as exc:  # noqa: BLE001
            # 前向阶段我们也想兜住所有异常并标准化成 InferenceError，BLE001 是
            # ruff "Blind Except" 警告，这里属于受控的故意行为。
            raise InferenceError(f"前向推理失败：{exc}") from exc

        info = lookup_class(self.class_info, cls_idx)
        return {
            "class_id": cls_idx,
            "probability": prob,
            "plant_class": info.plant,
            "health_status": info.health_status,
            "disease_name": info.disease_name,
            "disease_degree": info.disease_degree,
        }
