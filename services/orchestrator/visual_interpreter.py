"""
services/orchestrator/visual_interpreter.py

Translates numeric model outputs (uncertainty, Grad-CAM, spatial features)
into plain-English clinical flags for the CoT prompt and qa_agent.

Runs inside the orchestrator container -- must NOT import from
services.knowledge or services.vision.
"""


def interpret_visual_features(
    bottleneck: dict,
    texture: dict,
    uncertainty: dict,
    gradcam_overlap: dict,
    spatial: dict,
    organ: str,
) -> dict:
    """
    Translate numeric features into clinical flags.

    Args:
        bottleneck:      dict from ModelOutput.bottleneck_enriched (reserved, not used)
        texture:         dict from ModelOutput.texture_features (reserved, not used)
        uncertainty:     dict from ModelOutput.uncertainty
        gradcam_overlap: dict from ModelOutput.gradcam_mask_overlap
        spatial:         dict from OrchestratorState["spatial"] (SpatialDerived)
        organ:           "breast" | "thyroid"

    Returns:
        {
            "clinical_flags": list[str],
            "risk_modifier":  int,
        }
    """
    flags: list = []
    risk_modifier: int = 0

    def _num(d: dict, key: str, default: float) -> float:
        """Returns default when the key is missing or explicitly None/NaN."""
        val = d.get(key, default)
        if val is None:
            return default
        try:
            val = float(val)
        except (TypeError, ValueError):
            return default
        return default if val != val else val  # val != val is True only for NaN

    # Texture flags (internal_heterogeneity, lesion_background_contrast) are
    # disabled pending a more grounded proxy metric -- args kept for API compat.
    # hetero = _num(texture, "internal_heterogeneity", 0.0)
    # contrast = _num(texture, "lesion_background_contrast", 0.0)
    #
    # if hetero > 0.4:
    #     flags.append("heterogeneous internal texture (suspicious)")
    #     risk_modifier += 1
    # elif hetero < 0.15:
    #     flags.append("homogeneous internal texture (favors benign)")
    #
    # if contrast < 0.1:
    #     flags.append("low lesion-background contrast (poorly defined lesion)")

    # Uncertainty flags (predictive_entropy, max_class_std) are disabled
    # pending a more grounded proxy metric -- arg kept for API compat.
    # pred_entropy = _num(uncertainty, "predictive_entropy", 0.0)
    # unc_per_class = uncertainty.get("uncertainty") or []
    # max_unc = max(unc_per_class) if unc_per_class else 0.0
    #
    # if pred_entropy > 0.5 or max_unc > 0.15:
    #     flags.append(
    #         f"high model uncertainty (entropy={pred_entropy:.3f}, "
    #         f"max_class_std={max_unc:.3f}) -- findings require radiologist review"
    #     )

    # Grad-CAM overlap flags (iou) are disabled pending a more grounded
    # proxy metric -- arg kept for API compat.
    # iou = _num(gradcam_overlap, "iou", 1.0)
    #
    # if iou < 0.3:
    #     flags.append(
    #         f"attention-segmentation mismatch (IoU={iou:.2f}) -- model's "
    #         "classification focus differs significantly from segmented region"
    #     )
    #     risk_modifier += 2
    # elif iou < 0.5:
    #     flags.append(
    #         f"partial attention-segmentation overlap (IoU={iou:.2f})"
    #     )
    #     risk_modifier += 1

    ar = _num(spatial, "aspect_ratio", 1.0)
    circ = _num(spatial, "circularity", 1.0)

    if organ == "breast":
        if ar < 0.8:
            flags.append(
                f"taller-than-wide (aspect_ratio={ar:.3f} < 0.8, "
                "suspicious per BI-RADS)"
            )
            risk_modifier += 1
    elif organ == "thyroid":
        if ar < 1.0:
            flags.append(
                f"taller-than-wide (aspect_ratio={ar:.3f} < 1.0, "
                "suspicious per TI-RADS)"
            )
            risk_modifier += 1

    if circ < 0.5:
        flags.append(
            f"irregular margin (circularity={circ:.3f} < 0.5, "
            "suspicious for malignancy)"
        )
        risk_modifier += 1

    # Bottleneck flags (center_periphery_ratio, spatial_entropy) are disabled
    # pending a more grounded proxy metric -- arg kept for API compat.
    # cpr = _num(bottleneck, "center_periphery_ratio", 1.0)
    # entropy = _num(bottleneck, "spatial_entropy", 0.0)
    #
    # if cpr < 0.8:
    #     flags.append(
    #         f"margin-focused attention (center_periphery_ratio={cpr:.3f} < 0.8)"
    #     )
    # if entropy > 3.0:
    #     flags.append(
    #         f"diffuse model attention (spatial_entropy={entropy:.3f}) -- "
    #         "model is uncertain about lesion location"
    #     )

    return {
        "clinical_flags": flags,
        "risk_modifier": risk_modifier,
    }