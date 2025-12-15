#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional


logger = logging.getLogger("Unmanic.Plugin.video_transcoder")


@dataclass
class SourceStats:
    """Snapshot of the source used to build recommendations."""

    width: int
    height: int
    duration: Optional[float]
    stream_bitrate: Optional[int]
    container_bitrate: Optional[int]
    filesize_bits: Optional[int]
    derived_bitrate: Optional[int]
    is_hdr: bool
    pix_fmt: Optional[str]
    fps: Optional[float]
    confidence: bool
    confidence_reasons: list


class SmartOutputTargetHelper:
    """Human-readable helper for selecting sane encoder params in Basic mode."""

    GOAL_PREFER_QUALITY = "prefer_quality"
    GOAL_BALANCED = "balanced"
    GOAL_PREFER_COMPRESSION = "prefer_compression"
    _GOALS = {GOAL_PREFER_QUALITY, GOAL_BALANCED, GOAL_PREFER_COMPRESSION}

    # How aggressively to cap bitrate relative to the source when downscaling.
    _CAP_FACTORS = {
        GOAL_PREFER_QUALITY:     1.3,
        GOAL_BALANCED:           1.1,
        GOAL_PREFER_COMPRESSION: 0.9,
    }

    _LOW_BITRATE_THRESHOLDS = {
        "sd": 800_000,
        "hd": 2_500_000,
        "uhd": 6_000_000,
    }

    # Lower bound for bitrate caps by goal + resolution bucket.
    _FLOOR_BITRATES = {
        GOAL_PREFER_QUALITY:     {"sd": 1_100_000, "hd": 3_000_000, "uhd": 7_000_000},
        GOAL_BALANCED:           {"sd": 900_000, "hd": 2_400_000, "uhd": 6_000_000},
        GOAL_PREFER_COMPRESSION: {"sd": 750_000, "hd": 2_000_000, "uhd": 5_000_000},
    }

    def __init__(self, probe, logger_override=None, max_probe_seconds: float = 2.0):
        self.probe = probe
        self.logger = logger_override or logger
        self.max_probe_seconds = max_probe_seconds

    # --------------------------
    # Parsing helpers
    # --------------------------
    def _safe_int(self, value: Optional[str]) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _safe_float(self, value: Optional[str]) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _duration_seconds(self, probe_dict: Dict) -> Optional[float]:
        fmt = probe_dict.get("format") or {}
        dur = self._safe_float(fmt.get("duration"))
        if dur:
            return dur

        streams = probe_dict.get("streams") or []
        for st in streams:
            dur = self._safe_float(st.get("duration"))
            if dur:
                return dur

        tags = fmt.get("tags") or {}
        tag_dur = tags.get("DURATION")
        if isinstance(tag_dur, str) and ":" in tag_dur:
            try:
                h, m, s = tag_dur.split(":")
                return float(h) * 3600 + float(m) * 60 + float(s)
            except (ValueError, TypeError):
                return None
        return None

    def _derive_bitrate(self, probe_dict: Dict, file_path: Optional[str], duration: Optional[float]):
        """Pull bitrates from stream/container; fall back to size/duration."""
        video_stream = next((st for st in probe_dict.get("streams", []) if st.get("codec_type") == "video"), None)
        stream_bitrate = self._safe_int(video_stream.get("bit_rate")) if video_stream else None
        container_bitrate = self._safe_int((probe_dict.get("format") or {}).get("bit_rate"))

        filesize_bits = None
        if file_path and os.path.exists(file_path):
            try:
                filesize_bits = os.path.getsize(file_path) * 8
            except OSError:
                filesize_bits = None

        derived = None
        if stream_bitrate:
            derived = stream_bitrate
        elif container_bitrate:
            derived = container_bitrate
        elif filesize_bits and duration and duration > 0:
            derived = int(filesize_bits / duration)

        return stream_bitrate, container_bitrate, filesize_bits, derived

    def collect_source_stats(self, file_path: Optional[str] = None) -> SourceStats:
        """
        Summarise the source into a simple struct (resolution, bitrate, HDR, confidence).
        """
        probe_dict = self.probe.get_probe()
        first_video = self.probe.get_first_video_stream() or {}

        width = self._safe_int(first_video.get("width")) or self._safe_int(first_video.get("coded_width")) or 0
        height = self._safe_int(first_video.get("height")) or self._safe_int(first_video.get("coded_height")) or 0

        fps = None
        if first_video.get("r_frame_rate"):
            num, den = str(first_video.get("r_frame_rate")).split("/")
            try:
                fps = float(num) / float(den)
            except (ValueError, ZeroDivisionError):
                fps = None

        duration = self._duration_seconds(probe_dict)
        stream_bitrate, container_bitrate, filesize_bits, derived = self._derive_bitrate(probe_dict, file_path, duration)

        confidence_reasons = []
        if not stream_bitrate:
            confidence_reasons.append("missing_stream_bitrate")
        if not duration:
            confidence_reasons.append("missing_duration")

        is_hdr = False
        try:
            is_hdr = self.probe.is_hdr_source()
        except Exception:
            confidence_reasons.append("hdr_detection_failed")

        if is_hdr and (not first_video.get("color_primaries") or not first_video.get("color_transfer") or not first_video.get("color_space")):
            confidence_reasons.append("missing_hdr_tags")

        pix_fmt = first_video.get("pix_fmt")
        confidence = len(confidence_reasons) == 0

        return SourceStats(
            width=width,
            height=height,
            duration=duration,
            stream_bitrate=stream_bitrate,
            container_bitrate=container_bitrate,
            filesize_bits=filesize_bits,
            derived_bitrate=derived,
            is_hdr=is_hdr,
            pix_fmt=pix_fmt,
            fps=fps,
            confidence=confidence,
            confidence_reasons=confidence_reasons,
        )

    def _resolution_bucket(self, width: int, height: int) -> str:
        if width <= 1280 and height <= 720:
            return "sd"
        if width <= 1920 and height <= 1088:
            return "hd"
        return "uhd"

    def _floor_bitrate(self, goal: str, bucket: str) -> int:
        return self._FLOOR_BITRATES.get(goal, {}).get(bucket, 0)

    def _clamp(self, value: float, min_value: float, max_value: float) -> float:
        return max(min_value, min(value, max_value))

    def _target_cap(self, goal: str, source_bitrate: Optional[int], pixel_ratio: float, bucket: str):
        """
        Guardrail: cap bitrate relative to source, scaled by pixel ratio.
        """
        if not source_bitrate:
            return None
        cap_factor = self._CAP_FACTORS.get(goal, 1.1)
        floor_br = self._floor_bitrate(goal, bucket)
        base_cap = source_bitrate * pixel_ratio
        same_res_cap = source_bitrate * 1.05
        upper_cap = source_bitrate if pixel_ratio < 1 else source_bitrate * 1.05
        target_cap = (base_cap if pixel_ratio < 1 else same_res_cap) * cap_factor
        min_cap = min(floor_br, upper_cap)
        return int(self._clamp(target_cap, min_cap, upper_cap))

    # --------------------------
    # Recommendation assembly
    # --------------------------
    def recommend_params(self, goal: str, source_stats: SourceStats, target_filters: Dict):
        """
        Build a minimal, easy-to-tune recommendation dict.

        Steps:
            1) Normalise goal/targets.
            2) Compute caps relative to source + scale.
            3) Pick a baseline profile per goal.
            4) Apply guardrails (low bitrate downscale -> VBR).
            5) Tweak AQ/lookahead when confidence is low.
        """
        goal = goal or self.GOAL_BALANCED
        goal = goal if goal in self._GOALS else self.GOAL_BALANCED

        target_width = int(target_filters.get("target_width") or source_stats.width or 0)
        target_height = int(target_filters.get("target_height") or source_stats.height or 0)
        source_width = max(source_stats.width, 1)
        source_height = max(source_stats.height, 1)

        pixel_ratio = (target_width * target_height) / float(source_width * source_height)
        bucket = self._resolution_bucket(target_width, target_height)
        source_bitrate = source_stats.derived_bitrate

        target_cap = self._target_cap(goal, source_bitrate, pixel_ratio, bucket)
        downgraded_to_vbr = False
        rc_mode = "constqp" if goal == self.GOAL_PREFER_QUALITY else "vbr"

        low_bitrate_threshold = self._LOW_BITRATE_THRESHOLDS.get(bucket, 0)
        if goal == self.GOAL_PREFER_QUALITY and pixel_ratio < 1 and source_bitrate and source_bitrate <= low_bitrate_threshold:
            rc_mode = "vbr"
            downgraded_to_vbr = True

        # Base params per goal
        if goal == self.GOAL_PREFER_QUALITY:
            qp = 19 if not source_stats.is_hdr else 17
            cq = None
            preset = "p7"
            lookahead = 20
        elif goal == self.GOAL_PREFER_COMPRESSION:
            qp = None
            cq = 30 if not source_stats.is_hdr else 28
            preset = "p6"
            lookahead = 8
        else:
            qp = None
            cq = 24 if not source_stats.is_hdr else 22
            preset = "p6"
            lookahead = 12

        enable_aq = True
        aq_strength = 8
        temporal_aq = goal != self.GOAL_PREFER_QUALITY

        # Build bitrate params for VBR modes
        maxrate = None
        bufsize = None
        if rc_mode == "vbr" and target_cap:
            maxrate = int(target_cap)
            bufsize_factor = 2.0 if goal != self.GOAL_PREFER_COMPRESSION else 1.6
            bufsize = int(target_cap * bufsize_factor)
            # Derive cq if we downgraded from constqp
            if downgraded_to_vbr and not cq:
                cq = 22 if not source_stats.is_hdr else 20
            elif not cq:
                cq = 26 if goal == self.GOAL_PREFER_COMPRESSION else 23
        elif rc_mode == "vbr" and not cq:
            cq = 26 if goal == self.GOAL_PREFER_COMPRESSION else 23

        # Confidence adjustments
        confidence = source_stats.confidence
        if not confidence:
            if cq is not None:
                cq = max(16, cq - 2)
            if qp is not None:
                qp = max(14, qp - 2)
            enable_aq = False
            temporal_aq = False
            lookahead = max(0, int(lookahead / 2))

        recommendation = {
            "goal": goal,
            "rc_mode": rc_mode,
            "qp": qp,
            "cq": cq,
            "preset": preset,
            "lookahead": lookahead,
            "enable_aq": enable_aq,
            "aq_strength": aq_strength,
            "temporal_aq": temporal_aq,
            "maxrate": maxrate,
            "bufsize": bufsize,
            "target_cap": target_cap,
            "pixel_ratio": pixel_ratio,
            "downgraded_to_vbr": downgraded_to_vbr,
            "confidence": confidence,
            "confidence_reasons": source_stats.confidence_reasons,
            "target_resolution": {"width": target_width, "height": target_height},
            "source_resolution": {"width": source_stats.width, "height": source_stats.height},
        }

        return recommendation
