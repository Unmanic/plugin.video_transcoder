#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    plugins.nvenc.py

    Written by:               Josh.5 <jsunnex@gmail.com>
    Date:                     16 Dec 2025, (09:36 AM)

    Copyright:
        Copyright (C) 2021 Josh Sunnex

        This program is free software: you can redistribute it and/or modify it under the terms of the GNU General
        Public License as published by the Free Software Foundation, version 3.

        This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
        implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
        for more details.

        You should have received a copy of the GNU General Public License along with this program.
        If not, see <https://www.gnu.org/licenses/>.

"""

import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, Optional


logger = logging.getLogger("Unmanic.Plugin.video_transcoder")


@dataclass
class SourceStats:
    """Snapshot of the source used to build recommendations."""

    width: int
    height: int
    codec_name: Optional[str]
    bit_depth: Optional[int]
    duration: Optional[float]
    stream_bitrate: Optional[int]
    container_bitrate: Optional[int]
    filesize_bits: Optional[int]
    derived_bitrate: Optional[int]
    is_hdr: bool
    pix_fmt: Optional[str]
    fps: Optional[float]
    confidence: bool
    confidence_level: str
    confidence_notes: list


class SmartOutputTargetHelper:
    """Human-readable helper for selecting sane encoder params in Basic mode."""

    GOAL_PREFER_QUALITY = "prefer_quality"
    GOAL_BALANCED = "balanced"
    GOAL_PREFER_COMPRESSION = "prefer_compression"

    _GOALS = {GOAL_PREFER_QUALITY, GOAL_BALANCED, GOAL_PREFER_COMPRESSION}

    # This table provides a selection of factors for each goal that modifies how aggressively to cap bitrate.
    # Goal adjusts the cap headroom (Prefer Quality keeps more bitrate, Prefer Compression clamps harder) relative
    # to the source-per-pixel when scaling, to avoid inflating downscales or starving upscales when deriving
    # maxrate/bufsize rails from the source bitrate.
    _CAP_FACTORS = {
        GOAL_PREFER_QUALITY:     1.3,
        GOAL_BALANCED:           1.1,
        GOAL_PREFER_COMPRESSION: 0.9,
    }

    # This table provides per-resolution thresholds that decide when a low-bitrate source should fall back from constqp to VBR.
    # sd/hd/uhd buckets raise the bar for what counts as "already low bitrate" before forcing the safer mode.
    _LOW_BITRATE_THRESHOLDS = {
        "sd": 800_000,
        "hd": 2_500_000,
        "uhd": 6_000_000,
    }

    # This table provides lower bounds for bitrate caps by goal and resolution bucket when deriving rails.
    # Goal shifts how low we allow the cap to drop, while sd/hd/uhd buckets scale minimum expectations.
    _FLOOR_BITRATES = {
        GOAL_PREFER_QUALITY:     {"sd": 1_100_000, "hd": 3_000_000, "uhd": 7_000_000},
        GOAL_BALANCED:           {"sd": 900_000, "hd": 2_400_000, "uhd": 6_000_000},
        GOAL_PREFER_COMPRESSION: {"sd": 750_000, "hd": 2_000_000, "uhd": 5_000_000},
    }

    # This table provides CQ ladders keyed by codec -> goal -> dynamic range -> resolution bucket.
    # Codec nudges the ladder based on encoder behaviour (hevc/h264/av1); goal balances size vs fidelity;
    # HDR entries lower CQ to reduce banding risk; sd/hd/uhd buckets scale quantisers with detail expectations.
    _BASE_CQ_LADDERS = {
        "hevc": {
            GOAL_PREFER_COMPRESSION: {
                "sdr": {"sd": 27, "hd": 28, "uhd": 29},
                "hdr": {"sd": 26, "hd": 27, "uhd": 28},
            },
            GOAL_BALANCED: {
                "sdr": {"sd": 24, "hd": 25, "uhd": 26},
                "hdr": {"sd": 23, "hd": 24, "uhd": 25},
            },
        },
        "h264": {
            GOAL_PREFER_COMPRESSION: {
                "sdr": {"sd": 25, "hd": 26, "uhd": 27},
                "hdr": {"sd": 24, "hd": 25, "uhd": 26},
            },
            GOAL_BALANCED: {
                "sdr": {"sd": 22, "hd": 23, "uhd": 24},
                "hdr": {"sd": 21, "hd": 22, "uhd": 23},
            },
        },
        "av1": {
            GOAL_PREFER_COMPRESSION: {
                "sdr": {"sd": 27, "hd": 28, "uhd": 29},
                "hdr": {"sd": 26, "hd": 27, "uhd": 28},
            },
            GOAL_BALANCED: {
                "sdr": {"sd": 24, "hd": 25, "uhd": 26},
                "hdr": {"sd": 23, "hd": 24, "uhd": 25},
            },
        },
    }

    # This table provides CQ bias offsets keyed by (source_codec, target_codec).
    # Efficient source codecs (hevc/av1/vp9) get negative offsets to preserve quality; h264 -> hevc can take a small
    # positive offset for extra compression; matching codecs lean negative to reduce generational loss.
    _REENCODE_CQ_OFFSETS = {
        ("h264", "hevc"): 1,
        ("hevc", "hevc"): -1,
        ("hevc", "h264"): -2,
        ("av1", "hevc"): -1,
        ("av1", "h264"): -2,
        ("av1", "av1"): -1,
        ("hevc", "av1"): -1,
        ("h264", "h264"): 0,
        ("vp9", "hevc"): -1,
        ("vp9", "h264"): -2,
        ("vp9", "av1"): -1,
        ("vp9", "vp9"): -1,
    }

    def __init__(self, probe, max_probe_seconds: float = 2.0):
        self.probe = probe
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

    def _bit_depth_from_pix_fmt(self, pix_fmt: Optional[str], default: Optional[int] = None) -> Optional[int]:
        if not pix_fmt:
            return default
        # p010le / p016le (10/16-bit formats)
        match = re.search(r"p0(\d{2})", pix_fmt)
        if match:
            return int(match.group(1))
        # The most common pattern: yuv420p10le -> contains 10
        match = re.search(r"p(\d{2})", pix_fmt)
        if match:
            return int(match.group(1))
        return default

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

        Confidence levels:
        - high: duration present AND a usable bitrate estimate exists (derived/container/stream)
        - medium: usable bitrate exists but duration missing, OR HDR signalling partially missing, OR HDR probe failed
        - low: no usable bitrate estimate, OR HDR status is ambiguous and could affect output profile selection
        """
        probe_dict = self.probe.get_probe()
        first_video = self.probe.get_first_video_stream() or {}

        width = self._safe_int(first_video.get("width")) or self._safe_int(first_video.get("coded_width")) or 0
        height = self._safe_int(first_video.get("height")) or self._safe_int(first_video.get("coded_height")) or 0
        codec_name = first_video.get("codec_name")

        # FPS (prefer avg_frame_rate if available, fall back to r_frame_rate)
        fps = None
        rate = first_video.get("avg_frame_rate") or first_video.get("r_frame_rate")
        if rate:
            try:
                num, den = str(rate).split("/")
                fps = float(num) / float(den) if float(den) != 0 else None
            except (ValueError, ZeroDivisionError):
                fps = None

        # Duration + bitrate derivation feed rail calculations; if either is weak, confidence drops later.
        duration = self._duration_seconds(probe_dict)
        stream_bitrate, container_bitrate, filesize_bits, derived_bitrate = self._derive_bitrate(
            probe_dict, file_path, duration
        )

        confidence_notes = []
        confidence_level = "high"
        has_duration = bool(duration and duration > 0)
        has_bitrate = any(b is not None for b in (derived_bitrate, container_bitrate, stream_bitrate))

        if not has_bitrate:
            confidence_notes.append("missing_bitrate")
            confidence_level = "low"
        elif not has_duration:
            confidence_notes.append("missing_duration")
            confidence_level = "medium"

        # Informational: stream/container bitrate missing is common (esp. VP9/WebM)
        if stream_bitrate is None:
            confidence_notes.append("missing_stream_bitrate")
        if container_bitrate is None:
            confidence_notes.append("missing_container_bitrate")

        # HDR detection: trust probe first, but downgrade confidence when signalling is incomplete or ambiguous.
        is_hdr = False
        hdr_probe_failed = False
        try:
            is_hdr = bool(self.probe.is_hdr_source())
        except Exception:
            hdr_probe_failed = True
            confidence_notes.append("hdr_detection_failed")
            if confidence_level == "high":
                confidence_level = "medium"

        # Fallback HDR detection from metadata
        transfer = first_video.get("color_transfer")
        if not is_hdr and transfer in ("smpte2084", "arib-std-b67"):
            is_hdr = True

        # If HDR is detected but key tags are missing, downgrade to medium (not low) since profile/tagging may be off.
        if is_hdr:
            if not first_video.get("color_primaries") or not transfer or not first_video.get("color_space"):
                confidence_notes.append("missing_hdr_tags")
                if confidence_level == "high":
                    confidence_level = "medium"

        side_data = first_video.get("side_data_list") or []
        has_mastering = any(sd.get("side_data_type") == "Mastering display metadata" for sd in side_data)
        has_cll = any(sd.get("side_data_type") == "Content light level metadata" for sd in side_data)

        # If HDR probe failed AND metadata suggests possible HDR but is incomplete, mark ambiguous -> low.
        # (Example: 10-bit + BT.2020 primaries but transfer missing.)
        if hdr_probe_failed and not is_hdr:
            primaries = first_video.get("color_primaries")
            pix_fmt = first_video.get("pix_fmt") or ""
            looks_10bit = "p10" in pix_fmt or "p010" in pix_fmt
            looks_bt2020 = primaries == "bt2020"
            looks_bt2020cs = first_video.get("color_space") == "bt2020nc"
            if (looks_10bit or has_mastering or has_cll or looks_bt2020 or looks_bt2020cs) and not transfer:
                confidence_notes.append("hdr_ambiguous")
                confidence_level = "low"

        # Bit depth (note: ensure _bit_depth_from_pix_fmt is correct; yuv420p10le should become 10)
        pix_fmt = first_video.get("pix_fmt")
        bit_depth = self._safe_int(first_video.get("bits_per_raw_sample")) or self._bit_depth_from_pix_fmt(pix_fmt)

        confidence = confidence_level != "low"

        stats = SourceStats(
            width=width,
            height=height,
            codec_name=codec_name,
            bit_depth=bit_depth,
            duration=duration,
            stream_bitrate=stream_bitrate,
            container_bitrate=container_bitrate,
            filesize_bits=filesize_bits,
            derived_bitrate=derived_bitrate,
            is_hdr=is_hdr,
            pix_fmt=pix_fmt,
            fps=fps,
            confidence=confidence,
            confidence_level=confidence_level,
            confidence_notes=confidence_notes,
        )

        logger.info(
            "Smart output target source stats: res=%sx%s codec=%s hdr=%s bit_depth=%s "
            "bitrate_derived=%s stream_bitrate=%s container_bitrate=%s "
            "confidence=%s level=%s reasons=%s",
            stats.width,
            stats.height,
            stats.codec_name,
            stats.is_hdr,
            stats.bit_depth,
            stats.derived_bitrate,
            stats.stream_bitrate,
            stats.container_bitrate,
            stats.confidence,
            stats.confidence_level,
            stats.confidence_notes,
        )

        return stats

    def _resolution_bucket(self, width: int, height: int) -> str:
        width = max(width, 0)
        height = max(height, 0)
        pixel_area = width * height
        # Use pixel area with width guards so near-720p content does not fall into SD.
        if width >= 2400 or pixel_area >= 3_000_000:
            return "uhd"
        if width >= 1100 or pixel_area >= 700_000:
            return "hd"
        return "sd"

    def _floor_bitrate(self, goal: str, resolution_bucket: str) -> int:
        return self._FLOOR_BITRATES.get(goal, {}).get(resolution_bucket, 0)

    def _clamp(self, value: float, min_value: float, max_value: float) -> float:
        return max(min_value, min(value, max_value))

    def _target_cap(self, goal: str, source_bitrate: Optional[int], pixel_ratio: float, resolution_bucket: str, is_hdr: bool):
        """
        Guardrail: cap bitrate relative to source, scaled by pixel ratio.
        """
        if not source_bitrate:
            return None
        cap_factor = self._CAP_FACTORS.get(goal, 1.1)
        if is_hdr and goal != self.GOAL_PREFER_QUALITY:
            cap_factor += 0.05
        floor_br = self._floor_bitrate(goal, resolution_bucket)
        base_cap = source_bitrate * pixel_ratio
        same_res_cap = source_bitrate * 1.05
        upper_cap = source_bitrate if pixel_ratio < 1 else source_bitrate * 1.05
        target_cap = (base_cap if pixel_ratio < 1 else same_res_cap) * cap_factor
        min_cap = min(floor_br, upper_cap)
        return int(self._clamp(target_cap, min_cap, upper_cap))

    # --------------------------
    # Recommendation assembly
    # --------------------------
    def _base_cq_for_codec(self, target_codec: str, goal: str, is_hdr: bool, resolution_bucket: str) -> Optional[int]:
        """
        CQ ladders are per encoder/codec, with HDR slightly more conservative.
        """
        target_codec = (target_codec or "").lower()
        codec_table = self._BASE_CQ_LADDERS.get(target_codec) or self._BASE_CQ_LADDERS.get("hevc")
        goal_table = codec_table.get(goal)
        if not goal_table:
            return None
        bucket_table = goal_table["hdr" if is_hdr else "sdr"]
        return bucket_table.get(resolution_bucket)

    def _reencode_cq_offset(self, source_codec: Optional[str], target_codec: Optional[str]) -> int:
        """
        Bias CQ based on codec direction to avoid over-compressing already efficient sources.
        Negative offsets reduce CQ (more quality), positive offsets allow more compression.
        """
        src = (source_codec or "").lower()
        dst = (target_codec or "").lower()
        return self._REENCODE_CQ_OFFSETS.get((src, dst), 0)

    def recommend_params(self, goal: str, source_stats: SourceStats, target_filters: Dict, target_codec: str, target_encoder: str):
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
        resolution_bucket = self._resolution_bucket(target_width, target_height)
        source_bitrate = source_stats.derived_bitrate
        target_codec = (target_codec or "").lower()
        target_encoder = (target_encoder or "").lower()

        logger.info(
            "Smart output target recommend_params input: goal=%s target_codec=%s target_encoder=%s target=%sx%s source_bitrate=%s pixel_ratio=%.3f",
            goal,
            target_codec,
            target_encoder,
            target_width,
            target_height,
            source_bitrate,
            pixel_ratio,
        )

        # Guardrail cap:
        #   Scale relative to source bitrate and pixel ratio (downscales should not exceed source-per-pixel bitrate).
        target_cap = self._target_cap(goal, source_bitrate, pixel_ratio, resolution_bucket, source_stats.is_hdr)
        downgraded_to_vbr = False

        # Rate control mode:
        #   Default Prefer Quality to constqp (no bitrate cap), but switch to VBR if downscaling a low-bitrate source.
        #   Why: constqp on already-starved inputs can blow up bitrate when scaling down; CQ-VBR with caps protects size.
        rc_mode = "constqp" if goal == self.GOAL_PREFER_QUALITY else "vbr"

        # Low bitrate + downscale guardrail:
        #   If the source is already low bitrate and we’re downscaling, Prefer Quality falls back to CQ-VBR to avoid
        #   runaway bitrate from constqp while still respecting a cap based on the source and pixel ratio.
        low_bitrate_threshold = self._LOW_BITRATE_THRESHOLDS.get(resolution_bucket, 0)
        if goal == self.GOAL_PREFER_QUALITY and pixel_ratio < 1 and source_bitrate and source_bitrate <= low_bitrate_threshold:
            rc_mode = "vbr"
            downgraded_to_vbr = True

        # HDR handling guardrails:
        #   - hdr_output_limited: HDR source but target codec is H.264 (no 10-bit), so treat as banding-prone.
        #   - hdr_8bit_source: source is HDR but only 8-bit; also banding-prone.
        hdr_output_limited = source_stats.is_hdr and target_codec == "h264"
        hdr_8bit_source = source_stats.is_hdr and source_stats.bit_depth and source_stats.bit_depth <= 8

        # Base params per goal:
        #   Prefer Quality uses constqp/QP for maximum preservation.
        #   Balanced/Prefer Compression use CQ-VBR with codec-specific CQ ladders.
        #   Keep this split so quality-first avoids VBR variability while compression goals can lean on maxrate/bufsize guardrails.
        if goal == self.GOAL_PREFER_QUALITY:
            qp = 19 if not source_stats.is_hdr else 17
            cq = None
            lookahead = 20
        else:
            qp = None
            cq = self._base_cq_for_codec(target_codec, goal, source_stats.is_hdr, resolution_bucket)
            if not cq:
                cq = 24 if not source_stats.is_hdr else 22
                if goal == self.GOAL_PREFER_COMPRESSION:
                    cq = 30 if not source_stats.is_hdr else 28
            lookahead = 12

        # AQ defaults:
        #   Leave spatial AQ on for all goals.
        #   Use temporal AQ only when running VBR (constqp disables it).
        enable_aq = True
        aq_strength = 8
        temporal_aq = rc_mode == "vbr"

        # HDR 8-bit handling (CQ/QP side):
        #   Be conservative and lower CQ/QP slightly.
        #   HDR in 8-bit is much more prone to banding.
        #   When you push compression harder, the first “bad-looking” failure
        #   mode is often banding rather than blockiness.
        hdr_banding_sensitive = source_stats.is_hdr and (hdr_output_limited or hdr_8bit_source)
        if hdr_banding_sensitive:
            if cq is not None:
                cq = max(0, cq - 2)
            if rc_mode == "constqp" and qp is not None:
                qp = max(0, qp - 1)

        # Build bitrate params for VBR modes:
        #   Cap maxrate/bufsize using derived guardrails.
        maxrate = None
        bufsize = None
        if rc_mode == "vbr" and target_cap:
            maxrate = int(target_cap)
            bufsize_factor = 2.0
            bufsize = int(target_cap * bufsize_factor)
            # Derive cq if we downgraded from constqp
            if downgraded_to_vbr and not cq:
                cq = 22 if not source_stats.is_hdr else 20
            elif not cq:
                cq = 26 if goal == self.GOAL_PREFER_COMPRESSION else 23
        elif rc_mode == "vbr" and not cq:
            cq = 26 if goal == self.GOAL_PREFER_COMPRESSION else 23

        # HDR 8-bit handling (bitrate rails):
        #   Nudge caps upward slightly to soften banding risk on sensitive HDR outputs.
        if hdr_banding_sensitive:
            if maxrate:
                maxrate = int(maxrate * 1.05)
            if bufsize:
                bufsize = int(bufsize * 1.05)

        # Re-encode penalty/bonus per codec direction:
        #   Aggressive CQ on already-efficient sources (HEVC/AV1) compounds artifacts, so bias toward lower CQ.
        #   Conversely, H.264 -> HEVC can tolerate a slightly higher CQ for similar perceptual results.
        cq_offset = self._reencode_cq_offset(source_stats.codec_name, target_codec)
        if cq is not None:
            cq = max(0, cq + cq_offset)
        if rc_mode == "constqp" and qp is not None and cq_offset:
            qp = max(0, qp - cq_offset)  # lower qp for negative offsets = higher quality

        # Confidence tracking and adjustments:
        #   When key probe stats are missing, ease quantizers and drop temporal AQ.
        confidence_level = source_stats.confidence_level
        confidence = confidence_level != "low"
        confidence_notes = list(source_stats.confidence_notes)
        if hdr_output_limited:
            confidence = False
            confidence_level = "low"
            confidence_notes.append("hdr_forced_8bit_output")
            logger.info("HDR source detected but target codec lacks 10-bit; applying conservative settings.")
        elif hdr_8bit_source:
            confidence = False
            confidence_level = "low"
            confidence_notes.append("hdr_8bit_source")
        if confidence_level != "high":
            logger.info("Smart output target low confidence: level=%s reasons=%s", confidence_level, confidence_notes)
        if not confidence:
            if cq is not None:
                cq = max(16, cq - 2)
            if qp is not None:
                qp = max(14, qp - 2)
            temporal_aq = False

        recommendation = {
            "goal": goal,
            "rc_mode": rc_mode,
            "qp": qp,
            "cq": cq,
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
            "confidence_level": confidence_level,
            "confidence_notes": confidence_notes,
            "target_resolution": {"width": target_width, "height": target_height},
            "source_resolution": {"width": source_stats.width, "height": source_stats.height},
        }
        logger.info(
            "Smart output target recommendation: goal=%s rc=%s qp=%s cq=%s preset_lookahead=%s aq=%s temporal_aq=%s maxrate=%s bufsize=%s confidence=%s confidence_notes=%s",
            recommendation.get("goal"),
            recommendation.get("rc_mode"),
            recommendation.get("qp"),
            recommendation.get("cq"),
            recommendation.get("lookahead"),
            recommendation.get("enable_aq"),
            recommendation.get("temporal_aq"),
            recommendation.get("maxrate"),
            recommendation.get("bufsize"),
            recommendation.get("confidence"),
            recommendation.get("confidence_notes"),
        )

        return recommendation
