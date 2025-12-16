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

    QUALITY_CONST = "const_quality"
    QUALITY_CAPPED = "capped_quality"
    QUALITY_TARGET = "target_bitrate"

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
    _BASE_QUALITY_LADDERS = {
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
        if bit_depth is None and pix_fmt:
            bit_depth = 8

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

    def _target_cap(
        self,
        goal: str,
        source_bitrate: Optional[int],
        pixel_ratio: float,
        is_hdr: bool,
        apply_floor: bool,
        floor_br: int,
    ):
        """
        Guardrail: cap bitrate relative to source, scaled by pixel ratio.
        """
        if not source_bitrate:
            return int(floor_br) if apply_floor and floor_br else None
        cap_factor = self._CAP_FACTORS.get(goal, 1.1)
        if is_hdr and goal != self.GOAL_PREFER_QUALITY:
            cap_factor += 0.05
        base_cap = source_bitrate * pixel_ratio
        same_res_cap = source_bitrate * 1.05
        upper_cap = source_bitrate if pixel_ratio < 1 else source_bitrate * 1.05
        target_cap = (base_cap if pixel_ratio < 1 else same_res_cap) * cap_factor
        candidate = target_cap
        if apply_floor and floor_br:
            candidate = max(candidate, floor_br)
        return int(self._clamp(candidate, 0, upper_cap))

    # --------------------------
    # Recommendation assembly
    # --------------------------
    def _base_quality_index(self, target_codec: str, goal: str, is_hdr: bool, resolution_bucket: str) -> Optional[int]:
        """
        Quality ladders are per codec, with HDR slightly more conservative.
        """
        target_codec = (target_codec or "").lower()
        codec_table = self._BASE_QUALITY_LADDERS.get(target_codec) or self._BASE_QUALITY_LADDERS.get("hevc")
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

    def recommend_params(
        self,
        goal: str,
        source_stats: SourceStats,
        target_filters: Dict,
        target_codec: str,
        target_encoder: str,
        target_supports_hdr10: bool,
    ):
        """
        Build encoder-agnostic recommendations:
            - quality_mode: const_quality / capped_quality / target_bitrate
            - quality_index: abstract quantiser step (ladder entry)
            - wants_cap: desire to apply a bitrate rail
        """
        goal = goal or self.GOAL_BALANCED
        goal = goal if goal in self._GOALS else self.GOAL_BALANCED

        target_width = int(target_filters.get("target_width") or source_stats.width or 0)
        target_height = int(target_filters.get("target_height") or source_stats.height or 0)
        source_width = max(source_stats.width, 1)
        source_height = max(source_stats.height, 1)

        pixel_ratio = (target_width * target_height) / float(source_width * source_height)
        target_bucket = self._resolution_bucket(target_width, target_height)
        source_bucket = self._resolution_bucket(source_width, source_height)
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

        # Bitrate cap guardrail:
        #   Scale relative to the source and the pixel ratio.
        #   This only applies a floor when bitrate is unknown or when we are not downscaling.
        apply_floor = (source_bitrate is None) or pixel_ratio >= 1
        floor_br = self._floor_bitrate(goal, target_bucket)
        target_cap = self._target_cap(
            goal,
            source_bitrate,
            pixel_ratio,
            source_stats.is_hdr,
            apply_floor,
            floor_br,
        )

        # Downscale guardrail for low-bitrate sources:
        #   When Prefer Quality is downscaling a weak source, switch to capped mode to avoid runaway constqp.
        low_bitrate_threshold = self._LOW_BITRATE_THRESHOLDS.get(source_bucket, 0)

        # Pick quality mode/index and record any downgrade reason.
        quality_mode = self.QUALITY_CONST if goal == self.GOAL_PREFER_QUALITY else self.QUALITY_CAPPED
        downgraded_reason = None
        if (
            quality_mode == self.QUALITY_CONST
            and pixel_ratio < 1
            and source_bitrate
            and source_bitrate <= low_bitrate_threshold
        ):
            quality_mode = self.QUALITY_CAPPED
            downgraded_reason = "low_bitrate_downscale"

        # HDR handling guardrails:
        #   Flag HDR outputs that are likely to band.
        #   That happens when there is no HDR10 path or when the HDR source is only 8-bit.
        hdr_output_limited = False
        hdr_output_limited_reason = None
        if source_stats.is_hdr:
            if target_supports_hdr10 is False:
                hdr_output_limited = True
                hdr_output_limited_reason = "hdr_output_not_supported"
            elif source_stats.bit_depth and source_stats.bit_depth <= 8:
                hdr_output_limited = True
                hdr_output_limited_reason = "hdr_8bit_source"

        # Base params per goal:
        #   Prefer Quality uses fixed CQ/QP values.
        #   The other goals pull from codec/goal/resolution ladders and fall back to safe defaults when needed.
        if goal == self.GOAL_PREFER_QUALITY:
            quality_index = 17 if source_stats.is_hdr else 19
        else:
            quality_index = self._base_quality_index(target_codec, goal, source_stats.is_hdr, target_bucket)
            if quality_index is None:
                quality_index = 24 if not source_stats.is_hdr else 22
                if goal == self.GOAL_PREFER_COMPRESSION:
                    quality_index = 30 if not source_stats.is_hdr else 28

        # Apply re-encode bias and HDR banding sensitivity adjustments.
        cq_offset = self._reencode_cq_offset(source_stats.codec_name, target_codec)
        if quality_index is not None and quality_mode != self.QUALITY_CONST:
            quality_index = max(0, quality_index + cq_offset)

        # HDR-limited output (no HDR10 path or 8-bit HDR):
        #   Lower CQ/QP a bit to reduce the chance of banding.
        if hdr_output_limited and quality_index is not None:
            quality_index = max(0, quality_index - 2)

        # Build bitrate rails (maxrate/bufsize) only when caps make sense for the chosen mode/scale.
        wants_cap = bool(target_cap) and quality_mode in (self.QUALITY_CAPPED, self.QUALITY_TARGET)
        if downgraded_reason and target_cap:
            wants_cap = True

        # Build bitrate params for VBR modes using the derived guardrail cap.
        maxrate = int(target_cap) if target_cap and wants_cap else None
        bufsize = int(target_cap * 2.0) if target_cap and wants_cap else None

        # Confidence tracking and adjustments:
        #   Mark low confidence when HDR output is limited or probe stats are weak.
        #   Ease CQ when confidence is low.
        confidence_level = source_stats.confidence_level
        confidence = confidence_level != "low"
        confidence_notes = list(source_stats.confidence_notes)
        if hdr_output_limited:
            confidence = False
            confidence_level = "low"
            if hdr_output_limited_reason:
                confidence_notes.append(hdr_output_limited_reason)
            if hdr_output_limited_reason == "hdr_output_not_supported":
                logger.info("HDR source detected but target encoder lacks HDR10 output; applying conservative settings.")
        if not confidence:
            if quality_index is not None:
                quality_index = max(0, quality_index - 2)
        if confidence_level != "high":
            logger.info("Smart output target low confidence: level=%s reasons=%s", confidence_level, confidence_notes)

        master_display = None
        max_cll = None
        try:
            hdr_md = self.probe.get_hdr_static_metadata()
            master_display = hdr_md.get("master_display")
            max_cll = hdr_md.get("max_cll")
        except Exception:
            master_display = None
            max_cll = None

        recommendation = {
            "goal": goal,
            "quality_mode": quality_mode,
            "quality_index": quality_index,
            "wants_cap": wants_cap,
            "maxrate": maxrate,
            "bufsize": bufsize,
            "target_cap": target_cap,
            "pixel_ratio": pixel_ratio,
            "downgraded_reason": downgraded_reason,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "confidence_notes": confidence_notes,
            "target_resolution": {"width": target_width, "height": target_height},
            "source_resolution": {"width": source_stats.width, "height": source_stats.height},
            "hdr": {
                "is_hdr": source_stats.is_hdr,
                "bit_depth": source_stats.bit_depth,
                "output_supported": not hdr_output_limited,
                "master_display": master_display,
                "max_cll": max_cll,
            },
            "target_supports_hdr10": target_supports_hdr10,
            "source_bucket": source_bucket,
            "target_bucket": target_bucket,
        }
        logger.info(
            "Smart output target recommendation: goal=%s quality_mode=%s quality_index=%s cap=%s maxrate=%s bufsize=%s confidence=%s notes=%s",
            recommendation.get("goal"),
            recommendation.get("quality_mode"),
            recommendation.get("quality_index"),
            recommendation.get("target_cap"),
            recommendation.get("maxrate"),
            recommendation.get("bufsize"),
            recommendation.get("confidence"),
            recommendation.get("confidence_notes"),
        )

        return recommendation
