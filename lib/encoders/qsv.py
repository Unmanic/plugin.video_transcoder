#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    plugins.qsv.py

    Written by:               Josh.5 <jsunnex@gmail.com>
    Date:                     08 Jun 2022, (8:14 AM)

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
"""
Notes:
    - Listing available encoder options:
        ffmpeg -h encoder=h264_qsv
        ffmpeg -h encoder=hevc_qsv
        ffmpeg -h encoder=av1_qsv
    - Good breakdown on FFmpeg general args for QSV HW accel: 
        https://gist.github.com/jackleaks/776d2de2688d238c95ed7eafb3d5bae8
"""


class QsvEncoder:

    def __init__(self, settings):
        self.settings = settings

    def provides(self):
        return {
            "h264_qsv": {
                "codec": "h264",
                "label": "QSV - h264_qsv",
            },
            "hevc_qsv": {
                "codec": "hevc",
                "label": "QSV - hevc_qsv",
            },
            "av1_qsv":  {
                "codec": "av1",
                "label": "QSV - av1_qsv",
            },
        }

    def options(self):
        return {
            "qsv_decoding_method":            "cpu",
            "qsv_preset":                     "slow",
            "qsv_tune":                       "film",
            "qsv_encoder_ratecontrol_method": "LA_ICQ",
            "qsv_constant_quantizer_scale":   "25",
            "qsv_constant_quality_scale":     "23",
            "qsv_average_bitrate":            "5",
        }

    def generate_default_args(self):
        """
        Generate a list of args for using a QSV decoder

        :param settings:
        :return:
        """
        # Encode only (no decoding)
        #   REF: https://trac.ffmpeg.org/wiki/Hardware/QuickSync#Transcode
        generic_kwargs = {
            "-init_hw_device":   "qsv=hw",
            "-filter_hw_device": "hw",
        }
        advanced_kwargs = {}
        # Check if we are using a HW accelerated decoder> Modify args as required
        if self.settings.get_setting('qsv_decoding_method') in ['qsv']:
            generic_kwargs = {
                "-hwaccel":               "qsv",
                "-hwaccel_output_format": "qsv",
                "-init_hw_device":        "qsv=hw",
                "-filter_hw_device":      "hw",
            }
        return generic_kwargs, advanced_kwargs

    def generate_filtergraphs(self, settings, has_sw_filters, hw_smart_filters, target_fmt="nv12"):
        """
        Generate the required filter for enabling QSV HW acceleration

        :return:
        """
        generic_kwargs = {}
        advanced_kwargs = {}
        hw_filter_args = []
        sw_filter_prefix_args = []
        sw_filter_suffix_args = []

        # Check if we are decoding with QSV
        hw_decode = settings.get_setting('qsv_decoding_method') in ['qsv']
        # Check software format to use
        sw_fmt = "p010le" if target_fmt == "p010" else "nv12"

        # If we have SW filters:
        if has_sw_filters:
            # If we have SW filters and HW decode is enabled, make decoder produce SW frames
            if hw_decode:
                generic_kwargs['-hwaccel_output_format'] = sw_fmt
            # Add filter to upload software frames to QSV for QSV filters
            # Note, format conversion (if any - eg yuv422p10le -> p010le) happens after the software filters.
            # If a user applies a custom software filter that does not support the pix_fmt, then will need to prefix it with 'format=p010le'
            sw_filter_suffix_args.append(
                f'format={sw_fmt}|qsv,hwupload=extra_hw_frames=64,format=qsv,vpp_qsv=format={target_fmt}')
        # If we have no software filters:
        else:
            # Add hwupload filter that can handle when the frame was decoded in software or hardware
            hw_filter_args.append(f'format={sw_fmt}|qsv,hwupload=extra_hw_frames=64,format=qsv,vpp_qsv=format={target_fmt}')

        # Loop over any HW smart filters to be applied and add them as required.
        for smart_filter in hw_smart_filters:
            if smart_filter.get('scale'):
                scale_values = smart_filter.get('scale')
                hw_filter_args.append('scale_qsv=w={}:h=-1'.format(scale_values["width"]))

        # Return built args
        return {
            "generic_kwargs":        generic_kwargs,
            "advanced_kwargs":       advanced_kwargs,
            "hw_filter_args":        hw_filter_args,
            "sw_filter_prefix_args": sw_filter_prefix_args,
            "sw_filter_suffix_args": sw_filter_suffix_args,
        }

    def encoder_details(self, encoder):
        provides = self.provides()
        return provides.get(encoder, {})

    def args(self, stream_id):
        stream_encoding = []

        # Use defaults for basic mode
        if self.settings.get_setting('mode') in ['basic']:
            defaults = self.options()
            # Use default LA_ICQ mode
            stream_encoding += [
                '-preset', str(defaults.get('qsv_preset')),
                '-global_quality', str(defaults.get('qsv_constant_quality_scale')), '-look_ahead', '1',
            ]
            return stream_encoding

        # Add the preset and tune
        if self.settings.get_setting('qsv_preset'):
            stream_encoding += ['-preset', str(self.settings.get_setting('qsv_preset'))]
        if self.settings.get_setting('qsv_tune') and self.settings.get_setting('qsv_tune') != 'auto':
            stream_encoding += ['-tune', str(self.settings.get_setting('qsv_tune'))]

        if self.settings.get_setting('qsv_encoder_ratecontrol_method'):
            if self.settings.get_setting('qsv_encoder_ratecontrol_method') in ['CQP', 'LA_ICQ', 'ICQ']:
                # Configure QSV encoder with a quality-based mode
                if self.settings.get_setting('qsv_encoder_ratecontrol_method') == 'CQP':
                    # Set values for constant quantizer scale
                    stream_encoding += [
                        '-q', str(self.settings.get_setting('qsv_constant_quantizer_scale')),
                    ]
                elif self.settings.get_setting('qsv_encoder_ratecontrol_method') in ['LA_ICQ', 'ICQ']:
                    # Set the global quality
                    stream_encoding += [
                        '-global_quality', str(self.settings.get_setting('qsv_constant_quality_scale')),
                    ]
                    # Set values for constant quality scale
                    if self.settings.get_setting('qsv_encoder_ratecontrol_method') == 'LA_ICQ':
                        # Add lookahead
                        stream_encoding += [
                            '-look_ahead', '1',
                        ]
            else:
                # Configure the QSV encoder with a bitrate-based mode
                # Set the max and average bitrate (used by all bitrate-based modes)
                stream_encoding += [
                    '-b:v:{}'.format(stream_id), '{}M'.format(self.settings.get_setting('qsv_average_bitrate')),
                ]
                if self.settings.get_setting('qsv_encoder_ratecontrol_method') == 'LA':
                    # Add lookahead
                    stream_encoding += [
                        '-look_ahead', '1',
                    ]
                elif self.settings.get_setting('qsv_encoder_ratecontrol_method') == 'CBR':
                    # Add 'maxrate' with the same value to make CBR mode
                    stream_encoding += [
                        '-maxrate', '{}M'.format(self.settings.get_setting('qsv_average_bitrate')),
                    ]
        return stream_encoding

    def __set_default_option(self, select_options, key, default_option=None):
        """
        Sets the default option if the currently set option is not available

        :param select_options:
        :param key:
        :return:
        """
        available_options = []
        for option in select_options:
            available_options.append(option.get('value'))
            if not default_option:
                default_option = option.get('value')
        if self.settings.get_setting(key) not in available_options:
            self.settings.set_setting(key, default_option)

    def get_qsv_decoding_method_form_settings(self):
        values = {
            "label":          "Enable HW Accelerated Decoding",
            "description":    "Warning: Ensure your device supports decoding the source video codec or it will fail.\n"
                              "This enables full hardware transcode with QSV, using only GPU memory for the entire video transcode.\n"
                              "If filters are configured in the plugin, decoder will output NV12 or P010LE software surfaces to\n"
                              "those filters which will be slightly slower.",
            "sub_setting":    True,
            "input_type":     "select",
            "select_options": [
                {
                    "value": "cpu",
                    "label": "Disabled - Use CPU to decode of video source (provides best compatibility)",
                },
                {
                    "value": "qsv",
                    "label": "QSV - Enable QSV decoding",
                }
            ]
        }
        self.__set_default_option(values['select_options'], 'qsv_decoding_method', 'cpu')
        if self.settings.get_setting('mode') not in ['standard']:
            values["display"] = "hidden"
        return values

    def get_qsv_preset_form_settings(self):
        values = {
            "label":          "Encoder quality preset",
            "sub_setting":    True,
            "input_type":     "select",
            "select_options": [
                {
                    "value": "veryfast",
                    "label": "Very fast - Fastest setting, biggest quality drop",
                },
                {
                    "value": "faster",
                    "label": "Faster - Close to medium/fast quality, faster performance",
                },
                {
                    "value": "fast",
                    "label": "Fast",
                },
                {
                    "value": "medium",
                    "label": "Medium - Balanced performance and quality",
                },
                {
                    "value": "slow",
                    "label": "Slow",
                },
                {
                    "value": "slower",
                    "label": "Slower - Close to 'very slow' quality, faster performance",
                },
                {
                    "value": "veryslow",
                    "label": "Very Slow - Best quality",
                },
            ],
        }
        self.__set_default_option(values['select_options'], 'qsv_preset')
        if self.settings.get_setting('mode') not in ['standard']:
            values["display"] = "hidden"
        return values

    def get_qsv_tune_form_settings(self):
        values = {
            "label":          "Tune for a particular type of source or situation",
            "sub_setting":    True,
            "input_type":     "select",
            "select_options": [
                {
                    "value": "auto",
                    "label": "Disabled – Do not apply any tune",
                },
                {
                    "value": "film",
                    "label": "Film – use for high quality movie content; lowers deblocking",
                },
                {
                    "value": "animation",
                    "label": "Animation – good for cartoons; uses higher deblocking and more reference frames",
                },
                {
                    "value": "grain",
                    "label": "Grain – preserves the grain structure in old, grainy film material",
                },
                {
                    "value": "stillimage",
                    "label": "Still image – good for slideshow-like content",
                },
                {
                    "value": "fastdecode",
                    "label": "Fast decode – allows faster decoding by disabling certain filters",
                },
                {
                    "value": "zerolatency",
                    "label": "Zero latency – good for fast encoding and low-latency streaming",
                },
            ],
        }
        self.__set_default_option(values['select_options'], 'qsv_tune')
        if self.settings.get_setting('mode') not in ['standard']:
            values["display"] = "hidden"
        return values

    def get_qsv_encoder_ratecontrol_method_form_settings(self):
        values = {
            "label":          "Encoder ratecontrol method",
            "sub_setting":    True,
            "input_type":     "select",
            "select_options": [
                {
                    "value": "CQP",
                    "label": "CQP - Quality based mode using constant quantizer scale",
                },
                {
                    "value": "ICQ",
                    "label": "ICQ - Quality based mode using intelligent constant quality",
                },
                {
                    "value": "LA_ICQ",
                    "label": "LA_ICQ - Quality based mode using intelligent constant quality with lookahead",
                },
                {
                    "value": "VBR",
                    "label": "VBR - Bitrate based mode using variable bitrate",
                },
                {
                    "value": "LA",
                    "label": "LA - Bitrate based mode using VBR with lookahead",
                },
                {
                    "value": "CBR",
                    "label": "CBR - Bitrate based mode using constant bitrate",
                },
            ]
        }
        self.__set_default_option(values['select_options'], 'qsv_encoder_ratecontrol_method', default_option='LA_ICQ')
        if self.settings.get_setting('mode') not in ['standard']:
            values["display"] = "hidden"
        return values

    def get_qsv_constant_quantizer_scale_form_settings(self):
        # Lower is better
        values = {
            "label":          "Constant quantizer scale",
            "sub_setting":    True,
            "input_type":     "slider",
            "slider_options": {
                "min": 0,
                "max": 51,
            },
        }
        if self.settings.get_setting('mode') not in ['standard']:
            values["display"] = "hidden"
        if self.settings.get_setting('qsv_encoder_ratecontrol_method') != 'CQP':
            values["display"] = "hidden"
        return values

    def get_qsv_constant_quality_scale_form_settings(self):
        # Lower is better
        values = {
            "label":          "Constant quality scale",
            "sub_setting":    True,
            "input_type":     "slider",
            "slider_options": {
                "min": 1,
                "max": 51,
            },
        }
        if self.settings.get_setting('mode') not in ['standard']:
            values["display"] = "hidden"
        if self.settings.get_setting('qsv_encoder_ratecontrol_method') not in ['LA_ICQ', 'ICQ']:
            values["display"] = "hidden"
        return values

    def get_qsv_average_bitrate_form_settings(self):
        values = {
            "label":          "Bitrate",
            "sub_setting":    True,
            "input_type":     "slider",
            "slider_options": {
                "min":    1,
                "max":    20,
                "suffix": "M"
            },
        }
        if self.settings.get_setting('mode') not in ['standard']:
            values["display"] = "hidden"
        if self.settings.get_setting('qsv_encoder_ratecontrol_method') not in ['VBR', 'LA', 'CBR']:
            values["display"] = "hidden"
        return values
