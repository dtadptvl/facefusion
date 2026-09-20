#!/usr/bin/env python3
"""
Python standard library script to generate an original minimal opaque 1024x1024 PNG
app icon and asset catalog entries for iFaceFusion without external dependencies.
"""

import os
import math
import zlib
import struct
import json

def create_png_rgb(width, height, rgb_pixels):
    """
    Encodes raw RGB pixel bytearray (width * height * 3) into an opaque PNG format.
    Color type 2 (RGB), 8-bit depth, no alpha channel.
    """
    raw_data = bytearray()
    row_bytes = width * 3
    for y in range(height):
        raw_data.append(0)  # Filter type 0 (None)
        start = y * row_bytes
        raw_data.extend(rgb_pixels[start:start + row_bytes])

    compressed = zlib.compress(raw_data, 9)

    def chunk(tag, data):
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        return length + tag + data + crc

    png = bytearray(b"\x89PNG\r\n\x1a\n")
    # IHDR: width, height, bit_depth=8, color_type=2 (RGB), compression=0, filter=0, interlace=0
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png.extend(chunk(b"IHDR", ihdr_data))
    png.extend(chunk(b"IDAT", compressed))
    png.extend(chunk(b"IEND", b""))
    return bytes(png)

def render_icon(size=1024):
    """
    Renders an opaque minimal photo-editor icon with dark neutral slate background,
    purple gradient glow, camera aperture geometry, and dual identity arcs.
    """
    buf = bytearray(size * size * 3)
    cx, cy = (size - 1) / 2.0, (size - 1) / 2.0
    max_radius = size / 2.0

    for y in range(size):
        dy = y - cy
        row_offset = y * size * 3
        for x in range(size):
            dx = x - cx
            dist = math.sqrt(dx * dx + dy * dy)
            norm_dist = dist / max_radius

            # 1. Base dark background with radial purple/indigo gradient
            bg_factor = 1.0 - min(1.0, norm_dist * 0.9)
            # Base dark: #110E1B (17, 14, 27) blending to #1E1238 (30, 18, 56)
            r = int(17 + 25 * bg_factor)
            g = int(14 + 14 * bg_factor)
            b = int(27 + 45 * bg_factor)

            # 2. Glowing outer lens ring (radius ~ 340 to 390 pt)
            ring_center = size * 0.36
            ring_dist = abs(dist - ring_center)
            if ring_dist < 40:
                glow = math.exp(-(ring_dist ** 2) / 350.0)
                # Purple accent #9333EA / #A855F7 (168, 85, 247)
                r = min(255, int(r + 147 * glow))
                g = min(255, int(g + 51 * glow))
                b = min(255, int(b + 234 * glow))

            # 3. Inner dual fusion arcs (representing source and target faces)
            # Left arc center: cx - 60, cy; right arc center: cx + 60, cy
            d_left = math.sqrt((dx + size * 0.08) ** 2 + dy ** 2)
            d_right = math.sqrt((dx - size * 0.08) ** 2 + dy ** 2)

            arc_radius = size * 0.22
            arc_thickness = 18.0

            arc1_dist = abs(d_left - arc_radius)
            arc2_dist = abs(d_right - arc_radius)

            if arc1_dist < arc_thickness and dx < size * 0.12:
                intensity = math.exp(-(arc1_dist ** 2) / (arc_thickness * 8.0))
                r = min(255, int(r + 192 * intensity))
                g = min(255, int(g + 132 * intensity))
                b = min(255, int(b + 252 * intensity))

            if arc2_dist < arc_thickness and dx > -size * 0.12:
                intensity = math.exp(-(arc2_dist ** 2) / (arc_thickness * 8.0))
                r = min(255, int(r + 230 * intensity))
                g = min(255, int(g + 180 * intensity))
                b = min(255, int(b + 255 * intensity))

            # 4. Central focal spark (aperture center)
            if dist < size * 0.05:
                spark = math.exp(-(dist ** 2) / (size * 0.02) ** 2)
                r = min(255, int(r + 255 * spark))
                g = min(255, int(g + 240 * spark))
                b = min(255, int(b + 255 * spark))

            col_offset = row_offset + x * 3
            buf[col_offset] = r
            buf[col_offset + 1] = g
            buf[col_offset + 2] = b

    return create_png_rgb(size, size, buf)

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    assets_dir = os.path.join(root_dir, "iFaceFusion", "Resources", "Assets.xcassets")
    appicon_dir = os.path.join(assets_dir, "AppIcon.appiconset")
    accent_dir = os.path.join(assets_dir, "AccentColor.colorset")

    os.makedirs(appicon_dir, exist_ok=True)
    os.makedirs(accent_dir, exist_ok=True)

    # 1. Assets.xcassets root Contents.json
    with open(os.path.join(assets_dir, "Contents.json"), "w", encoding="utf-8") as f:
        json.dump({"info": {"author": "xcode", "version": 1}}, f, indent=2)

    # 2. Render and save 1024x1024 AppIcon PNG
    icon_png_path = os.path.join(appicon_dir, "AppIcon-1024.png")
    print(f"Generating 1024x1024 opaque PNG icon: {icon_png_path}...")
    png_bytes = render_icon(1024)
    with open(icon_png_path, "wb") as f:
        f.write(png_bytes)
    print(f"App icon written: {len(png_bytes):,} bytes.")

    # 3. AppIcon.appiconset Contents.json (iOS 18 universal single-size icon)
    appicon_meta = {
        "images": [
            {
                "filename": "AppIcon-1024.png",
                "idiom": "universal",
                "platform": "ios",
                "size": "1024x1024"
            }
        ],
        "info": {
            "author": "xcode",
            "version": 1
        }
    }
    with open(os.path.join(appicon_dir, "Contents.json"), "w", encoding="utf-8") as f:
        json.dump(appicon_meta, f, indent=2)

    # 4. AccentColor.colorset Contents.json (Purple accent #9333EA)
    accent_meta = {
        "colors": [
            {
                "color": {
                    "color-space": "srgb",
                    "components": {
                        "alpha": "1.000",
                        "blue": "0.918",
                        "green": "0.200",
                        "red": "0.576"
                    }
                },
                "idiom": "universal"
            }
        ],
        "info": {
            "author": "xcode",
            "version": 1
        }
    }
    with open(os.path.join(accent_dir, "Contents.json"), "w", encoding="utf-8") as f:
        json.dump(accent_meta, f, indent=2)

    print("App icon assets generated successfully.")

if __name__ == "__main__":
    main()
