"""RGB frame and H.264 playback encoding for interactive world models."""
import logging
import shutil
import struct
import subprocess
import zlib


def png_rgb(rgb, width, height):
    if len(rgb) != width * height * 3:
        raise ValueError("wrong RGB frame size")

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind+data) & 0xffffffff)

    scanlines = b"".join(b"\0" + rgb[y*width*3:(y+1)*width*3] for y in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(scanlines, 3)) + chunk(b"IEND", b""))


def find_video_encoder():
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        try:
            codecs = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True,
                                    check=True, timeout=5).stdout.splitlines()
            if any(line.split()[1:2] == [b"libx264"] for line in codecs):
                return ffmpeg
        except (OSError, subprocess.SubprocessError):
            pass
    return None


def encode_clip(ffmpeg, rgb, media, path):
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-video_size", f"{media['width']}x{media['height']}", "-framerate", str(media["fps"]), "-i", "pipe:0",
               "-filter_threads", "2", "-vf", "scale=out_color_matrix=bt709", "-c:v", "libx264", "-threads", "4",
               "-preset", "veryfast", "-crf", "18", "-profile:v", "main", "-pix_fmt", "yuv420p",
               "-colorspace", "bt709", "-movflags", "+faststart", str(path)]
    try:
        subprocess.run(command, input=rgb, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                       check=True, timeout=60)
        return True
    except (OSError, subprocess.SubprocessError) as exc:
        logging.getLogger(__name__).warning("MP4 encoding failed for %s, saving PNG frames: %s", path, exc)
        path.unlink(missing_ok=True)
        return False
