from __future__ import annotations
import logging
import os
import tempfile
import threading
import warnings
from typing import Callable

from dotenv import load_dotenv
import telebot
from telebot import types, apihelper

from gtts import gTTS
from googletrans import Translator
import librosa
import soundfile as sf
import whisper  # type: ignore
from moviepy import (
    VideoFileClip,
    AudioFileClip,
    concatenate_videoclips,
    concatenate_audioclips,
)
from moviepy.video.fx.MultiplySpeed import MultiplySpeed

# ─────────────────── Config ─────────────────── #
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN missing in .env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("bot")

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU*")

# ─────────────────── Heavy Globals ─────────────────── #
WHISPER_MODEL = whisper.load_model("base")
TRANSLATOR = Translator()

# ─────────────────── Helpers ─────────────────── #

def change_audio_speed(src: str, dst: str, rate: float) -> float:
    y, sr = librosa.load(src, sr=None)
    y_out = librosa.effects.time_stretch(y, rate=rate)
    sf.write(dst, y_out, sr)
    return len(y_out) / sr


def compress_video(src: str, dst: str) -> None:
    clip = VideoFileClip(src)
    # Downscale & reduce bitrate to keep file <50 MB
    clip_resized = clip.resize(width=480)
    clip_resized.write_videofile(
        dst,
        codec="libx264",
        bitrate="700k",
        audio_codec="aac",
        audio_bitrate="96k",
        temp_audiofile="temp-audio.m4a",
        remove_temp=True,
        logger=None,
    )
    clip.close()

# ─────────────────── Core Pipeline ─────────────────── #

def process_video(path: str, progress: Callable[[str], None]) -> str:
    vid = VideoFileClip(path)
    prog = progress
    prog(f"🎬 Duration: {vid.duration:.2f}s")

    wav = tempfile.mktemp(suffix=".wav")
    vid.audio.write_audiofile(wav, logger=None)
    segments = WHISPER_MODEL.transcribe(wav)["segments"]

    v_clips, a_clips = [], []
    for i, s in enumerate(segments):
        st, e0 = s["start"], s["end"]
        if st >= vid.duration:
            break
        en = min(e0, vid.duration)
        dur = en - st
        if dur <= 0:
            continue
        eng = s["text"].strip()
        prog(f"\nSeg {i}: {st:.2f}-{en:.2f}s\nEN: {eng}")
        hin = TRANSLATOR.translate(eng, src="en", dest="hi").text.strip()
        prog(f"HI: {hin}")

        mp3 = tempfile.mktemp(suffix=".mp3")
        gTTS(hin, lang="hi").save(mp3)
        wav_in = tempfile.mktemp(suffix=".wav")
        y, sr = librosa.load(mp3, sr=None)
        sf.write(wav_in, y, sr)

        wav_fast = tempfile.mktemp(suffix=".wav")
        fast_len = change_audio_speed(wav_in, wav_fast, 1.10)
        prog(f"🔊 @1.10×: {fast_len:.2f}s")

        if fast_len <= dur:
            rate = fast_len / dur
            wav_adj = tempfile.mktemp(suffix=".wav")
            change_audio_speed(wav_fast, wav_adj, rate)
            a_clip = AudioFileClip(wav_adj)
            v_clip = vid.subclipped(st, en)
        else:
            slow = dur / fast_len
            a_clip = AudioFileClip(wav_fast)
            v_clip = vid.subclipped(st, en)
            if slow < 1:
                v_clip = v_clip.with_effects([MultiplySpeed(factor=slow)])
        v_clips.append(v_clip)
        a_clips.append(a_clip)
        for f in (mp3, wav_in):
            if os.path.exists(f):
                os.remove(f)

    final_v = concatenate_videoclips(v_clips, method="compose")
    final_a = concatenate_audioclips(a_clips)
    final_v = final_v.with_audio(final_a)

    out = tempfile.mktemp(suffix="_hi.mp4")
    final_v.write_videofile(
        out,
        codec="libx264",
        audio_codec="aac",
        audio_bitrate="192k",
        temp_audiofile="temp-audio.m4a",
        remove_temp=True,
        logger=None,
    )

    os.remove(wav)
    vid.close(); final_v.close()
    return out

# ─────────────────── Bot Handlers ─────────────────── #

bot = telebot.TeleBot(BOT_TOKEN, parse_mode="HTML")
MAX_SIZE = 49 * 1024 * 1024  # Telegram bot upload limit ≈50 MB

@bot.message_handler(commands=["start", "help"])
def start_cmd(msg):
    bot.reply_to(msg, "Send me an MP4 and I’ll return a Hindi‑dubbed version (≤50 MB)")

@bot.message_handler(content_types=["video", "document"])
def handle_video(msg):
    cid = msg.chat.id
    file_id = None
    if msg.content_type == "video":
        file_id = msg.video.file_id
    elif msg.document and msg.document.mime_type.startswith("video/"):
        file_id = msg.document.file_id
    else:
        bot.reply_to(msg, "⚠️ Please send an MP4 video.")
        return

    status = bot.reply_to(msg, "⬇️ Downloading…")
    try:
        info = bot.get_file(file_id)
        data = bot.download_file(info.file_path)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(data)
            local = tmp.name
    except Exception as e:
        logger.exception("Download error")
        bot.edit_message_text(f"❌ Download error: {e}", cid, status.id)
        return

    bot.edit_message_text("⚙️ Processing…", cid, status.id)

    def progress(text):
        bot.send_message(cid, text)

    def worker():
        out = None
        try:
            out = process_video(local, progress)
            size = os.path.getsize(out)
            if size > MAX_SIZE:
                comp = tempfile.mktemp(suffix="_small.mp4")
                progress("⚠️ Compressing large file…")
                compress_video(out, comp)
                os.remove(out)
                out = comp
            bot.send_chat_action(cid, "upload_video")
            with open(out, "rb") as f:
                try:
                    bot.send_document(cid, f, caption="Hindi‑dubbed ✔️")
                except apihelper.ApiTelegramException as e:
                    bot.send_message(cid, f"❌ Telegram error: {e}")
        except Exception as e:
            logger.exception("Processing failed")
            bot.send_message(cid, f"❌ Error: {e}")
        finally:
            for p in (local, out):
                if p and os.path.exists(p):
                    os.remove(p)
            bot.delete_message(cid, status.id)

    threading.Thread(target=worker, daemon=True).start()

# ─────────────────── Main ─────────────────── #
if __name__ == "__main__":
    logger.info("Bot polling…")
    bot.infinity_polling(skip_pending=True)
