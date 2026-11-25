"""
Non-blocking audio alarm controller using pygame.mixer.
Automatically synthesizes a clean WAV alert tone if no asset exists.
"""

import math
from pathlib import Path
import struct
import wave
import pygame


def ensure_alarm_wav(wav_path: Path, frequency_hz: int = 880, duration_sec: float = 0.6) -> Path:
    """Generates a pulsed two-tone alert WAV file if the file does not already exist."""
    if wav_path.exists():
        return wav_path

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 44100
    num_samples = int(sample_rate * duration_sec)
    amplitude = 24000  # Out of 32767 for 16-bit PCM

    with wave.open(str(wav_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)

        frames = bytearray()
        for i in range(num_samples):
            t = i / sample_rate
            # Alternate between base frequency and +220 Hz every 0.15s for urgency
            freq = frequency_hz if int(t / 0.15) % 2 == 0 else frequency_hz + 220
            # Apply short fade envelope to prevent audio clicks
            env = min(1.0, i / 500.0, (num_samples - i) / 500.0)
            sample = int(amplitude * env * math.sin(2.0 * math.pi * freq * t))
            frames.extend(struct.pack("<h", sample))

        wf.writeframes(bytes(frames))

    return wav_path


class AlarmManager:
    """Manages non-blocking looped audio playback without spawning duplicate instances."""

    def __init__(self, sound_path: Path, fallback_freq_hz: int = 880):
        self.sound_path = ensure_alarm_wav(sound_path, frequency_hz=fallback_freq_hz)
        self._is_playing = False
        self._audio_available = False

        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
            self._sound = pygame.mixer.Sound(str(self.sound_path))
            self._audio_available = True
        except pygame.error as exc:
            print(f"[WARNING] Audio device unavailable ({exc}). Falling back to visual alerts only.")
            self._sound = None

    def start(self) -> None:
        """Starts looped alarm playback if not already active."""
        if not self._is_playing:
            self._is_playing = True
            if self._audio_available and self._sound is not None:
                self._sound.play(loops=-1)

    def stop(self) -> None:
        """Stops alarm playback immediately."""
        if self._is_playing:
            self._is_playing = False
            if self._audio_available and self._sound is not None:
                self._sound.stop()

    def cleanup(self) -> None:
        """Stops audio and quits pygame.mixer cleanly."""
        self.stop()
        if self._audio_available:
            pygame.mixer.quit()