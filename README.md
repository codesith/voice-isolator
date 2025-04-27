# 🎤 Voice Isolator

A real-time voice isolation system focused on **your personal voice**, running on macOS.  
Outputs clean voice audio to a **virtual microphone (BlackHole)** usable by Zoom, Discord, Teams, etc.

---

## 📦 Project Structure
- `src/train_profile.py` — Record your voice and generate a voice profile
- `src/run_isolator.py` — Run live voice isolation, outputting to a virtual mic
- `models/` — Stores your personal voice profile
- `config.yaml` — Device settings and tuning parameters

---

## 🚀 Getting Started

### 1. Install BlackHole (macOS Virtual Mic)

```bash
brew install blackhole-2ch
