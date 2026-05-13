use std::path::PathBuf;
use std::process::Command;
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};

use crate::event::Event;
use crate::state::VoiceLine;
use crate::workers::VoiceWorker;

const VOICE_DIR: &str = "../butterbot_voice";

/// Plays a voice clip in a background thread.
///
/// Pauses the voice worker before playback starts and resumes it after, so the
/// robot cannot hear and transcribe its own audio. Sends `Event::AudioFinished`
/// once playback is complete and the worker has been resumed.
pub fn play(line: VoiceLine, voice: Arc<Mutex<VoiceWorker>>, tx: Sender<Event>) {
    let path = voice_path(&line);
    std::thread::spawn(move || {
        voice.lock().unwrap().pause();
        play_blocking(&path);
        voice.lock().unwrap().resume();
        tx.send(Event::AudioFinished).ok();
    });
}

fn voice_path(line: &VoiceLine) -> PathBuf {
    PathBuf::from(VOICE_DIR).join(line.filename())
}

fn play_blocking(path: &PathBuf) {
    let path_str = path.to_string_lossy();

    #[cfg(target_os = "macos")]
    let result = Command::new("afplay").arg(path_str.as_ref()).status();

    #[cfg(not(target_os = "macos"))]
    let result = Command::new("mpg123")
        .args(["-q", path_str.as_ref()])
        .status();

    if let Err(e) = result {
        eprintln!("[Audio] Failed to play {:?}: {}", path, e);
    }
}
