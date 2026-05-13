use std::path::PathBuf;
use std::process::Command;
use std::sync::mpsc::Sender;

use crate::event::Event;
use crate::state::VoiceLine;

const VOICE_DIR: &str = "../butterbot_voice";

/// Plays a voice clip in a background thread. Sends `Event::AudioFinished` when done.
/// The voice worker is already stopped before any audio plays (the orchestrator only
/// plays clips in non-Listening states), so no muting is needed here.
pub fn play(line: VoiceLine, tx: Sender<Event>) {
    let path = voice_path(&line);
    std::thread::spawn(move || {
        play_blocking(&path);
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
